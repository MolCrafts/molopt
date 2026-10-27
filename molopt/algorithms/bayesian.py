"""Bayesian optimization.

MolOpt owns the search space, the noise policy, pending-point handling, and
which candidates may be proposed. Gaussian-process inference and acquisition
functions come from BoTorch. Those imports happen when this class is
constructed, not when the rest of MolOpt is imported.

Construct it with ``noise=0.0`` when the evaluation is known to be noiseless.
A missing ``stderr`` is unknown noise and is rejected unless ``noise`` was set.
The small variance used when that noise is zero is numerical jitter for the
Cholesky factor, not a claim that an unobserved uncertainty was zero.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from molopt.errors import CapabilityError, MolOptError, StateError, TellError, ValidationError
from molopt.optimizer import Candidate, DuplicatePolicy, Optimizer
from molopt.problem import Direction, Problem
from molopt.result import Result, Status
from molopt.space import Choice, Constant, FiniteSpace, Integer, Real, Space, freeze, pythonize

_JITTER = 1e-12


class BayesianOptimization(Optimizer):
    """Single- and multi-objective Bayesian optimization."""

    algorithm_name = "bayesian"

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        n_initial: int,
        noise: float | None = None,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
        max_discrete: int = 4096,
        max_choice_combinations: int = 16,
        num_restarts: int = 5,
        raw_samples: int = 32,
        mc_samples: int = 64,
        opt_maxiter: int = 100,
    ) -> None:
        if not isinstance(problem, Problem):
            raise ValidationError("problem must be a Problem")
        self.n_initial = _at_least(n_initial, 2, "n_initial")
        self.noise = _optional_noise(noise)
        self.max_discrete = _at_least(max_discrete, 1, "max_discrete")
        self.max_choice_combinations = _at_least(
            max_choice_combinations, 1, "max_choice_combinations"
        )
        self.num_restarts = _at_least(num_restarts, 1, "num_restarts")
        self.raw_samples = _at_least(raw_samples, 1, "raw_samples")
        self.mc_samples = _at_least(mc_samples, 2, "mc_samples")
        self.opt_maxiter = _at_least(opt_maxiter, 1, "opt_maxiter")
        self._mode = _classify(
            problem.space,
            max_discrete=self.max_discrete,
            max_choice_combinations=self.max_choice_combinations,
        )
        self._bo = _load_bo()
        super().__init__(problem, seed=seed, duplicates=duplicates)
        self._encoder = _make_encoder(problem.space, self._mode)
        if self._encoder.dim < 1:
            raise CapabilityError(
                "BayesianOptimization needs at least one varying coordinate"
            )
        torch = self._bo.torch
        self._torch_gen = torch.Generator(device="cpu")
        self._torch_gen.manual_seed(self.seed & ((1 << 63) - 1))
        self._initial_queue: list[dict[str, Any]] | None = None

    def _ask(self, n: int) -> list[Candidate]:
        open_slots = self.n_initial - self._completed() - len(self._pending)
        if open_slots > 0:
            return self._ask_initial(min(n, open_slots))
        if self._completed() < 2:
            return []
        return self._ask_model(n)

    def _validate_extra(self, result: Result) -> None:
        if result.status is not Status.COMPLETED or result.values is None:
            return
        for name in self.problem.measured_names:
            known = result.stderr is not None and name in result.stderr
            if not known and self.noise is None:
                raise TellError(
                    f"noise for {name!r} is unknown. Pass stderr or construct "
                    "BayesianOptimization with an explicit noise= value. "
                    "noise=0.0 asserts a noiseless evaluation."
                )

    def _ask_initial(self, n: int) -> list[Candidate]:
        chosen: list[Candidate] = []
        seen = set(self._blocked)
        if self._mode == "continuous":
            self._ensure_initial_design()
        attempts = 0
        while len(chosen) < n and attempts < 1000 * n:
            attempts += 1
            params = self._next_initial()
            if params is None:
                break
            identity = self.problem.space.identity(params)
            if identity in seen and self.duplicates is DuplicatePolicy.SKIP:
                continue
            seen.add(identity)
            chosen.append((params, None, {}))
        return chosen

    def _ensure_initial_design(self) -> None:
        if self._initial_queue is not None:
            return
        from scipy.stats import qmc

        encoder = self._encoder
        assert isinstance(encoder, BoxEncoder)
        engine = qmc.LatinHypercube(
            d=len(encoder.continuous),
            scramble=True,
            strength=1,
            optimization=None,
            seed=self.seed,
        )
        unit = np.asarray(engine.random(self.n_initial), dtype=np.float64)
        self._initial_queue = [
            encoder.decode_continuous(unit[row], {}) for row in range(self.n_initial)
        ]

    def _next_initial(self) -> dict[str, Any] | None:
        if self._mode == "continuous":
            assert self._initial_queue is not None
            if not self._initial_queue:
                return None
            return self._initial_queue.pop(0)
        return self.problem.space.sample(1, self._rng)[0]

    def _ask_model(self, n: int) -> list[Candidate]:
        torch = self._bo.torch
        train_x, train_y, train_var, pending = self._training_tensors()
        if train_x.shape[0] < 2:
            return []
        seed = self._draw_seed()

        def propose() -> list[dict[str, Any]]:
            model = self._fit(train_x, train_y, train_var)
            if self._mode in {"finite", "finite_product"}:
                return self._propose_library(model, train_x, pending, n, seed)
            if self._mode == "mixed":
                return self._propose_mixed(model, train_x, pending, n, seed)
            return self._propose_box(model, train_x, pending, n, seed)

        try:
            params = self._under_seed(seed, propose)
        except Exception as exc:
            raise MolOptError(f"Bayesian optimization proposal failed: {exc}") from exc
        chosen: list[Candidate] = []
        seen = set(self._blocked)
        for candidate in params:
            validated = self.problem.space.validate(candidate)
            identity = self.problem.space.identity(validated)
            if identity in seen and self.duplicates is DuplicatePolicy.SKIP:
                continue
            seen.add(identity)
            chosen.append((validated, None, {}))
            if len(chosen) == n:
                break
        return chosen

    def _completed(self) -> int:
        return sum(result.status is Status.COMPLETED for result in self._results.values())

    def _training_tensors(self):
        torch = self._bo.torch
        dtype = torch.float64
        rows: list[list[float]] = []
        outcomes: list[list[float]] = []
        variances: list[list[float]] = []
        for trial_id, result in self._results.items():
            if result.status is not Status.COMPLETED or result.values is None:
                continue
            trial = self._trials[trial_id]
            rows.append(self._encoder.encode(trial.params))
            outcomes.append(self._outcome_row(result))
            variances.append(self._variance_row(result))
        pending_rows = [self._encoder.encode(trial.params) for trial in self._pending.values()]
        train_x = torch.tensor(rows, dtype=dtype)
        train_y = torch.tensor(outcomes, dtype=dtype)
        train_var = torch.tensor(variances, dtype=dtype)
        pending = torch.tensor(pending_rows, dtype=dtype) if pending_rows else None
        return train_x, train_y, train_var, pending

    def _outcome_row(self, result: Result) -> list[float]:
        assert result.values is not None
        row: list[float] = []
        for objective in self.problem.objectives:
            number = float(result.values[objective.name])
            if objective.direction is Direction.MINIMIZE:
                number = -number
            row.append(number)
        for constraint in self.problem.constraints:
            row.append(float(result.values[constraint.name]))
        return row

    def _variance_row(self, result: Result) -> list[float]:
        assert result.values is not None
        names = [item.name for item in self.problem.objectives]
        names.extend(item.name for item in self.problem.constraints)
        row = []
        for name in names:
            if result.stderr is not None and name in result.stderr:
                sigma = float(result.stderr[name])
            else:
                assert self.noise is not None
                sigma = self.noise
            row.append(max(sigma * sigma, _JITTER))
        return row

    def _fit(self, train_x, train_y, train_var):
        bo = self._bo
        models = []
        for index in range(train_y.shape[-1]):
            model = bo.SingleTaskGP(
                train_x,
                train_y[:, index : index + 1],
                train_var[:, index : index + 1],
                outcome_transform=bo.Standardize(m=1),
            )
            models.append(model)
        if len(models) == 1:
            model = models[0]
            mll = bo.ExactMarginalLogLikelihood(model.likelihood, model)
        else:
            model = bo.ModelListGP(*models)
            mll = bo.SumMarginalLogLikelihood(model.likelihood, model)
        bo.fit_gpytorch_mll(mll)
        return model

    def _acquisition(self, model, train_x, train_y, pending, seed: int):
        bo = self._bo
        torch = bo.torch
        sampler = bo.SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
            seed=seed,
        )
        constraints = self._constraint_callables()
        pending_x = pending if pending is not None and pending.shape[0] else None
        if self.problem.n_objectives == 1:
            objective = None
            if train_y.shape[-1] > 1:
                objective = bo.GenericMCObjective(lambda samples, X: samples[..., 0])
            return bo.qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_x,
                sampler=sampler,
                objective=objective,
                constraints=constraints,
                X_pending=pending_x,
                prune_baseline=True,
            )
        objective = bo.IdentityMCMultiOutputObjective(
            outcomes=list(range(self.problem.n_objectives))
        )
        reference = self._reference(train_y)
        return bo.qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference.tolist(),
            X_baseline=train_x,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=pending_x,
            prune_baseline=True,
        )

    def _constraint_callables(self):
        torch = self._bo.torch
        if not self.problem.constraints:
            return None
        callables = []
        for offset, constraint in enumerate(self.problem.constraints):
            index = self.problem.n_objectives + offset
            lower = constraint.lower
            upper = constraint.upper

            def function(samples, index=index, lower=lower, upper=upper):
                column = samples[..., index]
                pieces = []
                if lower is not None:
                    pieces.append(lower - column)
                if upper is not None:
                    pieces.append(column - upper)
                violation = pieces[0] if len(pieces) == 1 else torch.maximum(*pieces)
                return violation - 1e-12

            callables.append(function)
        return callables

    def _reference(self, train_y):
        objectives = train_y[:, : self.problem.n_objectives]
        feasible = []
        for trial_id, result in self._results.items():
            if result.status is Status.COMPLETED and result.values is not None:
                feasible.append(self.problem.feasible(result.values))
        mask = self._bo.torch.tensor(feasible, dtype=self._bo.torch.bool)
        chosen = objectives[mask] if bool(mask.any()) else objectives
        lowest = chosen.min(dim=0).values
        return lowest - 0.1 * (1.0 + lowest.abs())

    def _propose_box(self, model, train_x, pending, n: int, seed: int) -> list[dict[str, Any]]:
        torch = self._bo.torch
        _, train_y, train_var, _ = self._training_tensors()
        del train_var
        acq = self._acquisition(model, train_x, train_y, pending, seed)
        bounds = torch.zeros(2, self._encoder.dim, dtype=torch.float64)
        bounds[1] = 1.0
        candidates, _ = self._bo.optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=n,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": self.opt_maxiter, "seed": seed},
            sequential=True,
        )
        assert isinstance(self._encoder, BoxEncoder)
        return [
            self._encoder.decode_continuous(candidates[index].detach().tolist(), {})
            for index in range(candidates.shape[0])
        ]

    def _propose_library(self, model, train_x, pending, n: int, seed: int) -> list[dict[str, Any]]:
        torch = self._bo.torch
        encoder = self._encoder
        assert isinstance(encoder, LibraryEncoder)
        kept_params = []
        kept_rows = []
        for params, row in zip(encoder.candidates, encoder.features, strict=True):
            if self._is_blocked(params):
                continue
            kept_params.append(params)
            kept_rows.append(row)
        if not kept_rows:
            return []
        choices = torch.tensor(kept_rows, dtype=torch.float64)
        _, train_y, _, _ = self._training_tensors()
        acq = self._acquisition(model, train_x, train_y, pending, seed)
        q = min(n, choices.shape[0])
        selected, _ = self._bo.optimize_acqf_discrete(
            acq_function=acq,
            q=q,
            choices=choices,
            max_batch_size=2048,
            unique=True,
        )
        found = []
        for row in selected.detach():
            match = _match_row(row, choices)
            if match is not None:
                found.append(dict(kept_params[match]))
        return found

    def _propose_mixed(self, model, train_x, pending, n: int, seed: int) -> list[dict[str, Any]]:
        torch = self._bo.torch
        encoder = self._encoder
        assert isinstance(encoder, BoxEncoder)
        _, train_y, _, _ = self._training_tensors()
        local_pending = [] if pending is None else [row for row in pending]
        picked: list[dict[str, Any]] = []
        for _ in range(n):
            pending_x = torch.stack(local_pending) if local_pending else None
            acq = self._acquisition(model, train_x, train_y, pending_x, seed)
            best_value = -math.inf
            best_params: dict[str, Any] | None = None
            best_row: list[float] | None = None
            for cell in encoder.cells():
                params, row, value = self._optimize_cell(acq, cell, seed)
                if value > best_value:
                    best_value = value
                    best_params = params
                    best_row = row
            if best_params is None or best_row is None:
                break
            picked.append(best_params)
            local_pending.append(torch.tensor(best_row, dtype=torch.float64))
        return picked

    def _optimize_cell(self, acq, cell: dict[str, Any], seed: int):
        torch = self._bo.torch
        encoder = self._encoder
        assert isinstance(encoder, BoxEncoder)
        fixed = encoder.fixed_feature(cell)
        if self._bo.FixedFeatureAcquisitionFunction is None or not encoder.continuous:
            return self._shoot_cell(acq, cell)
        reduced = self._bo.FixedFeatureAcquisitionFunction(
            acq_function=acq,
            d=encoder.dim,
            columns=encoder.discrete_columns,
            values=fixed,
        )
        bounds = torch.zeros(2, len(encoder.continuous), dtype=torch.float64)
        bounds[1] = 1.0
        candidate, value = self._bo.optimize_acqf(
            acq_function=reduced,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options={"batch_limit": 5, "maxiter": self.opt_maxiter, "seed": seed},
            sequential=False,
        )
        unit = candidate.detach().reshape(-1).tolist()
        params = encoder.decode_continuous(unit, cell)
        row = encoder.encode(params)
        return params, row, float(value.reshape(-1)[0])

    def _shoot_cell(self, acq, cell: dict[str, Any]):
        torch = self._bo.torch
        encoder = self._encoder
        assert isinstance(encoder, BoxEncoder)
        count = self.raw_samples
        unit = torch.rand(
            count,
            len(encoder.continuous),
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        rows = []
        params_list = []
        for index in range(count):
            params = encoder.decode_continuous(unit[index].tolist(), cell)
            params_list.append(params)
            rows.append(encoder.encode(params))
        batch = torch.tensor(rows, dtype=torch.float64).unsqueeze(1)
        values = acq(batch).detach().reshape(-1)
        best = int(torch.argmax(values))
        return params_list[best], rows[best], float(values[best])

    def _draw_seed(self) -> int:
        torch = self._bo.torch
        drawn = torch.randint(0, 2**31 - 1, (1,), generator=self._torch_gen)
        return int(drawn.item())

    def _under_seed(self, seed: int, function):
        torch = self._bo.torch
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            return function()
        finally:
            torch.random.set_rng_state(state)

    def _config(self) -> dict[str, Any]:
        return {
            "n_initial": self.n_initial,
            "noise": self.noise,
            "max_discrete": self.max_discrete,
            "max_choice_combinations": self.max_choice_combinations,
            "num_restarts": self.num_restarts,
            "raw_samples": self.raw_samples,
            "mc_samples": self.mc_samples,
            "opt_maxiter": self.opt_maxiter,
            "mode": self._mode,
        }

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "config": self._config(),
            "initial_queue": self._initial_queue,
            "torch_generator": self._torch_gen.get_state().detach().cpu().tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if state.get("config") != self._config():
            raise StateError("checkpoint Bayesian optimization settings do not match")
        torch = self._bo.torch
        blob = state["torch_generator"]
        tensor = torch.tensor(blob, dtype=torch.uint8)
        self._torch_gen.set_state(tensor)
        queue = state.get("initial_queue")
        self._initial_queue = None if queue is None else [dict(item) for item in queue]


class _BoModules:
    def __init__(self, **modules: Any) -> None:
        self.__dict__.update(modules)


def _load_bo() -> _BoModules:
    try:
        import torch
        from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.acquisition.multi_objective.objective import (
            IdentityMCMultiOutputObjective,
        )
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls import SumMarginalLogLikelihood
        from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            'BayesianOptimization requires optional dependencies. '
            'Install them with: pip install "molopt[bo]"'
        ) from exc
    return _BoModules(
        torch=torch,
        FixedFeatureAcquisitionFunction=FixedFeatureAcquisitionFunction,
        qLogNoisyExpectedImprovement=qLogNoisyExpectedImprovement,
        qLogNoisyExpectedHypervolumeImprovement=qLogNoisyExpectedHypervolumeImprovement,
        IdentityMCMultiOutputObjective=IdentityMCMultiOutputObjective,
        GenericMCObjective=GenericMCObjective,
        fit_gpytorch_mll=fit_gpytorch_mll,
        SingleTaskGP=SingleTaskGP,
        ModelListGP=ModelListGP,
        Standardize=Standardize,
        optimize_acqf=optimize_acqf,
        optimize_acqf_discrete=optimize_acqf_discrete,
        SobolQMCNormalSampler=SobolQMCNormalSampler,
        SumMarginalLogLikelihood=SumMarginalLogLikelihood,
        ExactMarginalLogLikelihood=ExactMarginalLogLikelihood,
    )


def _classify(space: Space | FiniteSpace, *, max_discrete: int, max_choice_combinations: int) -> str:
    if isinstance(space, FiniteSpace):
        if space.cardinality() > max_discrete:
            raise CapabilityError(
                f"FiniteSpace cardinality {space.cardinality()} exceeds "
                f"max_discrete={max_discrete}"
            )
        return "finite"
    if not isinstance(space, Space):
        raise CapabilityError("unsupported space type")
    continuous = 0
    discrete_sizes: list[int] = []
    for variable in space.variables.values():
        if isinstance(variable, Constant):
            continue
        if isinstance(variable, Real) and variable.step is None:
            continuous += 1
            continue
        size = variable.size()
        if size is None:
            raise CapabilityError("BayesianOptimization cannot encode this variable")
        discrete_sizes.append(size)
    if continuous == 0 and not discrete_sizes:
        raise CapabilityError("space has no free coordinates")
    if continuous == 0:
        cardinality = space.cardinality()
        if cardinality is None or cardinality > max_discrete:
            raise CapabilityError(
                "fully discrete space is larger than max_discrete="
                f"{max_discrete}"
            )
        return "finite_product"
    if not discrete_sizes:
        return "continuous"
    combinations = 1
    for size in discrete_sizes:
        combinations *= size
    if combinations <= max_choice_combinations:
        return "mixed"
    raise CapabilityError(
        f"discrete combinations {combinations} exceed "
        f"max_choice_combinations={max_choice_combinations}"
    )


class BoxEncoder:
    """Unit-cube encoding of a product space. Discrete coordinates stay on their levels."""

    def __init__(self, space: Space) -> None:
        self.space = space
        self.continuous: list[tuple[str, Real]] = []
        self.discrete: list[tuple[str, list[Any]]] = []
        for name, variable in space.variables.items():
            if isinstance(variable, Constant):
                continue
            if isinstance(variable, Real) and variable.step is None:
                self.continuous.append((name, variable))
            else:
                self.discrete.append((name, list(variable.enumerate())))
        self.dim = len(self.continuous) + len(self.discrete)
        self.discrete_columns = list(
            range(len(self.continuous), len(self.continuous) + len(self.discrete))
        )

    def encode(self, params: Mapping[str, Any]) -> list[float]:
        row = [float(_warp(variable, params[name])) for name, variable in self.continuous]
        for name, levels in self.discrete:
            row.append(_encode_level(levels, params[name]))
        return row

    def decode_continuous(self, unit: Sequence[float], cell: Mapping[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for index, (name, variable) in enumerate(self.continuous):
            params[name] = _unwarp(variable, float(unit[index]))
        params.update(cell)
        return self.space.validate(params)

    def cells(self):
        if not self.discrete:
            yield {}
            return
        names = [name for name, _ in self.discrete]
        levels = [values for _, values in self.discrete]
        for combo in itertools.product(*levels):
            yield dict(zip(names, combo, strict=True))

    def fixed_feature(self, cell: Mapping[str, Any]) -> list[float]:
        return [_encode_level(levels, cell[name]) for name, levels in self.discrete]


class LibraryEncoder:
    """Feature rows for an enumerated library. Proposals are rows of that library."""

    def __init__(self, space: Space | FiniteSpace) -> None:
        self.candidates = [dict(candidate) for candidate in space.enumerate()]
        self.features = _library_matrix(self.candidates)
        self.dim = 0 if not self.features else len(self.features[0])

    def encode(self, params: Mapping[str, Any]) -> list[float]:
        identity = tuple((name, freeze(pythonize(params[name]))) for name in self.candidates[0])
        for candidate, row in zip(self.candidates, self.features, strict=True):
            other = tuple((name, freeze(candidate[name])) for name in candidate)
            if other == identity:
                return list(row)
        raise MolOptError("observation is outside the encoded library")


def _make_encoder(space: Space | FiniteSpace, mode: str) -> BoxEncoder | LibraryEncoder:
    if mode in {"finite", "finite_product"}:
        return LibraryEncoder(space)
    assert isinstance(space, Space)
    return BoxEncoder(space)


def _library_matrix(candidates: list[dict[str, Any]]) -> list[list[float]]:
    if not candidates:
        return []
    names = list(candidates[0])
    columns: list[tuple[str, str, Any]] = []
    for name in names:
        values = [candidate[name] for candidate in candidates]
        if all(_is_number(value) for value in values):
            low = min(float(value) for value in values)
            high = max(float(value) for value in values)
            if low != high:
                columns.append((name, "number", (low, high)))
            continue
        levels: list[Any] = []
        seen: set[tuple] = set()
        for value in values:
            token = freeze(value)
            if token not in seen:
                seen.add(token)
                levels.append(value)
        if len(levels) > 1:
            columns.append((name, "level", levels))
    matrix = []
    for candidate in candidates:
        row = []
        for name, kind, payload in columns:
            if kind == "number":
                low, high = payload
                row.append((float(candidate[name]) - low) / (high - low))
            else:
                row.append(_encode_level(payload, candidate[name]))
        matrix.append(row)
    return matrix


def _encode_level(levels: Sequence[Any], value: Any) -> float:
    token = freeze(pythonize(value))
    for index, level in enumerate(levels):
        if freeze(pythonize(level)) == token:
            if len(levels) == 1:
                return 0.0
            return index / (len(levels) - 1)
    raise MolOptError("value is not one of the encoded levels")


def _warp(variable: Real, value: Any) -> float:
    number = float(value)
    if variable.log:
        return (math.log(number) - math.log(variable.low)) / (
            math.log(variable.high) - math.log(variable.low)
        )
    return (number - variable.low) / (variable.high - variable.low)


def _unwarp(variable: Real, unit: float) -> float:
    unit = min(1.0, max(0.0, unit))
    if variable.log:
        return math.exp(math.log(variable.low) + unit * (math.log(variable.high) - math.log(variable.low)))
    return variable.low + unit * (variable.high - variable.low)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _match_row(row, choices) -> int | None:
    distance = (choices - row).abs().sum(dim=-1)
    index = int(distance.argmin())
    if float(distance[index]) > 1e-6:
        return None
    return index


def _optional_noise(noise: float | None) -> float | None:
    if noise is None:
        return None
    if isinstance(noise, bool) or not isinstance(noise, (int, float, np.integer, np.floating)):
        raise ValidationError("noise must be a non-negative number or None")
    number = float(noise)
    if not math.isfinite(number) or number < 0:
        raise ValidationError("noise must be a non-negative number or None")
    return number


def _at_least(value: int, minimum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return int(value)
