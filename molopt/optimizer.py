"""Ask–tell contract shared by every algorithm.

The base class owns trial ids, pending tracking, result validation, observed
``best`` / ``pareto_front``, and the checkpoint envelope. An algorithm only
proposes candidates and consumes newly accepted results. Reported solutions
are measurements. Subclasses cannot replace ``best`` or ``pareto_front``.
"""

from __future__ import annotations

import json
import math
from abc import ABC
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from molopt._version import __version__
from molopt.errors import MolOptError, StateError, TellError, ValidationError
from molopt.metrics.pareto import as_minimization, non_dominated_mask
from molopt.problem import Problem
from molopt.result import Result, Status
from molopt.solution import Solution
from molopt.trial import Trial

CONTRACT_VERSION = 1
Candidate = tuple[Mapping[str, Any], float | None, Mapping[str, Any] | None]


class DuplicatePolicy(Enum):
    """Whether an identity may be proposed again.

    ``SKIP`` blocks pending, completed, failed, and cancelled identities.
    ``ALLOW`` issues a new trial id for an intentional replicate.
    """

    SKIP = "skip"
    ALLOW = "allow"


class Optimizer(ABC):
    """Stable optimizer interface.

    ``ask`` may return fewer candidates than requested. ``tell`` accepts one
    result or a list, in any order. An identical repeat is ignored. A
    conflicting payload or an unknown id raises ``TellError``.
    """

    algorithm_name = "optimizer"
    algorithm_version = 1

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
    ) -> None:
        if not isinstance(problem, Problem):
            raise ValidationError("problem must be a Problem")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValidationError("seed must be a non-negative integer")
        if not isinstance(duplicates, DuplicatePolicy):
            raise ValidationError("duplicates must be a DuplicatePolicy")
        self.problem = problem
        self.seed = int(seed)
        self.duplicates = duplicates
        self._rng = np.random.default_rng(self.seed)
        self._trial_counter = 0
        self._trials: dict[str, Trial] = {}
        self._pending: dict[str, Trial] = {}
        self._results: dict[str, Result] = {}
        self._blocked: set[tuple] = set()

    def ask(self, n: int = 1) -> list[Trial]:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
            raise ValidationError("n must be an integer >= 1")
        raw = self._ask(int(n))
        if len(raw) > n:
            raise MolOptError("algorithm returned more candidates than requested")
        prepared: list[tuple[dict[str, Any], float | None, dict[str, Any], tuple]] = []
        seen: set[tuple] = set()
        for params, fidelity, metadata in raw:
            try:
                validated = self.problem.space.validate(params)
            except ValidationError as exc:
                raise MolOptError("algorithm proposed a candidate outside the space") from exc
            identity = self.problem.space.identity(validated)
            if identity in seen and self.duplicates is DuplicatePolicy.SKIP:
                raise MolOptError("algorithm proposed a duplicate candidate")
            if self.duplicates is DuplicatePolicy.SKIP and identity in self._blocked:
                raise MolOptError("algorithm proposed a blocked candidate")
            seen.add(identity)
            meta = {} if metadata is None else dict(metadata)
            _assert_plain(meta, "metadata")
            if fidelity is not None:
                fidelity = float(fidelity)
                if not math.isfinite(fidelity):
                    raise MolOptError("algorithm proposed a non-finite fidelity")
            prepared.append((validated, fidelity, meta, identity))
        trials: list[Trial] = []
        for validated, fidelity, meta, identity in prepared:
            trial = Trial(
                id=str(self._trial_counter),
                params=validated,
                fidelity=fidelity,
                metadata=meta,
            )
            self._trial_counter += 1
            self._trials[trial.id] = trial
            self._pending[trial.id] = trial
            if self.duplicates is DuplicatePolicy.SKIP:
                self._blocked.add(identity)
            trials.append(trial)
        self._register(trials)
        return trials

    def tell(self, results: Result | Sequence[Result]) -> None:
        batch = [results] if isinstance(results, Result) else list(results)
        if not batch and not isinstance(results, Result):
            return
        staged: dict[str, Result] = {}
        for result in batch:
            if not isinstance(result, Result):
                raise TellError("tell() expects Result instances")
            existing = self._results.get(result.trial_id)
            if existing is not None:
                if _same_result(existing, result):
                    continue
                raise TellError(f"conflicting result for trial {result.trial_id}")
            if result.trial_id in staged:
                if _same_result(staged[result.trial_id], result):
                    continue
                raise TellError(f"conflicting result for trial {result.trial_id}")
            if result.trial_id not in self._pending:
                raise TellError(f"unknown trial id {result.trial_id}")
            self._validate_result(result)
            self._validate_extra(result)
            staged[result.trial_id] = result
        if not staged:
            return
        accepted = list(staged.values())
        for result in accepted:
            self._pending.pop(result.trial_id)
            self._results[result.trial_id] = result
        self._tell(accepted)

    def best(self) -> Solution | None:
        if self.problem.n_objectives != 1:
            raise MolOptError(
                "best() is for single-objective problems; use pareto_front()"
            )
        front = self.pareto_front()
        return front[0] if front else None

    def pareto_front(self) -> list[Solution]:
        """Observed feasible non-dominated set, in stable objective order."""
        values: list[Mapping[str, Any]] = []
        solutions: list[Solution] = []
        for trial_id, result in self._results.items():
            if result.status is not Status.COMPLETED or result.values is None:
                continue
            if not self.problem.feasible(result.values):
                continue
            trial = self._trials[trial_id]
            values.append(result.values)
            solutions.append(
                Solution(trial_id=trial_id, params=dict(trial.params), values=dict(result.values))
            )
        if not solutions:
            return []
        mask = non_dominated_mask(self.problem, values)
        front = [solution for solution, keep in zip(solutions, mask, strict=True) if keep]
        front.sort(key=lambda solution: self._solution_key(solution))
        return front

    def state_dict(self) -> dict[str, Any]:
        state = {
            "molopt_version": __version__,
            "contract_version": CONTRACT_VERSION,
            "algorithm": self.algorithm_name,
            "algorithm_version": self.algorithm_version,
            "problem": self.problem.to_spec(),
            "seed": self.seed,
            "duplicates": self.duplicates.value,
            "rng": to_plain(self._rng.bit_generator.state),
            "trial_counter": self._trial_counter,
            "trials": [_trial_plain(trial) for trial in self._trials.values()],
            "results": [_result_plain(result) for result in self._results.values()],
            "algorithm_state": to_plain(self._algorithm_state()),
        }
        # Fail here if a subclass put a live array or model into the checkpoint.
        json.dumps(state)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._check_state(state)
        self._apply_state(state)

    def _ask(self, n: int) -> list[Candidate]:
        raise NotImplementedError

    def _tell(self, results: list[Result]) -> None:
        del results

    def _register(self, trials: list[Trial]) -> None:
        del trials

    def _validate_extra(self, result: Result) -> None:
        del result

    def _algorithm_state(self) -> dict[str, Any]:
        return {}

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        del state

    def _validate_result(self, result: Result) -> None:
        if result.cost is not None and (
            not math.isfinite(result.cost) or result.cost < 0
        ):
            raise TellError("cost must be finite and >= 0")
        if result.status is not Status.COMPLETED:
            if result.values is not None or result.stderr is not None:
                raise TellError(
                    "failed and cancelled results must not carry values or stderr"
                )
            return
        if result.values is None:
            raise TellError("a completed result needs values")
        required = self.problem.measured_names
        missing = [name for name in required if name not in result.values]
        if missing:
            raise TellError(
                "completed result is missing measurements: " + ", ".join(missing)
            )
        for name, value in result.values.items():
            _require_finite(value, f"value {name}")
        if result.stderr is not None:
            known = set(required)
            for name, value in result.stderr.items():
                if name not in known:
                    raise TellError(f"stderr contains unknown measurement {name!r}")
                number = _require_finite(value, f"stderr {name}")
                if number < 0:
                    raise TellError(f"stderr {name} must be >= 0")

    def _solution_key(self, solution: Solution) -> tuple:
        minimized = tuple(float(item) for item in as_minimization(self.problem, solution.values))
        return (minimized, int(solution.trial_id))

    def _check_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise StateError("checkpoint must be a dict")
        if state.get("contract_version") != CONTRACT_VERSION:
            raise StateError("checkpoint contract version is incompatible")
        if state.get("algorithm") != self.algorithm_name:
            raise StateError(
                f"checkpoint belongs to {state.get('algorithm')!r}, "
                f"not {self.algorithm_name!r}"
            )
        if state.get("algorithm_version") != self.algorithm_version:
            raise StateError("checkpoint algorithm version is incompatible")
        if state.get("problem") != self.problem.to_spec():
            raise StateError(
                "checkpoint problem does not match this search space, "
                "objectives, or constraints"
            )
        if state.get("duplicates") != self.duplicates.value or state.get("seed") != self.seed:
            raise StateError("checkpoint seed or duplicate policy does not match")

    def _apply_state(self, state: Mapping[str, Any]) -> None:
        self._rng.bit_generator.state = state["rng"]
        self._trial_counter = int(state["trial_counter"])
        self._trials = {}
        self._pending = {}
        self._results = {}
        self._blocked = set()
        for payload in state["trials"]:
            trial = Trial(
                id=payload["id"],
                params=payload["params"],
                fidelity=payload["fidelity"],
                metadata=payload["metadata"],
            )
            self._trials[trial.id] = trial
        told: set[str] = set()
        for payload in state["results"]:
            result = Result(
                trial_id=payload["trial_id"],
                values=payload["values"],
                stderr=payload["stderr"],
                cost=payload["cost"],
                status=payload["status"],
                failure=payload["failure"],
            )
            self._results[result.trial_id] = result
            told.add(result.trial_id)
        for trial_id, trial in self._trials.items():
            if trial_id not in told:
                self._pending[trial_id] = trial
            if self.duplicates is DuplicatePolicy.SKIP:
                self._blocked.add(self.problem.space.identity(trial.params))
        self._load_algorithm_state(state["algorithm_state"])

    def _is_blocked(self, params: Mapping[str, Any]) -> bool:
        if self.duplicates is DuplicatePolicy.ALLOW:
            return False
        return self.problem.space.identity(params) in self._blocked


def to_plain(value: Any) -> Any:
    """Convert checkpoint contents to JSON-safe Python values."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise StateError("checkpoint keys must be strings")
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, np.generic):
        return to_plain(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise StateError("checkpoint cannot store a non-finite float")
        return value
    raise StateError(f"cannot store {type(value).__name__} in a checkpoint")


def _trial_plain(trial: Trial) -> dict[str, Any]:
    return {
        "id": trial.id,
        "params": dict(trial.params),
        "fidelity": trial.fidelity,
        "metadata": dict(trial.metadata),
    }


def _result_plain(result: Result) -> dict[str, Any]:
    return {
        "trial_id": result.trial_id,
        "values": None if result.values is None else dict(result.values),
        "stderr": None if result.stderr is None else dict(result.stderr),
        "cost": result.cost,
        "status": result.status.value,
        "failure": result.failure,
    }


def _same_result(left: Result, right: Result) -> bool:
    return _result_payload(left) == _result_payload(right)


def _result_payload(result: Result) -> dict[str, Any]:
    return {
        "trial_id": result.trial_id,
        "status": result.status.value,
        "values": _number_map(result.values),
        "stderr": _number_map(result.stderr),
        "cost": None if result.cost is None else float(result.cost),
        "failure": result.failure,
    }


def _number_map(values: Mapping[str, Any] | None) -> dict[str, float] | None:
    if values is None:
        return None
    return {name: float(values[name]) for name in sorted(values)}


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TellError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise TellError(f"{label} must be finite; submit status='failed' instead of NaN")
    return number


def _assert_plain(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_plain(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} keys must be strings")
            _assert_plain(item, f"{path}.{key}")
        return
    raise ValidationError(f"{path} is not JSON-safe")
