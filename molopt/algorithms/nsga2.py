"""NSGA-II with Deb constrained dominance.

Variation stays inside the declared space. A mating is crossed with
``crossover_prob``; each continuous variable then uses simulated binary
crossover with probability 1/2, which is Deb's variable-wise rate. Polynomial
mutation is applied afterwards. A finite library is searched by index, so
crossover cannot invent a design outside the list. Failed trials are left
out of ranking and are not given penalty objectives.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from molopt.errors import StateError, ValidationError
from molopt.metrics.pareto import crowding_distance, non_dominated_sort
from molopt.optimizer import Candidate, DuplicatePolicy, Optimizer
from molopt.problem import Problem
from molopt.result import Result, Status
from molopt.space import Choice, Constant, FiniteSpace, Integer, Real, Space

Member = dict[str, Any]


class NSGA2(Optimizer):
    """Elitist non-dominated sorting genetic algorithm."""

    algorithm_name = "nsga2"

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        population_size: int,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,
        eta_crossover: float = 15.0,
        eta_mutation: float = 20.0,
        max_attempts: int = 1000,
    ) -> None:
        if isinstance(population_size, bool) or not isinstance(population_size, (int, np.integer)):
            raise ValidationError("population_size must be an integer >= 2")
        if int(population_size) < 2:
            raise ValidationError("population_size must be an integer >= 2")
        self.population_size = int(population_size)
        self.crossover_prob = _unit_prob(crossover_prob, "crossover_prob")
        self.eta_crossover = _positive(eta_crossover, "eta_crossover")
        self.eta_mutation = _positive(eta_mutation, "eta_mutation")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)):
            raise ValidationError("max_attempts must be a positive integer")
        if int(max_attempts) < 1:
            raise ValidationError("max_attempts must be a positive integer")
        self.max_attempts = int(max_attempts)
        super().__init__(problem, seed=seed, duplicates=duplicates)
        space = problem.space
        n_var = 1 if isinstance(space, FiniteSpace) else _variable_count(space)
        if mutation_prob is None:
            mutation_prob = 1.0 / n_var
        self.mutation_prob = _unit_prob(mutation_prob, "mutation_prob")
        self._index: dict[tuple, int] | None = None
        if isinstance(space, FiniteSpace):
            self._index = {
                space.identity(candidate): index
                for index, candidate in enumerate(space.candidates)
            }
        elif not isinstance(space, Space):
            raise ValidationError("space must be a Space or FiniteSpace")
        self._stage = "initial"
        self._generation: list[Member] | None = None
        self._parents: list[Member] | None = None
        self._batch_index: list[int] = []

    def _ask(self, n: int) -> list[Candidate]:
        self._ensure_generation()
        assert self._generation is not None
        self._batch_index = []
        raw: list[Candidate] = []
        for index, member in enumerate(self._generation):
            if member["trial_id"] is None:
                self._batch_index.append(index)
                raw.append((member["params"], None, {}))
                if len(raw) == n:
                    break
        return raw

    def _register(self, trials) -> None:
        assert self._generation is not None
        if len(trials) != len(self._batch_index):
            raise RuntimeError("NSGA-II lost track of the issued generation")
        for index, trial in zip(self._batch_index, trials, strict=True):
            self._generation[index]["trial_id"] = trial.id
        self._batch_index = []

    def _tell(self, results: list[Result]) -> None:
        del results
        generation = self._generation
        if not generation:
            return
        if any(member["trial_id"] is None for member in generation):
            return
        if any(member["trial_id"] not in self._results for member in generation):
            return
        self._evolve()

    def _ensure_generation(self) -> None:
        if self._generation is None:
            self._stage = "initial"
            self._parents = None
            self._generation = self._sample_members(self.population_size)

    def _evolve(self) -> None:
        assert self._generation is not None
        successful = _successful(self._generation, self._results)
        if self._stage == "initial" or not self._parents:
            pool = successful
        else:
            pool = _successful(self._parents, self._results) + successful
        selected = self._select(pool, self.population_size)
        if not selected:
            self._parents = []
            self._stage = "initial"
            self._generation = self._sample_members(self.population_size)
            return
        self._parents = [_public_member(member) for member in selected]
        self._stage = "offspring"
        self._generation = self._vary(selected, self.population_size)

    def _select(self, pool: list[Member], count: int) -> list[Member]:
        if not pool:
            return []
        values = [self._results[member["trial_id"]].values for member in pool]
        fronts = non_dominated_sort(self.problem, values)
        rank: dict[int, int] = {}
        distance: dict[int, float] = {}
        for level, front in enumerate(fronts):
            crowding = crowding_distance(self.problem, values, front)
            for index in front:
                rank[index] = level
                distance[index] = float(crowding[index])
        chosen: list[int] = []
        for front in fronts:
            room = count - len(chosen)
            if room <= 0:
                break
            if len(front) <= room:
                chosen.extend(front)
                continue
            ordered = sorted(
                front,
                key=lambda index: (-distance[index], str(pool[index]["trial_id"])),
            )
            chosen.extend(ordered[:room])
            break
        selected: list[Member] = []
        for index in chosen:
            member = _public_member(pool[index])
            member["rank"] = rank[index]
            member["crowding"] = distance[index]
            selected.append(member)
        return selected

    def _vary(self, parents: list[Member], count: int) -> list[Member]:
        children: list[Member] = []
        seen: set[tuple] = set(self._blocked)
        attempts = 0
        limit = self.max_attempts * count
        while len(children) < count and attempts < limit and parents:
            attempts += 1
            first = self._tournament(parents)
            second = self._tournament(parents)
            for child in self._offspring(first, second):
                if len(children) >= count:
                    break
                identity = self.problem.space.identity(child["params"])
                if self.duplicates is DuplicatePolicy.SKIP and identity in seen:
                    continue
                seen.add(identity)
                children.append(child)
        return children

    def _sample_members(self, count: int) -> list[Member]:
        members: list[Member] = []
        seen: set[tuple] = set(self._blocked)
        attempts = 0
        limit = self.max_attempts * count
        while len(members) < count and attempts < limit:
            attempts += 1
            member = self._random_member()
            identity = self.problem.space.identity(member["params"])
            if self.duplicates is DuplicatePolicy.SKIP and identity in seen:
                continue
            seen.add(identity)
            members.append(member)
        return members

    def _random_member(self) -> Member:
        space = self.problem.space
        if isinstance(space, FiniteSpace):
            assert self._index is not None
            index = int(self._rng.integers(0, space.cardinality()))
            return {
                "params": dict(space.candidates[index]),
                "index": index,
                "trial_id": None,
            }
        params = space.sample(1, self._rng)[0]
        return {"params": params, "index": None, "trial_id": None}

    def _tournament(self, parents: list[Member]) -> Member:
        left, right = (parents[int(index)] for index in self._rng.integers(0, len(parents), size=2))
        if left["rank"] != right["rank"]:
            return left if left["rank"] < right["rank"] else right
        if left["crowding"] != right["crowding"]:
            return left if left["crowding"] > right["crowding"] else right
        return left if float(self._rng.random()) < 0.5 else right

    def _offspring(self, first: Member, second: Member) -> tuple[Member, Member]:
        space = self.problem.space
        if isinstance(space, FiniteSpace):
            return self._finite_offspring(first, second)
        assert isinstance(space, Space)
        child_a = dict(first["params"])
        child_b = dict(second["params"])
        mate = float(self._rng.random()) < self.crossover_prob
        for name, variable in space.variables.items():
            if isinstance(variable, Constant):
                child_a[name] = variable.value
                child_b[name] = variable.value
                continue
            if mate:
                child_a[name], child_b[name] = self._cross_variable(
                    variable, child_a[name], child_b[name]
                )
            if float(self._rng.random()) < self.mutation_prob:
                child_a[name] = self._mutate_variable(variable, child_a[name])
            if float(self._rng.random()) < self.mutation_prob:
                child_b[name] = self._mutate_variable(variable, child_b[name])
            child_a[name] = self._repair(variable, child_a[name])
            child_b[name] = self._repair(variable, child_b[name])
        return (
            {"params": space.validate(child_a), "index": None, "trial_id": None},
            {"params": space.validate(child_b), "index": None, "trial_id": None},
        )

    def _finite_offspring(self, first: Member, second: Member) -> tuple[Member, Member]:
        space = self.problem.space
        assert isinstance(space, FiniteSpace)
        left = int(first["index"])
        right = int(second["index"])
        if float(self._rng.random()) < self.crossover_prob:
            child_left = right if float(self._rng.random()) < 0.5 else left
            child_right = left if float(self._rng.random()) < 0.5 else right
        else:
            child_left, child_right = left, right
        if float(self._rng.random()) < self.mutation_prob:
            child_left = self._mutate_index(child_left, space.cardinality())
        if float(self._rng.random()) < self.mutation_prob:
            child_right = self._mutate_index(child_right, space.cardinality())
        return (self._from_index(child_left), self._from_index(child_right))

    def _from_index(self, index: int) -> Member:
        space = self.problem.space
        assert isinstance(space, FiniteSpace)
        return {
            "params": dict(space.candidates[index]),
            "index": index,
            "trial_id": None,
        }

    def _mutate_index(self, index: int, size: int) -> int:
        if size == 1:
            return 0
        jump = int(self._rng.integers(0, size - 1))
        if jump >= index:
            jump += 1
        return jump

    def _cross_variable(self, variable: Real | Integer | Choice, left: Any, right: Any) -> tuple[Any, Any]:
        if isinstance(variable, (Real, Integer)):
            # Inside a selected mating, Deb applies SBX to each variable with probability 1/2.
            if float(self._rng.random()) >= 0.5:
                return left, right
            low = float(variable.low)
            high = float(variable.high)
            first, second = _sbx(float(left), float(right), low, high, self.eta_crossover, self._rng)
            if float(self._rng.random()) < 0.5:
                return second, first
            return first, second
        if float(self._rng.random()) < 0.5:
            return right, left
        return left, right

    def _mutate_variable(self, variable: Real | Integer | Choice, value: Any) -> Any:
        if isinstance(variable, Real):
            return _polynomial_mutation(
                float(value),
                variable.low,
                variable.high,
                self.eta_mutation,
                self._rng,
            )
        if isinstance(variable, Integer):
            return _polynomial_mutation(
                float(value),
                float(variable.low),
                float(variable.high),
                self.eta_mutation,
                self._rng,
            )
        index = int(self._rng.integers(0, len(variable.values)))
        return variable.values[index]

    def _repair(self, variable: Real | Integer | Choice, value: Any) -> Any:
        if isinstance(variable, Integer):
            return variable.validate(_snap_int(float(value), variable))
        if isinstance(variable, Real) and variable.step is not None:
            count = variable.size()
            assert count is not None
            index = int(round((float(value) - variable.low) / variable.step))
            index = max(0, min(index, count - 1))
            return variable.low + index * variable.step
        if isinstance(variable, Real):
            return _clip(float(value), variable.low, variable.high)
        return value

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "stage": self._stage,
            "generation": None if self._generation is None else [_public_member(m) for m in self._generation],
            "parents": None if self._parents is None else [_public_member(m) for m in self._parents],
            "population_size": self.population_size,
            "crossover_prob": self.crossover_prob,
            "mutation_prob": self.mutation_prob,
            "eta_crossover": self.eta_crossover,
            "eta_mutation": self.eta_mutation,
            "max_attempts": self.max_attempts,
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        for name in (
            "population_size",
            "crossover_prob",
            "mutation_prob",
            "eta_crossover",
            "eta_mutation",
            "max_attempts",
        ):
            if state.get(name) != getattr(self, name):
                raise StateError(f"checkpoint {name} does not match")
        self._stage = state["stage"]
        self._generation = state["generation"]
        self._parents = state["parents"]
        self._batch_index = []
        if isinstance(self.problem.space, FiniteSpace) and self._generation:
            for member in self._generation:
                member["index"] = self._index[self.problem.space.identity(member["params"])]
            if self._parents:
                for member in self._parents:
                    member["index"] = self._index[self.problem.space.identity(member["params"])]


def _public_member(member: Member) -> Member:
    return {
        "params": dict(member["params"]),
        "index": member.get("index"),
        "trial_id": member.get("trial_id"),
    }


def _successful(members: list[Member], results: dict[str, Result]) -> list[Member]:
    kept = []
    for member in members:
        result = results[member["trial_id"]]
        if result.status is Status.COMPLETED:
            kept.append(member)
    return kept


def _variable_count(space: Space) -> int:
    return sum(1 for variable in space.variables.values() if not isinstance(variable, Constant)) or 1


def _sbx(
    left: float,
    right: float,
    low: float,
    high: float,
    eta: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if abs(left - right) < 1e-14:
        return _clip(left, low, high), _clip(right, low, high)
    swapped = left > right
    if swapped:
        left, right = right, left
    first = _sbx_child(left, right, low, high, eta, rng, side="lower")
    second = _sbx_child(left, right, low, high, eta, rng, side="upper")
    if swapped:
        first, second = second, first
    return first, second


def _sbx_child(
    left: float,
    right: float,
    low: float,
    high: float,
    eta: float,
    rng: np.random.Generator,
    *,
    side: str,
) -> float:
    span = right - left
    if side == "lower":
        beta = 1.0 + (2.0 * (left - low) / span)
        sign = -1.0
    else:
        beta = 1.0 + (2.0 * (high - right) / span)
        sign = 1.0
    alpha = 2.0 - beta ** (-(eta + 1.0))
    draw = float(rng.random())
    if draw <= 1.0 / alpha:
        betaq = (draw * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - draw * alpha)) ** (1.0 / (eta + 1.0))
    child = 0.5 * ((left + right) + sign * betaq * span)
    return _clip(child, low, high)


def _polynomial_mutation(
    value: float,
    low: float,
    high: float,
    eta: float,
    rng: np.random.Generator,
) -> float:
    if high <= low:
        return low
    delta_low = (value - low) / (high - low)
    delta_high = (high - value) / (high - low)
    draw = float(rng.random())
    mut_pow = 1.0 / (eta + 1.0)
    if draw < 0.5:
        xy = 1.0 - delta_low
        val = 2.0 * draw + (1.0 - 2.0 * draw) * (xy ** (eta + 1.0))
        delta = val**mut_pow - 1.0
    else:
        xy = 1.0 - delta_high
        val = 2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * (xy ** (eta + 1.0))
        delta = 1.0 - val**mut_pow
    return _clip(value + delta * (high - low), low, high)


def _snap_int(value: float, variable: Integer) -> int:
    index = int(round((value - variable.low) / variable.step))
    index = max(0, min(index, variable.size() - 1))
    return variable.low + index * variable.step


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _unit_prob(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValidationError(f"{label} must be a probability")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValidationError(f"{label} must be between 0 and 1")
    return number


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValidationError(f"{label} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValidationError(f"{label} must be a positive number")
    return number
