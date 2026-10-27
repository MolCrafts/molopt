"""Search spaces.

A space samples, validates, and identifies candidates. It does not know which
optimizer will use those candidates. ``FiniteSpace`` is a space, not a variable:
its candidates are valid designs, and algorithms must not invent points outside
that set.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from molopt.errors import ValidationError

Json = Any


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def pythonize(value: Any) -> Json:
    """Return a JSON-safe copy of a parameter value."""
    if _is_bool(value):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError("parameter values must be finite")
        return number
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    raise ValidationError(
        "parameter values must be JSON scalars (str, int, float, bool, null) "
        "or lists of those"
    )


def freeze(value: Json) -> tuple:
    """Hashable identity token that keeps bool distinct from int."""
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, float):
        return ("f", value)
    if isinstance(value, str):
        return ("s", value)
    if value is None:
        return ("n",)
    if isinstance(value, list):
        return ("l", tuple(freeze(item) for item in value))
    raise ValidationError(f"cannot freeze value of type {type(value).__name__}")


def _finite_number(value: object, label: str) -> float:
    if _is_bool(value) or isinstance(value, np.bool_):
        raise ValidationError(f"{label} must be a real number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be finite")
    return number


def _step_count(low: float, high: float, step: float) -> int:
    span = (high - low) / step
    count = int(math.floor(span + 1e-9))
    last = low + count * step
    tolerance = 1e-8 * max(1.0, abs(high), abs(step))
    if last > high + tolerance:
        count -= 1
    if count < 0:
        raise ValidationError("step does not fit inside the bounds")
    return count + 1


@dataclass(frozen=True)
class Real:
    """Bounded continuous variable.

    ``log=True`` changes the sampling measure, not the valid set. ``step``
    makes the variable a finite grid. Logarithmic sampling and a step cannot
    be combined.
    """

    low: float
    high: float
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        low = _finite_number(self.low, "Real.low")
        high = _finite_number(self.high, "Real.high")
        if low >= high:
            raise ValidationError("Real bounds must satisfy low < high")
        step = self.step
        if step is not None:
            step = _finite_number(step, "Real.step")
            if step <= 0:
                raise ValidationError("Real.step must be positive")
        if self.log and step is not None:
            raise ValidationError("Real cannot combine log=True with step")
        if self.log and low <= 0:
            raise ValidationError("Real(log=True) requires low > 0")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "step", step)

    def size(self) -> int | None:
        if self.step is None:
            return None
        return _step_count(self.low, self.high, self.step)

    def sample(self, rng: np.random.Generator) -> float:
        if self.step is not None:
            index = int(rng.integers(0, self.size()))
            return self.low + index * self.step
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def validate(self, value: object) -> float:
        number = _finite_number(value, "real value")
        if self.step is None:
            if number < self.low or number > self.high:
                raise ValidationError(
                    f"value {number} is outside [{self.low}, {self.high}]"
                )
            return number
        count = self.size()
        assert count is not None
        index = int(round((number - self.low) / self.step))
        if index < 0 or index >= count:
            raise ValidationError(
                f"value {number} is outside the stepped range "
                f"[{self.low}, {self.high}] step {self.step}"
            )
        canonical = self.low + index * self.step
        tolerance = 1e-8 * max(1.0, abs(self.step), abs(number))
        if abs(canonical - number) > tolerance:
            raise ValidationError(
                f"value {number} is not on the step grid of {self.step}"
            )
        return float(canonical)

    def enumerate(self) -> Iterator[float]:
        count = self.size()
        if count is None:
            raise ValidationError("continuous Real cannot be enumerated")
        assert self.step is not None
        for index in range(count):
            yield self.low + index * self.step

    def to_spec(self) -> dict[str, Json]:
        return {
            "type": "real",
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }


@dataclass(frozen=True)
class Integer:
    """Inclusive integer range.

    ``log=True`` samples uniformly in log space and rounds onto the valid
    set. Enumeration and validation still use every stepped integer.
    """

    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if _is_bool(self.low) or _is_bool(self.high) or _is_bool(self.step):
            raise ValidationError("Integer bounds and step must be integers")
        if not isinstance(self.low, (int, np.integer)) or not isinstance(
            self.high, (int, np.integer)
        ):
            raise ValidationError("Integer bounds must be integers")
        low = int(self.low)
        high = int(self.high)
        if not isinstance(self.step, (int, np.integer)):
            raise ValidationError("Integer.step must be an integer")
        step = int(self.step)
        if low > high:
            raise ValidationError("Integer bounds must satisfy low <= high")
        if step <= 0:
            raise ValidationError("Integer.step must be positive")
        if self.log and low <= 0:
            raise ValidationError("Integer(log=True) requires low > 0")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "step", step)

    def size(self) -> int:
        return (self.high - self.low) // self.step + 1

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            drawn = int(round(math.exp(rng.uniform(math.log(self.low), math.log(self.high)))))
            drawn = min(self.high, max(self.low, drawn))
            index = int(round((drawn - self.low) / self.step))
            index = min(self.size() - 1, max(0, index))
            return self.low + index * self.step
        index = int(rng.integers(0, self.size()))
        return self.low + index * self.step

    def validate(self, value: object) -> int:
        if _is_bool(value) or not isinstance(value, (int, np.integer)):
            raise ValidationError("integer value must be an int")
        number = int(value)
        if number < self.low or number > self.high:
            raise ValidationError(
                f"value {number} is outside [{self.low}, {self.high}]"
            )
        if (number - self.low) % self.step != 0:
            raise ValidationError(
                f"value {number} is not on the integer step grid of {self.step}"
            )
        return number

    def enumerate(self) -> Iterator[int]:
        for index in range(self.size()):
            yield self.low + index * self.step

    def to_spec(self) -> dict[str, Json]:
        return {
            "type": "integer",
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }


@dataclass(frozen=True)
class Choice:
    """Categorical or discrete choice, in the order given."""

    values: tuple[Json, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValidationError("Choice values must be a sequence")
        if len(values) == 0:
            raise ValidationError("Choice values must be non-empty")
        plain = tuple(pythonize(value) for value in values)
        seen: set[tuple] = set()
        for value in plain:
            token = freeze(value)
            if token in seen:
                raise ValidationError(f"duplicate choice value {value!r}")
            seen.add(token)
        object.__setattr__(self, "values", plain)

    def size(self) -> int:
        return len(self.values)

    def sample(self, rng: np.random.Generator) -> Json:
        index = int(rng.integers(0, len(self.values)))
        return self.values[index]

    def validate(self, value: object) -> Json:
        try:
            plain = pythonize(value)
            token = freeze(plain)
        except ValidationError as exc:
            raise ValidationError(f"invalid choice {value!r}") from exc
        for candidate in self.values:
            if freeze(candidate) == token:
                return candidate
        raise ValidationError(f"value {value!r} is not in the choice list")

    def enumerate(self) -> Iterator[Json]:
        yield from self.values

    def to_spec(self) -> dict[str, Json]:
        return {"type": "choice", "values": list(self.values)}


@dataclass(frozen=True)
class Constant:
    """Fixed parameter included in every candidate."""

    value: Json

    def __init__(self, value: Any) -> None:
        object.__setattr__(self, "value", pythonize(value))

    def size(self) -> int:
        return 1

    def sample(self, rng: np.random.Generator) -> Json:
        del rng
        return self.value

    def validate(self, value: object) -> Json:
        plain = pythonize(value)
        if freeze(plain) != freeze(self.value):
            raise ValidationError(
                f"constant value {value!r} does not match {self.value!r}"
            )
        return self.value

    def enumerate(self) -> Iterator[Json]:
        yield self.value

    def to_spec(self) -> dict[str, Json]:
        return {"type": "constant", "value": self.value}


Variable = Real | Integer | Choice | Constant


class Space:
    """Cartesian product of named variables.

    Keyword order is the canonical variable order. Sampling draws each
    variable independently. Identity is exact on validated values, with bool
    kept distinct from int.
    """

    def __init__(self, **variables: Variable) -> None:
        if not variables:
            raise ValidationError("Space requires at least one variable")
        ordered: dict[str, Variable] = {}
        for name, variable in variables.items():
            if not isinstance(name, str) or name == "":
                raise ValidationError("variable names must be non-empty strings")
            if not isinstance(variable, (Real, Integer, Choice, Constant)):
                raise ValidationError(
                    f"{name} must be Real, Integer, Choice, or Constant"
                )
            ordered[name] = variable
        self._variables = ordered

    @property
    def variables(self) -> Mapping[str, Variable]:
        return self._variables

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._variables)

    def cardinality(self) -> int | None:
        total = 1
        for variable in self._variables.values():
            size = variable.size()
            if size is None:
                return None
            total *= size
        return total

    def sample(self, n: int, rng: np.random.Generator) -> list[dict[str, Json]]:
        _check_count(n)
        return [self._sample_one(rng) for _ in range(n)]

    def _sample_one(self, rng: np.random.Generator) -> dict[str, Json]:
        return {name: variable.sample(rng) for name, variable in self._variables.items()}

    def validate(self, params: Mapping[str, Any]) -> dict[str, Json]:
        if not isinstance(params, Mapping):
            raise ValidationError("params must be a mapping")
        unknown = set(params).difference(self._variables)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValidationError(f"unknown parameters: {names}")
        validated: dict[str, Json] = {}
        for name, variable in self._variables.items():
            if name not in params:
                if isinstance(variable, Constant):
                    validated[name] = variable.value
                    continue
                raise ValidationError(f"missing parameter {name!r}")
            try:
                validated[name] = variable.validate(params[name])
            except ValidationError as exc:
                raise ValidationError(f"{name}: {exc}") from exc
        return validated

    def contains(self, params: Mapping[str, Any]) -> bool:
        try:
            self.validate(params)
        except ValidationError:
            return False
        return True

    def identity(self, params: Mapping[str, Any]) -> tuple:
        validated = self.validate(params)
        return tuple((name, freeze(validated[name])) for name in self.names)

    def enumerate(self) -> Iterator[dict[str, Json]]:
        if self.cardinality() is None:
            raise ValidationError("cannot enumerate an infinite space")
        names = list(self._variables)
        levels = [list(self._variables[name].enumerate()) for name in names]
        for combo in itertools.product(*levels):
            yield dict(zip(names, combo, strict=True))

    def to_spec(self) -> dict[str, Json]:
        return {
            "type": "product",
            "variables": [
                {"name": name, **variable.to_spec()}
                for name, variable in self._variables.items()
            ],
        }


class FiniteSpace:
    """Explicit finite set of valid designs.

    Every candidate must use the same keys. Duplicate identities are rejected.
    Componentwise crossover of a product space is a different object: it can
    leave this set, so algorithms that support ``FiniteSpace`` have to search
    inside the list.
    """

    def __init__(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValidationError("FiniteSpace candidates must be a sequence of mappings")
        if len(candidates) == 0:
            raise ValidationError("FiniteSpace requires at least one candidate")
        names: tuple[str, ...] | None = None
        stored: list[dict[str, Json]] = []
        seen: set[tuple] = set()
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValidationError(f"candidate {index} must be a mapping")
            plain = {str(key): pythonize(value) for key, value in candidate.items()}
            if any(not isinstance(key, str) or key == "" for key in candidate):
                raise ValidationError(f"candidate {index} has an invalid parameter name")
            order = tuple(plain)
            if names is None:
                if not order:
                    raise ValidationError("candidates must contain at least one parameter")
                names = order
            elif order != names:
                raise ValidationError(
                    "every FiniteSpace candidate must use the same keys in the same order"
                )
            token = tuple((name, freeze(plain[name])) for name in names)
            if token in seen:
                raise ValidationError(f"duplicate candidate at index {index}")
            seen.add(token)
            stored.append(plain)
        assert names is not None
        self._names = names
        self._candidates = tuple(stored)
        self._tokens = seen

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def candidates(self) -> tuple[dict[str, Json], ...]:
        return self._candidates

    @property
    def variables(self) -> Mapping[str, Variable]:
        raise AttributeError(
            "FiniteSpace has no independent variables; use .candidates"
        )

    def cardinality(self) -> int:
        return len(self._candidates)

    def sample(self, n: int, rng: np.random.Generator) -> list[dict[str, Json]]:
        _check_count(n)
        indexes = rng.integers(0, len(self._candidates), size=n)
        return [dict(self._candidates[int(index)]) for index in indexes]

    def validate(self, params: Mapping[str, Any]) -> dict[str, Json]:
        if not isinstance(params, Mapping):
            raise ValidationError("params must be a mapping")
        if set(params) != set(self._names):
            raise ValidationError(
                f"parameters must be exactly {list(self._names)}"
            )
        plain = {name: pythonize(params[name]) for name in self._names}
        token = tuple((name, freeze(plain[name])) for name in self._names)
        if token not in self._tokens:
            raise ValidationError("parameters are not in the finite candidate set")
        return plain

    def contains(self, params: Mapping[str, Any]) -> bool:
        try:
            self.validate(params)
        except ValidationError:
            return False
        return True

    def identity(self, params: Mapping[str, Any]) -> tuple:
        validated = self.validate(params)
        return tuple((name, freeze(validated[name])) for name in self._names)

    def enumerate(self) -> Iterator[dict[str, Json]]:
        for candidate in self._candidates:
            yield dict(candidate)

    def to_spec(self) -> dict[str, Json]:
        return {"type": "finite", "candidates": [dict(candidate) for candidate in self._candidates]}


def _check_count(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
        raise ValidationError("sample count must be a non-negative integer")
