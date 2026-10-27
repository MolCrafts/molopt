"""Optimization problem: domain, objective directions, and outcome constraints.

Input constraints belong to the search space. Outcome constraints are
measurements. This object does not evaluate anything and does not rescale
stored results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from molopt.errors import ValidationError
from molopt.space import FiniteSpace, Space


class Direction(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Direction


@dataclass(frozen=True)
class Maximize:
    """Declaration that a measured outcome should be maximized."""

    name: str


@dataclass(frozen=True)
class Minimize:
    """Declaration that a measured outcome should be minimized."""

    name: str


@dataclass(frozen=True)
class Constraint:
    """Bound on a measured outcome.

    A completed trial is feasible when every constraint satisfies
    ``lower <= value <= upper``. Missing bounds are open.
    """

    name: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ValidationError("constraint name must be a non-empty string")
        lower = _optional_bound(self.lower, "lower")
        upper = _optional_bound(self.upper, "upper")
        if lower is None and upper is None:
            raise ValidationError(f"constraint {self.name!r} needs a lower or upper bound")
        if lower is not None and upper is not None and lower > upper:
            raise ValidationError(f"constraint {self.name!r} has lower > upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


ObjectiveDecl = Maximize | Minimize | Objective


class Problem:
    """Mathematical task passed to every optimizer."""

    def __init__(
        self,
        space: Space | FiniteSpace,
        objectives: Sequence[ObjectiveDecl],
        constraints: Sequence[Constraint] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(space, (Space, FiniteSpace)):
            raise ValidationError("space must be a Space or FiniteSpace")
        if not isinstance(objectives, Sequence) or isinstance(objectives, (str, bytes)):
            raise ValidationError("objectives must be a sequence")
        parsed = tuple(_as_objective(item) for item in objectives)
        if not parsed:
            raise ValidationError("a problem needs at least one objective")
        names = [item.name for item in parsed]
        if len(names) != len(set(names)):
            raise ValidationError("objective names must be unique")
        constraint_list = () if constraints is None else tuple(constraints)
        if not isinstance(constraints, Sequence) and constraints is not None:
            raise ValidationError("constraints must be a sequence or None")
        for item in constraint_list:
            if not isinstance(item, Constraint):
                raise ValidationError("constraints must be Constraint instances")
        constraint_names = [item.name for item in constraint_list]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValidationError("constraint names must be unique")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValidationError("metadata must be a mapping or None")
        self.space = space
        self.objectives = parsed
        self.constraints = constraint_list
        self.metadata = None if metadata is None else dict(metadata)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.objectives)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.constraints)

    @property
    def measured_names(self) -> tuple[str, ...]:
        names = list(self.objective_names)
        for name in self.constraint_names:
            if name not in names:
                names.append(name)
        return tuple(names)

    def feasible(self, values: Mapping[str, Any]) -> bool:
        for constraint in self.constraints:
            if constraint.name not in values:
                raise ValidationError(f"missing constraint value {constraint.name!r}")
            number = _measurement(values[constraint.name], constraint.name)
            if constraint.lower is not None and number < constraint.lower:
                return False
            if constraint.upper is not None and number > constraint.upper:
                return False
        return True

    def to_spec(self) -> dict[str, Any]:
        """Checkpoint signature. Metadata is intentionally omitted."""
        return {
            "space": self.space.to_spec(),
            "objectives": [
                {"name": item.name, "direction": item.direction.value}
                for item in self.objectives
            ],
            "constraints": [
                {"name": item.name, "lower": item.lower, "upper": item.upper}
                for item in self.constraints
            ],
        }


def _as_objective(item: ObjectiveDecl) -> Objective:
    if isinstance(item, Objective):
        if not isinstance(item.name, str) or item.name == "":
            raise ValidationError("objective name must be a non-empty string")
        return item
    if isinstance(item, Maximize):
        direction = Direction.MAXIMIZE
    elif isinstance(item, Minimize):
        direction = Direction.MINIMIZE
    else:
        raise ValidationError("objectives must be Maximize, Minimize, or Objective")
    if not isinstance(item.name, str) or item.name == "":
        raise ValidationError("objective name must be a non-empty string")
    return Objective(item.name, direction)


def _optional_bound(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"constraint {label} must be a real number")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"constraint {label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"constraint {label} must be finite")
    return number


def _measurement(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a real measurement")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValidationError(f"{name} must be a real measurement")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number
