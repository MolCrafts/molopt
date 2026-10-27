"""Exact hypervolume for two objectives.

The reference point is required and is expressed in the original objective
units. There is no default. Points must all be strictly better than the
reference after the minimization transform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from molopt.errors import ValidationError
from molopt.metrics.pareto import as_minimization, non_dominated_mask
from molopt.problem import Direction, Problem


def hypervolume(
    problem: Problem,
    values_list: Sequence[Mapping[str, float]],
    reference: Mapping[str, float],
) -> float:
    """Two-objective hypervolume of the given measurements."""
    if problem.n_objectives != 2:
        raise NotImplementedError(
            "hypervolume in v0.1 supports exactly two objectives"
        )
    for objective in problem.objectives:
        if objective.name not in reference:
            raise ValidationError(
                f"reference point is missing objective {objective.name!r}"
            )
    if len(values_list) == 0:
        return 0.0
    reference_min = _reference_minimization(problem, reference)
    points = [as_minimization(problem, values) for values in values_list]
    for point in points:
        if not np.all(point < reference_min):
            raise ValidationError(
                "reference point must be strictly worse than every compared point"
            )
    mask = non_dominated_mask(problem, values_list)
    front = [points[index] for index in range(len(points)) if mask[index]]
    return _hypervolume_2d(front, reference_min)


def _reference_minimization(problem: Problem, reference: Mapping[str, float]) -> np.ndarray:
    row = np.empty(2, dtype=np.float64)
    for index, objective in enumerate(problem.objectives):
        number = float(reference[objective.name])
        if not np.isfinite(number):
            raise ValidationError("reference point must be finite")
        if objective.direction is Direction.MAXIMIZE:
            number = -number
        row[index] = number
    return row


def _hypervolume_2d(points: Sequence[np.ndarray], reference: np.ndarray) -> float:
    ordered = sorted(points, key=lambda point: (point[0], point[1]))
    volume = 0.0
    previous_x = float(reference[0])
    for point in reversed(ordered):
        width = previous_x - float(point[0])
        height = float(reference[1]) - float(point[1])
        if width > 0.0 and height > 0.0:
            volume += width * height
        previous_x = float(point[0])
    return volume
