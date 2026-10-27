"""Pareto dominance and Deb's constrained ranking.

Directions are normalized to minimization only inside these helpers. Callers
that show results keep the original measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from molopt.errors import MolOptError
from molopt.problem import Direction, Problem


def as_minimization(problem: Problem, values: Mapping[str, float]) -> np.ndarray:
    """Return one minimization vector in objective order."""
    row = np.empty(problem.n_objectives, dtype=np.float64)
    for index, objective in enumerate(problem.objectives):
        number = float(values[objective.name])
        if objective.direction is Direction.MAXIMIZE:
            number = -number
        row[index] = number
    return row


def violation(problem: Problem, values: Mapping[str, float]) -> float:
    """Sum of raw bound violations. Zero means the point meets every bound."""
    total = 0.0
    for constraint in problem.constraints:
        number = float(values[constraint.name])
        if constraint.lower is not None:
            total += max(0.0, constraint.lower - number)
        if constraint.upper is not None:
            total += max(0.0, number - constraint.upper)
    return total


def dominates_min(left: np.ndarray, right: np.ndarray) -> bool:
    """Strict minimization dominance."""
    return bool(np.all(left <= right) and np.any(left < right))


def constrained_dominates(
    problem: Problem,
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> bool:
    """Deb constrained dominance.

    A feasible point dominates an infeasible one. Between infeasible points,
    the smaller total violation wins. Feasible points use ordinary Pareto
    dominance. This is not a penalty added to the objectives.
    """
    left_ok = problem.feasible(left)
    right_ok = problem.feasible(right)
    if left_ok and not right_ok:
        return True
    if right_ok and not left_ok:
        return False
    if not left_ok and not right_ok:
        return violation(problem, left) < violation(problem, right)
    return dominates_min(as_minimization(problem, left), as_minimization(problem, right))


def non_dominated_mask(problem: Problem, values_list: Sequence[Mapping[str, float]]) -> np.ndarray:
    """Mask of minimization non-dominated rows. Feasibility is the caller's job."""
    count = len(values_list)
    if count == 0:
        return np.zeros(0, dtype=bool)
    points = np.vstack([as_minimization(problem, values) for values in values_list])
    keep = np.ones(count, dtype=bool)
    for index in range(count):
        if not keep[index]:
            continue
        for other in range(count):
            if index == other or not keep[other]:
                continue
            if dominates_min(points[other], points[index]):
                keep[index] = False
                break
    return keep


def non_dominated_sort(
    problem: Problem,
    values_list: Sequence[Mapping[str, float]],
) -> list[list[int]]:
    """Fronts under constrained dominance. Every index appears once."""
    count = len(values_list)
    if count == 0:
        return []
    children: list[list[int]] = [[] for _ in range(count)]
    dominated = [0] * count
    for left in range(count):
        for right in range(left + 1, count):
            if constrained_dominates(problem, values_list[left], values_list[right]):
                children[left].append(right)
                dominated[right] += 1
            elif constrained_dominates(problem, values_list[right], values_list[left]):
                children[right].append(left)
                dominated[left] += 1
    fronts: list[list[int]] = []
    current = [index for index in range(count) if dominated[index] == 0]
    assigned = 0
    while current:
        fronts.append(current)
        assigned += len(current)
        nxt: list[int] = []
        for index in current:
            for child in children[index]:
                dominated[child] -= 1
                if dominated[child] == 0:
                    nxt.append(child)
        current = nxt
    if assigned != count:
        raise MolOptError("constrained non-dominated sort did not rank every point")
    return fronts


def crowding_distance(
    problem: Problem,
    values_list: Sequence[Mapping[str, float]],
    indices: Sequence[int],
) -> np.ndarray:
    """Crowding distance aligned with ``values_list``. Unlisted rows are zero.

    Distances use the original objective units. A sign flip per objective does
    not change them, so maximization and minimization share this calculation.
    """
    distances = np.zeros(len(values_list), dtype=np.float64)
    front = list(indices)
    if len(front) <= 2:
        for index in front:
            distances[index] = math_inf()
        return distances
    points = np.vstack([as_minimization(problem, values_list[index]) for index in front])
    for objective in range(problem.n_objectives):
        order = sorted(range(len(front)), key=lambda row: points[row, objective])
        distances[front[order[0]]] = math_inf()
        distances[front[order[-1]]] = math_inf()
        span = points[order[-1], objective] - points[order[0], objective]
        if span == 0:
            continue
        for position in range(1, len(order) - 1):
            gap = points[order[position + 1], objective] - points[order[position - 1], objective]
            distances[front[order[position]]] += gap / span
    return distances


def math_inf() -> float:
    return float("inf")
