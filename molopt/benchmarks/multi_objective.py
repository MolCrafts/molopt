"""Multi-objective test functions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from molopt.metrics.pareto import as_minimization
from molopt.problem import Minimize, Problem
from molopt.space import Real, Space


def zdt1(values: Sequence[float]) -> tuple[float, float]:
    xs = [float(value) for value in values]
    f1 = xs[0]
    tail = xs[1:]
    g = 1.0 + 9.0 * sum(tail) / len(tail)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return f1, f2


def zdt1_problem(n_var: int = 6) -> Problem:
    if n_var < 2:
        raise ValueError("ZDT1 needs at least two variables")
    variables = {f"x{index}": Real(0.0, 1.0) for index in range(n_var)}
    return Problem(space=Space(**variables), objectives=[Minimize("f1"), Minimize("f2")])


def evaluate_zdt1(params: Mapping[str, float]) -> dict[str, float]:
    xs = [params[f"x{index}"] for index in range(len(params))]
    f1, f2 = zdt1(xs)
    return {"f1": f1, "f2": f2}


def zdt1_reference_front(n: int = 101) -> list[dict[str, float]]:
    front = []
    for f1 in np.linspace(0.0, 1.0, n):
        front.append({"f1": float(f1), "f2": float(1.0 - math.sqrt(f1))})
    return front


def igd(problem: Problem, obtained: Sequence[Mapping[str, float]], front: Sequence[Mapping[str, float]]) -> float:
    """Mean distance from the analytical front to the nearest obtained point."""
    if not obtained:
        return math.inf
    have = np.vstack([as_minimization(problem, values) for values in obtained])
    reference = np.vstack([as_minimization(problem, values) for values in front])
    distances = []
    for point in reference:
        distances.append(float(np.sqrt(np.sum((have - point) ** 2, axis=1)).min()))
    return float(np.mean(distances))
