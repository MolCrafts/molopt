"""Single-objective test functions."""

from __future__ import annotations

import math

from molopt.problem import Minimize, Problem
from molopt.space import Real, Space

# Global minimum of the standard Branin-Hoo function on [-5, 10] x [0, 15].
BRANIN_MINIMUM = 0.39788735772973816


def branin(x1: float, x2: float) -> float:
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return (
        a * (x2 - b * x1**2 + c * x1 - r) ** 2
        + s * (1.0 - t) * math.cos(x1)
        + s
    )


def branin_problem() -> Problem:
    return Problem(
        space=Space(x1=Real(-5.0, 10.0), x2=Real(0.0, 15.0)),
        objectives=[Minimize("y")],
    )


def evaluate_branin(params: dict[str, float]) -> dict[str, float]:
    return {"y": branin(params["x1"], params["x2"])}
