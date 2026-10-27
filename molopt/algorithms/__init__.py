"""Concrete optimizers.

Importing this package does not import PyTorch. ``BayesianOptimization`` is
loaded on first access and imports BoTorch only when constructed.
"""

from __future__ import annotations

from molopt.algorithms.grid import GridSearch
from molopt.algorithms.lhs import LatinHypercube
from molopt.algorithms.nsga2 import NSGA2
from molopt.algorithms.random import RandomSearch

__all__ = [
    "BayesianOptimization",
    "GridSearch",
    "LatinHypercube",
    "NSGA2",
    "RandomSearch",
]


def __getattr__(name: str):
    if name == "BayesianOptimization":
        from molopt.algorithms.bayesian import BayesianOptimization

        return BayesianOptimization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + ["BayesianOptimization"])
