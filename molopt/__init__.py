"""Ask–tell black-box optimization.

The core package depends only on NumPy and SciPy. Bayesian optimization lives
in :mod:`molopt.algorithms` and imports PyTorch and BoTorch only when that
class is constructed.
"""

from __future__ import annotations

from molopt._version import __version__
from molopt.errors import (
    CapabilityError,
    MolOptError,
    StateError,
    TellError,
    ValidationError,
)
from molopt.optimizer import DuplicatePolicy, Optimizer
from molopt.problem import Constraint, Maximize, Minimize, Problem
from molopt.result import Result, Status
from molopt.solution import Solution
from molopt.space import Choice, Constant, FiniteSpace, Integer, Real, Space
from molopt.trial import Trial

__all__ = [
    "__version__",
    "CapabilityError",
    "Choice",
    "Constant",
    "Constraint",
    "DuplicatePolicy",
    "FiniteSpace",
    "Integer",
    "Maximize",
    "Minimize",
    "MolOptError",
    "Optimizer",
    "Problem",
    "Real",
    "Result",
    "Solution",
    "Space",
    "StateError",
    "Status",
    "TellError",
    "Trial",
    "ValidationError",
]
