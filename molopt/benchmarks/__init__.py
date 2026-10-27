"""Mathematical benchmarks used by the tests and examples."""

from molopt.benchmarks.multi_objective import (
    evaluate_zdt1,
    igd,
    zdt1_problem,
    zdt1_reference_front,
)
from molopt.benchmarks.single_objective import (
    BRANIN_MINIMUM,
    branin,
    branin_problem,
    evaluate_branin,
)

__all__ = [
    "BRANIN_MINIMUM",
    "branin",
    "branin_problem",
    "evaluate_branin",
    "evaluate_zdt1",
    "igd",
    "zdt1_problem",
    "zdt1_reference_front",
]
