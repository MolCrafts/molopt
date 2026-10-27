import pytest

from molopt import Constraint, Maximize, Minimize, Problem, Real, Space, ValidationError
from molopt.metrics import (
    constrained_dominates,
    hypervolume,
    non_dominated_sort,
    violation,
)


def _problem():
    return Problem(
        space=Space(x=Real(0, 1)),
        objectives=[Maximize("conductivity"), Minimize("cost")],
        constraints=[Constraint("stability", lower=0.8)],
    )


def test_directions_and_feasibility_are_not_penalties():
    problem = _problem()
    assert problem.feasible({"stability": 0.8})
    assert not problem.feasible({"stability": 0.79})
    assert violation(problem, {"stability": 0.5}) == pytest.approx(0.3)
    good = {"conductivity": 2.0, "cost": 1.0, "stability": 0.9}
    worse = {"conductivity": 1.0, "cost": 2.0, "stability": 0.9}
    infeasible = {"conductivity": 5.0, "cost": 0.1, "stability": 0.2}
    assert constrained_dominates(problem, good, worse)
    assert constrained_dominates(problem, worse, infeasible)
    assert not constrained_dominates(problem, infeasible, worse)


def test_deb_fronts_prefer_feasible_then_violation():
    problem = _problem()
    values = [
        {"conductivity": 1.0, "cost": 1.0, "stability": 0.9},
        {"conductivity": 0.0, "cost": 2.0, "stability": 0.9},
        {"conductivity": 3.0, "cost": 0.0, "stability": 0.7},
        {"conductivity": 3.0, "cost": 0.0, "stability": 0.4},
    ]
    fronts = non_dominated_sort(problem, values)
    assert fronts[0] == [0]
    assert fronts[1] == [1]
    assert fronts[2] == [2]
    assert fronts[3] == [3]


def test_hypervolume_hand_case_respects_direction():
    problem = Problem(
        space=Space(x=Real(0, 1)),
        objectives=[Minimize("f1"), Minimize("f2")],
    )
    values = [{"f1": 0.0, "f2": 1.0}, {"f1": 1.0, "f2": 0.0}]
    assert hypervolume(problem, values, {"f1": 2.0, "f2": 2.0}) == pytest.approx(3.0)

    flipped = Problem(
        space=Space(x=Real(0, 1)),
        objectives=[Maximize("f1"), Maximize("f2")],
    )
    flipped_values = [{"f1": 2.0, "f2": 1.0}, {"f1": 1.0, "f2": 2.0}]
    assert hypervolume(flipped, flipped_values, {"f1": 0.0, "f2": 0.0}) == pytest.approx(3.0)
    with pytest.raises(ValidationError):
        hypervolume(problem, values, {"f1": 0.5, "f2": 0.5})


def test_duplicate_objective_names_rejected():
    with pytest.raises(ValidationError):
        Problem(Space(x=Real(0, 1)), [Minimize("y"), Maximize("y")])
    with pytest.raises(ValidationError):
        Constraint("stability")
