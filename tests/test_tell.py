import pytest

from molopt import (
    Constraint,
    FiniteSpace,
    Maximize,
    Minimize,
    MolOptError,
    Problem,
    Real,
    Result,
    Space,
    Status,
    TellError,
)
from molopt.algorithms import RandomSearch


def _single():
    return Problem(Space(x=Real(0, 1)), [Minimize("y")])


def _evaluate(params):
    return {"y": float(params["x"])}


def test_out_of_order_matches_in_order_best():
    problem = _single()
    forward = RandomSearch(problem, seed=1)
    backward = RandomSearch(problem, seed=1)
    trials = forward.ask(4)
    same = backward.ask(4)
    assert [trial.params["x"] for trial in trials] == [trial.params["x"] for trial in same]
    forward.tell([Result(trial.id, _evaluate(trial.params)) for trial in trials])
    backward.tell(
        [Result(trial.id, _evaluate(trial.params)) for trial in reversed(same)]
    )
    assert forward.best().values["y"] == backward.best().values["y"]
    assert forward.best().trial_id == backward.best().trial_id


def test_failure_is_not_a_poor_score_and_nan_is_rejected():
    problem = _single()
    optimizer = RandomSearch(problem, seed=2)
    first, second, third = optimizer.ask(3)
    optimizer.tell(Result(first.id, {"y": 0.2}))
    optimizer.tell(Result(second.id, status=Status.FAILED, failure="simulation crashed"))
    with pytest.raises(TellError):
        optimizer.tell(Result(third.id, {"y": float("nan")}))
    assert optimizer.best().values["y"] == pytest.approx(0.2)
    optimizer.tell(Result(third.id, {"y": 0.9}))
    assert optimizer.best().values["y"] == pytest.approx(0.2)
    more = optimizer.ask(1)
    assert more


def test_idempotent_tell_and_conflicts():
    optimizer = RandomSearch(_single(), seed=3)
    trial = optimizer.ask(1)[0]
    result = Result(trial.id, {"y": 0.4})
    optimizer.tell(result)
    optimizer.tell(result)
    assert len(optimizer.state_dict()["results"]) == 1
    with pytest.raises(TellError):
        optimizer.tell(Result(trial.id, {"y": 0.1}))
    with pytest.raises(TellError):
        optimizer.tell(Result("missing", {"y": 0.1}))


def test_infeasible_observation_is_kept_out_of_the_front():
    problem = Problem(
        Space(x=Real(0, 1)),
        [Maximize("conductivity"), Maximize("modulus")],
        [Constraint("stability", lower=0.8)],
    )
    optimizer = RandomSearch(problem, seed=4)
    low, high = optimizer.ask(2)
    optimizer.tell(
        [
            Result(high.id, {"conductivity": 1.0, "modulus": 1.0, "stability": 0.2}),
            Result(low.id, {"conductivity": 0.4, "modulus": 0.5, "stability": 0.9}),
        ]
    )
    front = optimizer.pareto_front()
    assert [solution.trial_id for solution in front] == [low.id]
    with pytest.raises(MolOptError):
        optimizer.best()


def test_finite_space_does_not_repeat_under_skip():
    space = FiniteSpace([{"x": value} for value in range(5)])
    problem = Problem(space, [Minimize("y")])
    optimizer = RandomSearch(problem, seed=5)
    seen = []
    while True:
        trials = optimizer.ask(2)
        if not trials:
            break
        seen.extend(trial.params["x"] for trial in trials)
        optimizer.tell([Result(trial.id, {"y": float(trial.params["x"])}) for trial in trials])
    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert optimizer.ask(1) == []
