import numpy as np
import pytest

from molopt import Choice, Constant, Integer, Minimize, Problem, Real, Result, Space
from molopt.algorithms import GridSearch, LatinHypercube, RandomSearch
from molopt.errors import CapabilityError


def test_grid_order_is_definition_order():
    problem = Problem(
        Space(a=Integer(0, 1), b=Choice(["x", "y"])),
        [Minimize("y")],
    )
    optimizer = GridSearch(problem, seed=0)
    trials = []
    while True:
        batch = optimizer.ask(3)
        if not batch:
            break
        trials.extend(batch)
        optimizer.tell([Result(trial.id, {"y": 1.0}) for trial in batch])
    assert [(trial.params["a"], trial.params["b"]) for trial in trials] == [
        (0, "x"),
        (0, "y"),
        (1, "x"),
        (1, "y"),
    ]
    assert [trial.id for trial in trials] == ["0", "1", "2", "3"]


def test_grid_rejects_continuous_and_huge_spaces():
    problem = Problem(Space(x=Real(0, 1)), [Minimize("y")])
    with pytest.raises(CapabilityError):
        GridSearch(problem, seed=0)
    wide = Problem(Space(n=Integer(0, 100_001)), [Minimize("y")])
    with pytest.raises(CapabilityError):
        GridSearch(wide, seed=0, max_enumerate=1000)


def test_lhs_covers_strata_and_log_bounds():
    problem = Problem(
        Space(x=Real(0, 1), z=Real(0.01, 1.0, log=True), bias=Constant(1)),
        [Minimize("y")],
    )
    optimizer = LatinHypercube(problem, seed=7, n_samples=8)
    trials = optimizer.ask(8)
    assert optimizer.ask(1) == []
    xs = np.array([trial.params["x"] for trial in trials])
    logs = np.log([trial.params["z"] for trial in trials])
    assert len(set(np.floor(xs * 8).astype(int))) == 8
    log_low, log_high = np.log(0.01), np.log(1.0)
    scaled = (logs - log_low) / (log_high - log_low)
    assert len(set(np.floor(np.clip(scaled, 0, 0.999999) * 8).astype(int))) == 8
    assert all(trial.params["bias"] == 1 for trial in trials)
    assert all(0 <= trial.params["x"] < 1 for trial in trials)


def test_lhs_rejects_categories():
    problem = Problem(Space(x=Real(0, 1), kind=Choice(["a", "b"])), [Minimize("y")])
    with pytest.raises(CapabilityError):
        LatinHypercube(problem, seed=0, n_samples=4)


def test_random_seed_repeats():
    problem = Problem(Space(x=Real(-2, 2), n=Integer(0, 3)), [Minimize("y")])
    first = [dict(trial.params) for trial in RandomSearch(problem, seed=11).ask(5)]
    second = [dict(trial.params) for trial in RandomSearch(problem, seed=11).ask(5)]
    assert first == second
    assert first != [dict(trial.params) for trial in RandomSearch(problem, seed=12).ask(5)]
