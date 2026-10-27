import json

import pytest

from molopt import Minimize, Problem, Real, Result, Space, StateError
from molopt.algorithms import GridSearch, LatinHypercube, NSGA2, RandomSearch
from molopt.benchmarks import evaluate_zdt1, zdt1_problem


def _trace(factory, evaluate, steps, batch=2, restore_at=None):
    optimizer = factory()
    log = []
    for step in range(steps):
        if step == restore_at:
            state = json.loads(json.dumps(optimizer.state_dict()))
            optimizer = factory()
            optimizer.load_state_dict(state)
        trials = optimizer.ask(batch)
        log.append([(trial.id, dict(trial.params)) for trial in trials])
        if not trials:
            continue
        optimizer.tell(
            [Result(trial.id, evaluate(trial.params)) for trial in trials]
        )
    return log


def test_sampler_checkpoints_round_trip_json():
    problem = Problem(Space(x=Real(0, 1), z=Real(0.01, 1, log=True)), [Minimize("y")])

    def evaluate(params):
        return {"y": params["x"] + params["z"]}

    cases = [
        (lambda: RandomSearch(problem, seed=4), evaluate),
        (lambda: LatinHypercube(problem, seed=4, n_samples=12), evaluate),
    ]
    grid_problem = Problem(
        Space(a=Real(0, 1, step=0.25), b=Real(0, 1, step=0.5)),
        [Minimize("y")],
    )
    cases.append(
        (
            lambda: GridSearch(grid_problem, seed=4),
            lambda params: {"y": params["a"] + params["b"]},
        )
    )
    for factory, objective in cases:
        direct = _trace(factory, objective, steps=4)
        restored = _trace(factory, objective, steps=4, restore_at=2)
        assert restored == direct


def test_nsga_checkpoint_matches_uninterrupted_run():
    problem = zdt1_problem(3)

    def factory():
        return NSGA2(problem, seed=9, population_size=6)

    direct = _trace(factory, evaluate_zdt1, steps=6, batch=3)
    restored = _trace(factory, evaluate_zdt1, steps=6, batch=3, restore_at=3)
    assert restored == direct


def test_incompatible_problem_is_rejected_without_changes():
    problem = Problem(Space(x=Real(0, 1)), [Minimize("y")])
    optimizer = RandomSearch(problem, seed=1)
    trial = optimizer.ask(1)[0]
    optimizer.tell(Result(trial.id, {"y": 0.2}))
    state = optimizer.state_dict()
    other = Problem(Space(x=Real(0, 2)), [Minimize("y")])
    fresh = RandomSearch(other, seed=1)
    with pytest.raises(StateError):
        fresh.load_state_dict(state)
    assert fresh.pareto_front() == []
    assert optimizer.best().values["y"] == pytest.approx(0.2)
