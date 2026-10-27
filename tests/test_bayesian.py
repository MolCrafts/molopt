import importlib.util
import json

import numpy as np
import pytest

from molopt import (
    Choice,
    Constraint,
    FiniteSpace,
    Integer,
    Maximize,
    Minimize,
    Problem,
    Real,
    Result,
    Space,
    TellError,
)
from molopt.algorithms import BayesianOptimization, RandomSearch
from molopt.benchmarks import BRANIN_MINIMUM, evaluate_branin, evaluate_zdt1, zdt1_problem
from molopt.benchmarks.single_objective import branin_problem
from molopt.errors import CapabilityError
from molopt.metrics import hypervolume
from tests.conftest import evaluate_budget

needs_bo = pytest.mark.skipif(
    importlib.util.find_spec("botorch") is None,
    reason='Bayesian optimization extra is not installed; pip install "molopt[bo]"',
)


def _bo(problem, **kwargs):
    defaults = dict(
        seed=0,
        n_initial=4,
        noise=0.0,
        num_restarts=2,
        raw_samples=8,
        mc_samples=8,
        opt_maxiter=20,
    )
    defaults.update(kwargs)
    return BayesianOptimization(problem, **defaults)


def test_mixed_space_over_the_combination_limit_is_rejected():
    mixed = Problem(
        Space(x=Real(0, 1), kind=Choice([str(i) for i in range(20)])),
        [Minimize("y")],
    )
    with pytest.raises(CapabilityError, match="max_choice_combinations"):
        BayesianOptimization(mixed, seed=0, n_initial=2)


@needs_bo
def test_unknown_noise_is_rejected():
    problem = Problem(Space(x=Real(0, 1)), [Minimize("y")])
    optimizer = BayesianOptimization(problem, seed=0, n_initial=2, noise=None)
    trial = optimizer.ask(1)[0]
    with pytest.raises(TellError, match="noise"):
        optimizer.tell(Result(trial.id, {"y": 0.2}))


@needs_bo
def test_proposals_stay_in_finite_and_mixed_spaces():
    finite = Problem(
        FiniteSpace([{"a": left, "b": right} for left in (0.2, 0.8) for right in ("u", "v")]),
        [Minimize("y"), Maximize("z")],
        [Constraint("stable", lower=0.0)],
    )
    optimizer = _bo(finite, n_initial=2)
    evaluate_budget(
        optimizer,
        lambda params: {"y": params["a"], "z": 1.0, "stable": 1.0},
        budget=4,
        batch=2,
    )
    assert optimizer.pareto_front()
    for solution in optimizer.pareto_front():
        assert finite.space.contains(solution.params)

    mixed = Problem(
        Space(x=Real(0, 1), kind=Choice(["u", "v"])),
        [Minimize("y")],
    )
    optimizer = _bo(mixed, n_initial=4)
    trials = []
    done = 0
    while done < 6:
        batch = optimizer.ask(2)
        assert batch
        trials.extend(batch)
        optimizer.tell([Result(trial.id, {"y": trial.params["x"]}) for trial in batch])
        done += len(batch)
    assert all(mixed.space.contains(trial.params) for trial in trials)
    assert all(trial.params["kind"] in {"u", "v"} for trial in trials)


@needs_bo
def test_pending_batch_and_known_stderr():
    problem = Problem(Space(x=Real(-2, 2)), [Minimize("y")])
    optimizer = _bo(problem, n_initial=4)
    initial = optimizer.ask(4)
    optimizer.tell(
        [
            Result(trial.id, {"y": (trial.params["x"] - 0.3) ** 2}, stderr={"y": 0.01})
            for trial in initial
        ]
    )
    proposed = optimizer.ask(2)
    assert len(proposed) == 2
    assert proposed[0].params["x"] != pytest.approx(proposed[1].params["x"])
    optimizer.tell(Result(proposed[1].id, {"y": 0.4}, stderr={"y": 0.02}))
    optimizer.tell(Result(proposed[0].id, {"y": 0.2}, stderr={"y": 0.02}))
    assert optimizer.best() is not None


@needs_bo
def test_checkpoint_matches_uninterrupted_run():
    problem = Problem(Space(x=Real(-2, 2), z=Real(0.1, 1, log=True)), [Minimize("y")])

    def factory():
        return _bo(problem, seed=3)

    def evaluate(params):
        return {"y": (params["x"] - 0.2) ** 2 + params["z"]}

    def trace(restore_at):
        optimizer = factory()
        log = []
        for step in range(4):
            if step == restore_at:
                state = json.loads(json.dumps(optimizer.state_dict()))
                optimizer = factory()
                optimizer.load_state_dict(state)
            trials = optimizer.ask(2)
            log.append([(trial.id, dict(trial.params)) for trial in trials])
            optimizer.tell([Result(trial.id, evaluate(trial.params)) for trial in trials])
        return log

    direct = trace(None)
    assert trace(1) == direct
    assert trace(3) == direct


@needs_bo
def test_branin_beats_random_search():
    problem = branin_problem()
    bo_scores = []
    random_scores = []
    for seed in range(3):
        bo = BayesianOptimization(
            problem,
            seed=seed,
            n_initial=6,
            noise=0.0,
            num_restarts=3,
            raw_samples=16,
            mc_samples=16,
            opt_maxiter=40,
        )
        evaluate_budget(bo, evaluate_branin, budget=16, batch=1)
        rs = RandomSearch(problem, seed=seed)
        evaluate_budget(rs, evaluate_branin, budget=16, batch=1)
        bo_scores.append(bo.best().values["y"])
        random_scores.append(rs.best().values["y"])
    assert float(np.median(bo_scores)) < float(np.median(random_scores))
    assert float(np.median(bo_scores)) < 2.0
    assert min(bo_scores) < BRANIN_MINIMUM + 1.5


@needs_bo
def test_zdt1_front_uses_observed_values():
    problem = zdt1_problem(3)
    optimizer = BayesianOptimization(
        problem,
        seed=0,
        n_initial=6,
        noise=0.0,
        num_restarts=2,
        raw_samples=8,
        mc_samples=8,
        opt_maxiter=20,
    )
    evaluate_budget(optimizer, evaluate_zdt1, budget=8, batch=2)
    front = optimizer.pareto_front()
    assert front
    for solution in front:
        assert solution.values == pytest.approx(evaluate_zdt1(solution.params))
    reference = {"f1": 11.0, "f2": 11.0}
    assert hypervolume(problem, [dict(item.values) for item in front], reference) > 0
