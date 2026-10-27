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
    Status,
)
from molopt.algorithms import NSGA2, RandomSearch
from molopt.benchmarks import evaluate_zdt1, igd, zdt1_problem, zdt1_reference_front
from molopt.metrics import hypervolume
from tests.conftest import evaluate_budget


def test_offspring_stay_inside_mixed_and_finite_spaces():
    mixed = Problem(
        Space(
            x=Real(0, 1),
            n=Integer(0, 5, step=2),
            kind=Choice(["a", "b", "c"]),
        ),
        [Minimize("f1"), Minimize("f2")],
    )
    optimizer = NSGA2(mixed, seed=0, population_size=8)
    for _ in range(4):
        trials = optimizer.ask(8)
        assert trials
        assert all(mixed.space.contains(trial.params) for trial in trials)
        optimizer.tell(
            [
                Result(
                    trial.id,
                    {
                        "f1": float(trial.params["x"] + trial.params["n"]),
                        "f2": float(len(trial.params["kind"])),
                    },
                )
                for trial in trials
            ]
        )

    library = [{"x": value} for value in (0.1, 0.4, 0.9)]
    finite = Problem(FiniteSpace(library), [Minimize("y"), Maximize("z")])
    optimizer = NSGA2(finite, seed=1, population_size=6)
    seen = []
    for _ in range(3):
        trials = optimizer.ask(6)
        seen.extend(trial.params["x"] for trial in trials)
        optimizer.tell(
            [
                Result(trial.id, {"y": trial.params["x"], "z": 1.0 - trial.params["x"]})
                for trial in trials
            ]
        )
    assert set(seen).issubset({0.1, 0.4, 0.9})


def test_failed_member_is_not_ranked_as_a_penalty():
    problem = Problem(Space(x=Real(0, 1)), [Minimize("y")])
    optimizer = NSGA2(problem, seed=2, population_size=4)
    trials = optimizer.ask(4)
    results = []
    for index, trial in enumerate(trials):
        if index == 0:
            results.append(Result(trial.id, status=Status.FAILED, failure="no output"))
        else:
            results.append(Result(trial.id, {"y": 0.2 + index}))
    optimizer.tell(results)
    assert optimizer.best().values["y"] == pytest.approx(0.2 + 1)
    assert optimizer.ask(4)


def test_constrained_front_drops_infeasible_measurements():
    problem = Problem(
        Space(x=Real(0, 1)),
        [Minimize("y")],
        [Constraint("stable", lower=0.5)],
    )
    optimizer = NSGA2(problem, seed=3, population_size=4)
    trials = optimizer.ask(4)
    optimizer.tell(
        [
            Result(trial.id, {"y": float(index), "stable": 0.0 if index == 0 else 1.0})
            for index, trial in enumerate(trials)
        ]
    )
    front = optimizer.pareto_front()
    assert all(solution.values["stable"] >= 0.5 for solution in front)
    assert all(solution.values["y"] != 0.0 for solution in front)


def test_zdt1_beats_random_search_over_seeds():
    problem = zdt1_problem(4)
    # Strictly worse than every ZDT1 point: f1 is at most 1 and f2 is at most 10.
    reference = {"f1": 1.1, "f2": 11.0}
    front = zdt1_reference_front(101)
    nsga_hv = []
    random_hv = []
    nsga_igd = []
    random_igd = []
    for seed in range(5):
        nsga = NSGA2(problem, seed=seed, population_size=20)
        evaluate_budget(nsga, evaluate_zdt1, budget=320, batch=20)
        observed = [dict(solution.values) for solution in nsga.pareto_front()]
        assert observed
        assert all(0.0 <= item["f1"] <= 1.0 for item in observed)
        assert all(item["f2"] >= 1.0 - item["f1"] ** 0.5 - 1e-9 for item in observed)
        nsga_hv.append(hypervolume(problem, observed, reference))
        nsga_igd.append(igd(problem, observed, front))

        baseline = RandomSearch(problem, seed=seed)
        evaluate_budget(baseline, evaluate_zdt1, budget=320, batch=20)
        random_observed = [dict(solution.values) for solution in baseline.pareto_front()]
        random_hv.append(hypervolume(problem, random_observed, reference))
        random_igd.append(igd(problem, random_observed, front))
    assert float(np.median(nsga_hv)) > float(np.median(random_hv))
    assert float(np.median(nsga_igd)) < float(np.median(random_igd))
    assert float(np.median(nsga_igd)) < 0.40
