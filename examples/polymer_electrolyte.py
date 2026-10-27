"""Coarse zwitterionic polymer electrolyte demonstrator.

The candidate set is an explicit library. The oracle is a deterministic
stand-in used to compare algorithms through one problem and one evaluator.
It is not a physical simulator and it is not part of MolOpt's dependency graph.
Reported fronts are measured values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

try:
    import molopt  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molopt import Constraint, FiniteSpace, Maximize, Problem, Result
from molopt.algorithms import NSGA2, RandomSearch
from molopt.metrics import hypervolume


def candidates() -> list[dict]:
    fractions = (0.2, 0.35, 0.5, 0.65, 0.8)
    architectures = ("random", "gradient", "block")
    salts = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    return [
        {"mil_fraction": fraction, "architecture": architecture, "salt": salt}
        for fraction in fractions
        for architecture in architectures
        for salt in salts
    ]


def polymer_problem() -> Problem:
    return Problem(
        space=FiniteSpace(candidates()),
        objectives=[Maximize("conductivity"), Maximize("modulus")],
        constraints=[Constraint("stability", lower=0.8)],
    )


def evaluate(params) -> dict[str, float]:
    fraction = float(params["mil_fraction"])
    architecture = params["architecture"]
    salt = float(params["salt"])
    conductivity_factor = {"random": 0.8, "gradient": 1.0, "block": 1.15}[architecture]
    modulus_factor = {"random": 1.0, "gradient": 0.9, "block": 1.1}[architecture]
    conductivity = (
        conductivity_factor
        * math.exp(-((salt - 0.35) ** 2) / 0.02)
        * (1.15 - fraction)
    )
    modulus = modulus_factor * (1.4 - salt) * (0.4 + fraction)
    stability = 1.05 - 1.4 * abs(fraction - 0.5) - 0.9 * max(0.0, salt - 0.45)
    return {
        "conductivity": conductivity,
        "modulus": modulus,
        "stability": stability,
    }


REFERENCE = {"conductivity": 0.0, "modulus": 0.0}


def run_optimizer(optimizer, budget: int, batch: int) -> list[dict]:
    done = 0
    while done < budget:
        trials = optimizer.ask(min(batch, budget - done))
        if not trials:
            break
        optimizer.tell(
            [Result(trial.id, evaluate(trial.params)) for trial in trials]
        )
        done += len(trials)
    return [dict(solution.values) for solution in optimizer.pareto_front()]


def _factories(seed: int):
    problem = polymer_problem()
    factories: list[tuple[str, Callable]] = [
        ("random", lambda problem=problem, seed=seed: RandomSearch(problem, seed=seed)),
        (
            "nsga2",
            lambda problem=problem, seed=seed: NSGA2(problem, seed=seed, population_size=10),
        ),
    ]
    if importlib.util.find_spec("botorch") is not None:
        from molopt.algorithms import BayesianOptimization

        factories.append(
            (
                "bayesian",
                lambda problem=problem, seed=seed: BayesianOptimization(
                    problem,
                    seed=seed,
                    n_initial=8,
                    noise=0.0,
                    num_restarts=3,
                    raw_samples=16,
                    mc_samples=16,
                    opt_maxiter=40,
                ),
            )
        )
    return factories


def compare(budget: int, seeds: list[int], batch: int) -> dict:
    summary = {"budget": budget, "seeds": seeds, "algorithms": {}}
    names = [name for name, _ in _factories(seeds[0])]
    for name in names:
        volumes = []
        for seed in seeds:
            factory = dict(_factories(seed))[name]
            front = run_optimizer(factory(), budget, batch)
            volumes.append(
                hypervolume(polymer_problem(), front, REFERENCE) if front else 0.0
            )
        summary["algorithms"][name] = {
            "median_hypervolume": statistics.median(volumes),
            "hypervolumes": volumes,
        }
    return summary


def check_fronts(budget: int, seed: int) -> None:
    """Every reported value equals the oracle at that candidate."""
    for name, factory in _factories(seed):
        optimizer = factory()
        done = 0
        while done < budget:
            trials = optimizer.ask(min(4, budget - done))
            if not trials:
                break
            optimizer.tell([Result(trial.id, evaluate(trial.params)) for trial in trials])
            done += len(trials)
        for solution in optimizer.pareto_front():
            measured = evaluate(solution.params)
            for key, value in solution.values.items():
                if not math.isclose(value, measured[key], rel_tol=0, abs_tol=1e-12):
                    raise SystemExit(f"{name} reported a value that is not the oracle")
            if not polymer_problem().feasible(solution.values):
                raise SystemExit(f"{name} put an infeasible point on the front")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if len(candidates()) != 105:
        raise SystemExit("candidate library changed size")
    if args.check:
        check_fronts(args.budget, seed=0)
    summary = compare(args.budget, list(range(args.seeds)), args.batch)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
