"""Shared evaluation loops for the optimizer contract."""

from __future__ import annotations

from collections.abc import Callable

from molopt import Result


def evaluate_budget(optimizer, evaluate: Callable, budget: int, batch: int = 4) -> int:
    done = 0
    while done < budget:
        trials = optimizer.ask(min(batch, budget - done))
        if not trials:
            break
        optimizer.tell(
            [Result(trial_id=trial.id, values=evaluate(trial.params)) for trial in trials]
        )
        done += len(trials)
    return done
