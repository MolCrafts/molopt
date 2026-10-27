"""Uniform random search over any v0.1 space."""

from __future__ import annotations

from typing import Any

import numpy as np

from molopt.errors import ValidationError
from molopt.optimizer import Candidate, Optimizer
from molopt.problem import Problem
from molopt.optimizer import DuplicatePolicy


class RandomSearch(Optimizer):
    """Draw valid candidates until the duplicate policy or the space stops it."""

    algorithm_name = "random"

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
        max_attempts: int = 1000,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, (int, np.integer)):
            raise ValidationError("max_attempts must be a positive integer")
        if int(max_attempts) < 1:
            raise ValidationError("max_attempts must be a positive integer")
        self.max_attempts = int(max_attempts)
        super().__init__(problem, seed=seed, duplicates=duplicates)

    def _ask(self, n: int) -> list[Candidate]:
        cardinality = self.problem.space.cardinality()
        if (
            cardinality is not None
            and self.duplicates is DuplicatePolicy.SKIP
            and len(self._blocked) >= cardinality
        ):
            return []
        chosen: list[Candidate] = []
        seen = set(self._blocked)
        attempts = 0
        limit = self.max_attempts * n
        while len(chosen) < n and attempts < limit:
            attempts += 1
            params = self.problem.space.sample(1, self._rng)[0]
            identity = self.problem.space.identity(params)
            if identity in seen and self.duplicates is DuplicatePolicy.SKIP:
                continue
            if self.duplicates is DuplicatePolicy.ALLOW and identity in {
                self.problem.space.identity(item[0]) for item in chosen
            }:
                pass
            seen.add(identity)
            chosen.append((params, None, {}))
        return chosen

    def _algorithm_state(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts}

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if state.get("max_attempts") != self.max_attempts:
            from molopt.errors import StateError

            raise StateError("checkpoint max_attempts does not match")
