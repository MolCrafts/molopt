"""Deterministic enumeration of a finite space."""

from __future__ import annotations

from typing import Any

import numpy as np

from molopt.errors import CapabilityError, StateError, ValidationError
from molopt.optimizer import Candidate, DuplicatePolicy, Optimizer
from molopt.problem import Problem


class GridSearch(Optimizer):
    """Walk ``space.enumerate()`` in definition order.

    The seed is stored so every optimizer accepts one, and it does not change
    the order. Points are not resampled.
    """

    algorithm_name = "grid"

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
        max_enumerate: int = 100_000,
    ) -> None:
        if isinstance(max_enumerate, bool) or not isinstance(max_enumerate, (int, np.integer)):
            raise ValidationError("max_enumerate must be a positive integer")
        if int(max_enumerate) < 1:
            raise ValidationError("max_enumerate must be a positive integer")
        self.max_enumerate = int(max_enumerate)
        super().__init__(problem, seed=seed, duplicates=duplicates)
        cardinality = problem.space.cardinality()
        if cardinality is None:
            raise CapabilityError("GridSearch requires a finite space")
        if cardinality > self.max_enumerate:
            raise CapabilityError(
                f"space cardinality {cardinality} exceeds max_enumerate={self.max_enumerate}"
            )
        self._grid = list(problem.space.enumerate())
        self._cursor = 0

    def _ask(self, n: int) -> list[Candidate]:
        chosen: list[Candidate] = []
        while len(chosen) < n and self._cursor < len(self._grid):
            params = self._grid[self._cursor]
            self._cursor += 1
            if self._is_blocked(params):
                continue
            chosen.append((params, None, {}))
        return chosen

    def _algorithm_state(self) -> dict[str, Any]:
        return {"cursor": self._cursor, "max_enumerate": self.max_enumerate}

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if state.get("max_enumerate") != self.max_enumerate:
            raise StateError("checkpoint max_enumerate does not match")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > len(self._grid):
            raise StateError("checkpoint grid cursor is out of range")
        self._cursor = cursor
