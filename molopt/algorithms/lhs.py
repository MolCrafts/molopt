"""Latin hypercube designs for continuous boxes.

The design itself comes from :class:`scipy.stats.qmc.LatinHypercube`. MolOpt
owns the bounds, the log-space mapping, and the ask–tell cursor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import qmc

from molopt.errors import CapabilityError, StateError, ValidationError
from molopt.optimizer import Candidate, DuplicatePolicy, Optimizer
from molopt.problem import Problem
from molopt.space import Constant, Real, Space


class LatinHypercube(Optimizer):
    """One scrambled Latin hypercube, then no further suggestions."""

    algorithm_name = "latin_hypercube"

    def __init__(
        self,
        problem: Problem,
        *,
        seed: int,
        n_samples: int,
        duplicates: DuplicatePolicy = DuplicatePolicy.SKIP,
    ) -> None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise ValidationError("n_samples must be a positive integer")
        if int(n_samples) < 1:
            raise ValidationError("n_samples must be a positive integer")
        self.n_samples = int(n_samples)
        super().__init__(problem, seed=seed, duplicates=duplicates)
        space = problem.space
        if not isinstance(space, Space):
            raise CapabilityError("LatinHypercube requires a product of Real and Constant")
        self._reals: list[tuple[str, Real]] = []
        for name, variable in space.variables.items():
            if isinstance(variable, Constant):
                continue
            if not isinstance(variable, Real) or variable.step is not None:
                raise CapabilityError(
                    "LatinHypercube supports continuous Real variables and Constant only"
                )
            self._reals.append((name, variable))
        if not self._reals:
            raise CapabilityError("LatinHypercube requires at least one Real variable")
        self._design: list[dict[str, Any]] | None = None
        self._cursor = 0

    def _ask(self, n: int) -> list[Candidate]:
        self._ensure_design()
        assert self._design is not None
        chosen: list[Candidate] = []
        while len(chosen) < n and self._cursor < len(self._design):
            params = self._design[self._cursor]
            self._cursor += 1
            if self._is_blocked(params):
                continue
            chosen.append((dict(params), None, {}))
        return chosen

    def _ensure_design(self) -> None:
        if self._design is not None:
            return
        engine = qmc.LatinHypercube(
            d=len(self._reals),
            scramble=True,
            strength=1,
            optimization=None,
            seed=self.seed,
        )
        unit = np.asarray(engine.random(self.n_samples), dtype=np.float64)
        space = self.problem.space
        assert isinstance(space, Space)
        columns: list[np.ndarray] = []
        for index, (_, variable) in enumerate(self._reals):
            column = unit[:, index]
            if variable.log:
                log_low = math.log(variable.low)
                log_high = math.log(variable.high)
                mapped = np.exp(log_low + column * (log_high - log_low))
            else:
                mapped = variable.low + column * (variable.high - variable.low)
            columns.append(mapped)
        design: list[dict[str, Any]] = []
        for row in range(self.n_samples):
            params: dict[str, Any] = {}
            for name, variable in space.variables.items():
                if isinstance(variable, Constant):
                    params[name] = variable.value
            for index, (name, _) in enumerate(self._reals):
                params[name] = float(columns[index][row])
            design.append(space.validate(params))
        self._design = design

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "cursor": self._cursor,
            "design": self._design,
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if state.get("n_samples") != self.n_samples:
            raise StateError("checkpoint n_samples does not match")
        design = state.get("design")
        if design is not None:
            self._design = [dict(row) for row in design]
        else:
            self._design = None
        self._cursor = int(state["cursor"])
