"""An observed feasible candidate.

Solutions carry measured values. A model prediction is not a solution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Solution:
    trial_id: str
    params: Mapping[str, Any]
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
