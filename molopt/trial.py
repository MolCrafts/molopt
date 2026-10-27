"""A proposed evaluation.

A trial is not an experiment run. The caller may need several simulations or
replicates to produce one result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Trial:
    id: str
    params: Mapping[str, Any]
    fidelity: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or self.id == "":
            raise ValueError("trial id must be a non-empty string")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        if self.fidelity is not None:
            object.__setattr__(self, "fidelity", float(self.fidelity))
