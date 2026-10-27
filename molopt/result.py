"""Outcome of evaluating one trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from molopt.errors import ValidationError


class Status(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Result:
    """Measured outcome.

    ``stderr`` is the standard error of a reported estimate. ``None`` means the
    noise is unknown. It does not mean the noise is zero. A failed or cancelled
    result is not a low objective value; leave ``values`` empty.
    """

    trial_id: str
    values: Mapping[str, Any] | None = None
    stderr: Mapping[str, Any] | None = None
    cost: float | None = None
    status: Status | str = Status.COMPLETED
    failure: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or self.trial_id == "":
            raise ValidationError("trial_id must be a non-empty string")
        status = self.status
        if isinstance(status, str):
            try:
                status = Status(status)
            except ValueError as exc:
                raise ValidationError(f"unknown status {status!r}") from exc
        if not isinstance(status, Status):
            raise ValidationError("status must be a Status or its value")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", _freeze_map(self.values))
        object.__setattr__(self, "stderr", _freeze_map(self.stderr))
        if self.cost is not None:
            object.__setattr__(self, "cost", float(self.cost))
        if self.failure is not None and not isinstance(self.failure, str):
            raise ValidationError("failure must be a string or None")


def _freeze_map(values: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValidationError("values and stderr must be mappings or None")
    return MappingProxyType(dict(values))
