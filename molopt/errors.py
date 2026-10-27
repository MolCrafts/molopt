"""Public exceptions."""

from __future__ import annotations


class MolOptError(Exception):
    """Base error for MolOpt."""


class ValidationError(MolOptError):
    """A space, problem, parameter, or value is invalid."""


class CapabilityError(MolOptError):
    """An algorithm does not support this problem."""


class TellError(MolOptError):
    """A result cannot be accepted."""


class StateError(MolOptError):
    """A checkpoint is incompatible with this optimizer."""
