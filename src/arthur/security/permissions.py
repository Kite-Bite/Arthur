"""Permission levels and decisions.

The security model has four levels:

* ``SAFE``       - no friction, runs immediately.
* ``CONFIRM``    - runs after an explicit human confirmation.
* ``RESTRICTED`` - denied unless the operator opts in via configuration.
* ``DENIED``     - never runs (arbitrary shell lives here by default).
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel


class PermissionLevel(IntEnum):
    """Ordered from least to most restrictive (higher value = more risky)."""

    SAFE = 0
    CONFIRM = 1
    RESTRICTED = 2
    DENIED = 3

    @classmethod
    def parse(cls, value: str | PermissionLevel) -> PermissionLevel:
        """Parse a level from its name (case-insensitive) or return as-is."""
        if isinstance(value, PermissionLevel):
            return value
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown permission level {value!r}; expected one of "
                f"{', '.join(level.name for level in cls)}"
            ) from exc

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


class PermissionDecision(BaseModel):
    """The outcome of evaluating one (tool, arguments) pair against policy."""

    level: PermissionLevel
    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""

    @classmethod
    def allow(
        cls,
        level: PermissionLevel,
        *,
        confirm: bool = False,
        reason: str = "",
    ) -> PermissionDecision:
        return cls(
            level=level,
            allowed=True,
            requires_confirmation=confirm,
            reason=reason or "allowed",
        )

    @classmethod
    def deny(cls, level: PermissionLevel, reason: str) -> PermissionDecision:
        return cls(level=level, allowed=False, requires_confirmation=False, reason=reason)
