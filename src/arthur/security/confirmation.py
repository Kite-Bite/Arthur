"""Confirmation protocol for risky operations.

The executor builds a :class:`ConfirmationRequest` whenever policy requires
one. How (and whether) the user is asked is decided by the caller:

* CLI    -> interactive prompt callback
* API    -> ``confirm: true`` in the request (or a pending response)
* tests  -> scripted callbacks
* default-> unresolved (reported as ``confirmation_required``)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ConfirmationRequest(BaseModel):
    """Describes an operation that needs explicit human approval."""

    tool_name: str
    action: str
    target: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    risk: str = "medium"

    def describe(self) -> str:
        """Multi-line human-readable rendering used by the CLI prompt."""
        lines = [f"Arthur wants to: {self.action}", f"Target: {self.target}"]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


#: Callback contract: return True to approve, False to decline.
ConfirmationCallback = Callable[[ConfirmationRequest], bool]


def deny(_request: ConfirmationRequest) -> bool:
    """Deny every confirmation (safe default when no UI is attached)."""
    return False


def approve(_request: ConfirmationRequest) -> bool:
    """Approve every confirmation (explicit ``--yes`` mode; still audited)."""
    return True
