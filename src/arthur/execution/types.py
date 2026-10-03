"""Execution-layer value types shared by executor, audit and API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from arthur.security.confirmation import ConfirmationRequest
from arthur.security.permissions import PermissionLevel
from arthur.tools.base import ToolResult

ExecutionStatus = Literal[
    "success",
    "error",
    "denied",
    "declined",
    "confirmation_required",
    "invalid_args",
    "timeout",
    "unknown_tool",
]


class ExecutionRecord(BaseModel):
    """One auditable tool execution (persisted and logged)."""

    request_id: str | None = None
    request: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    permission: PermissionLevel = PermissionLevel.SAFE
    confirmed: bool = False
    status: ExecutionStatus = "success"
    result_summary: str | None = None
    error: str | None = None
    duration_ms: float = 0.0


#: Audit sink contract: accepts one execution record (no return value).
AuditSink = Callable[[ExecutionRecord], None]


class ExecutionOutcome(BaseModel):
    """The complete result of an executor invocation."""

    record: ExecutionRecord
    result: ToolResult | None = None
    request: ConfirmationRequest | None = None

    @property
    def status(self) -> ExecutionStatus:
        return self.record.status

    @property
    def ok(self) -> bool:
        return self.record.status == "success"

    def observation(self) -> str:
        """Render the outcome as text for the model's context window."""
        status = self.record.status
        if status == "success" and self.result is not None:
            return self.result.render()
        if status == "confirmation_required" and self.request is not None:
            return (
                "CONFIRMATION REQUIRED - the user must approve this operation "
                "before it runs:\n" + self.request.describe()
            )
        if status == "declined":
            return "The user declined this operation. It was NOT executed."
        if status == "invalid_args":
            return f"INVALID ARGUMENTS: {self.record.error}"
        if status == "unknown_tool":
            return f"UNKNOWN TOOL: {self.record.error}"
        if status == "timeout":
            return f"TIMEOUT: {self.record.error or 'tool exceeded its time limit'}"
        if status == "denied":
            return f"PERMISSION DENIED: {self.record.error}"
        return f"ERROR: {self.record.error or 'unknown failure'}"
