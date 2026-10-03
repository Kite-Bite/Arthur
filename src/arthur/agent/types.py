"""Agent-facing result types (what CLI and API consume)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from arthur.security.confirmation import ConfirmationRequest

StopReason = Literal[
    "answered",
    "max_steps",
    "confirmation_required",
    "llm_error",
    "parse_failure",
]


class AgentStep(BaseModel):
    """One visible step of the agent loop (for traces and explanations)."""

    index: int
    thought: str = ""
    plan: list[str] = Field(default_factory=list)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    status: str = ""
    permission: str = ""
    observation: str = ""
    duration_ms: float = 0.0
    error: str | None = None
    confirmation: ConfirmationRequest | None = None


class AgentResult(BaseModel):
    """Complete outcome of one agent run."""

    request: str
    request_id: str
    answer: str = ""
    conversation_id: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    removed_citations: list[str] = Field(default_factory=list)
    stopped_reason: StopReason = "answered"
    pending_confirmation: ConfirmationRequest | None = None
    error: str | None = None
    duration_ms: float = 0.0
