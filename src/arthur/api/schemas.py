"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat payload."""

    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(
        default=None, description="Continue an existing conversation"
    )
    confirm: bool = Field(
        default=False,
        description="Approve confirmation-required operations for this request",
    )


class ToolExecuteRequest(BaseModel):
    """POST /tools/{name}/execute payload."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    confirm: bool = Field(default=False, description="Approve the operation")


class MemoryCreateRequest(BaseModel):
    """POST /memory payload."""

    content: str = Field(min_length=1, max_length=4000)
    category: str = Field(default="general", max_length=64)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryItemResponse(BaseModel):
    """Serialized memory entry."""

    id: int
    content: str
    category: str
    source: str
    importance: float
    created_at: datetime
    expires_at: datetime | None = None


class IndexRequest(BaseModel):
    """POST /documents/index payload."""

    path: str
    recursive: bool = True


class ToolInfo(BaseModel):
    """Tool listing entry."""

    name: str
    description: str
    permission: str
    risk: str
    action: str
    parameters: dict[str, Any]


class LogEntry(BaseModel):
    """One audit-trail row."""

    id: int
    created_at: datetime
    request_id: str | None = None
    request: str | None = None
    tool_name: str
    arguments: dict[str, Any] | None = None
    permission: str
    confirmed: bool
    status: str
    result_summary: str | None = None
    error: str | None = None
    duration_ms: float


class ErrorResponse(BaseModel):
    """Uniform error payload."""

    detail: str
