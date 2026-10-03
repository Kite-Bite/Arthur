"""Exceptions raised inside tools (mapped to structured outcomes by the executor)."""

from __future__ import annotations


class ToolError(Exception):
    """Base class for controlled tool failures with user-facing messages."""

    def __init__(self, message: str, *, tool_name: str = "") -> None:
        self.tool_name = tool_name
        super().__init__(message)


class ToolTimeout(ToolError):
    """The tool exceeded its execution time budget."""
