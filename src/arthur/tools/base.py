"""Tool interfaces: every Arthur capability is a registered :class:`Tool`.

Design notes:

* Input schemas are Pydantic models, so validation is automatic and the JSON
  schema handed to the LLM is generated from the same source of truth.
* Tools declare a default permission level, but may refine it per call via
  :meth:`Tool.permission_for` (e.g. overwriting a file upgrades SAFE->CONFIRM).
* Tools never talk to the database or LLM directly; anything they need comes
  in through :class:`ToolContext`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from arthur.config.schema import Config
from arthur.security.paths import PathPolicy
from arthur.security.permissions import PermissionLevel

if TYPE_CHECKING:  # pragma: no cover
    from arthur.memory.store import MemoryStore
    from arthur.retrieval.service import Retriever


class ToolResult(BaseModel):
    """Uniform result envelope returned by every tool."""

    ok: bool = True
    summary: str = ""
    data: Any = None
    truncated: bool = False

    def render(self, max_chars: int = 8000) -> str:
        """Render as compact text for the model's observation window."""
        parts: list[str] = []
        if self.summary:
            parts.append(self.summary)
        if self.data is not None:
            try:
                parts.append(
                    json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                parts.append(str(self.data))
        text = "\n".join(parts)
        if self.truncated:
            text += "\n[output truncated]"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text


@dataclass
class ToolContext:
    """Services handed to tools at execution time."""

    config: Config
    paths: PathPolicy
    memory: MemoryStore | None = None
    retrieval: Retriever | None = None


class Tool(ABC):
    """Base class for all executable tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    #: Human phrasing shown in confirmation prompts, e.g. "Delete file".
    action: ClassVar[str] = ""
    permission: ClassVar[PermissionLevel] = PermissionLevel.SAFE
    risk: ClassVar[str] = "low"
    args_model: ClassVar[type[BaseModel]]
    result_model: ClassVar[type[BaseModel] | None] = None
    timeout: ClassVar[float] = 60.0

    def permission_for(self, args: BaseModel) -> PermissionLevel:
        """Dynamic permission for this call (default: the declared level)."""
        return type(self).permission

    def schema(self) -> dict[str, Any]:
        """JSON schema describing this tool for the LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }

    @abstractmethod
    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated arguments.

        Raises:
            ToolError / OSError / ValueError: mapped to an error outcome.
        """
