"""Memory tools: explicit recall/store operations.

Arthur never stores conversations automatically as long-term memory; the model
must call ``remember`` deliberately (and the user can see, search and delete
every entry with ``arthur memory``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.tools.base import Tool, ToolContext, ToolResult


class RememberArgs(BaseModel):
    content: str = Field(min_length=1, max_length=4000, description="Fact to remember")
    category: str = Field("general", max_length=64, description="Category, e.g. 'preference'")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="0 (low) to 1 (high)")


class RememberTool(Tool):
    name = "remember"
    description = (
        "Store a durable fact in Arthur's long-term memory for future "
        "sessions. Use only for facts the user asked to remember or that "
        "will clearly matter later; never store secrets."
    )
    action = "Store a long-term memory"
    args_model = RememberArgs

    def run(self, args: RememberArgs, ctx: ToolContext) -> ToolResult:
        if ctx.memory is None:
            raise ToolError("memory service is not available")
        sensitive = ("password", "secret", "api key", "token")
        if any(word in args.content.casefold() for word in sensitive):
            raise ToolError("refusing to store what looks like a secret (password/token/api key)")
        entry = ctx.memory.add(
            content=args.content,
            category=args.category,
            source="agent",
            importance=args.importance,
        )
        return ToolResult(
            summary=f"remembered (memory #{entry.id}): {entry.content[:80]}",
            data={"id": entry.id, "category": entry.category, "created_at": entry.created_at},
        )


class RecallArgs(BaseModel):
    query: str = Field(min_length=1, description="What to look for")
    limit: int = Field(5, ge=1, le=20)


class RecallTool(Tool):
    name = "recall"
    description = "Search Arthur's long-term memory for stored facts."
    action = "Search long-term memory"
    args_model = RecallArgs

    def run(self, args: RecallArgs, ctx: ToolContext) -> ToolResult:
        if ctx.memory is None:
            raise ToolError("memory service is not available")
        entries = ctx.memory.search(args.query, limit=args.limit)
        data = [
            {
                "id": e.id,
                "content": e.content,
                "category": e.category,
                "importance": e.importance,
                "created_at": e.created_at,
            }
            for e in entries
        ]
        return ToolResult(
            summary=f"{len(data)} memories matched" if data else "no memories matched",
            data={"count": len(data), "memories": data},
        )
