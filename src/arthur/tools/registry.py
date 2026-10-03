"""Tool registry: the single source of truth for what Arthur can execute."""

from __future__ import annotations

from typing import Any

from arthur.tools.base import Tool


class ToolRegistry:
    """Name-keyed registry of tool instances."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if not tool.name or not tool.name.isidentifier():
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if tool.name in self._tools and not replace:
            raise ValueError(f"tool already registered: {tool.name!r}")
        if not tool.description.strip():
            raise ValueError(f"tool {tool.name!r} needs a description")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name!r}")
        return tool

    def list(self) -> list[Tool]:
        """All tools, sorted by name for stable prompts and output."""
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.list()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def default_registry() -> ToolRegistry:
    """Build the registry with every built-in tool.

    Import of this module happens lazily inside the function so that tool
    modules (which import the executor's runner) never cycle back through the
    package importer.
    """
    from arthur.tools.documents import IndexDocumentsTool, SearchDocumentsTool
    from arthur.tools.files import (
        CopyFileTool,
        DeleteFileTool,
        ListFilesTool,
        MoveFileTool,
        ReadFileTool,
        SearchFilesTool,
        WriteFileTool,
    )
    from arthur.tools.git_tools import (
        GitBranchesTool,
        GitDiffTool,
        GitInfoTool,
        GitLogTool,
        GitStatusTool,
    )
    from arthur.tools.linux import RunCommandTool
    from arthur.tools.memory_tools import RecallTool, RememberTool
    from arthur.tools.system import (
        CpuInfoTool,
        DiskUsageTool,
        MemoryInfoTool,
        NetworkInfoTool,
        ProcessListTool,
        SystemInfoTool,
        UptimeTool,
    )

    registry = ToolRegistry()
    for tool in (
        # Files
        ListFilesTool(),
        SearchFilesTool(),
        ReadFileTool(),
        WriteFileTool(),
        CopyFileTool(),
        MoveFileTool(),
        DeleteFileTool(),
        # System
        SystemInfoTool(),
        CpuInfoTool(),
        MemoryInfoTool(),
        DiskUsageTool(),
        ProcessListTool(),
        NetworkInfoTool(),
        UptimeTool(),
        # Linux
        RunCommandTool(),
        # Git
        GitStatusTool(),
        GitLogTool(),
        GitBranchesTool(),
        GitDiffTool(),
        GitInfoTool(),
        # Documents / RAG
        IndexDocumentsTool(),
        SearchDocumentsTool(),
        # Memory
        RememberTool(),
        RecallTool(),
    ):
        registry.register(tool)
    return registry
