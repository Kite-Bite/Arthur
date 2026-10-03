"""Registry invariants: uniqueness, schemas, completeness."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from arthur.security.permissions import PermissionLevel
from arthur.tools.base import Tool, ToolContext, ToolResult
from arthur.tools.registry import ToolRegistry, default_registry


class _Args(BaseModel):
    value: str


class _Tool(Tool):
    name = "demo_tool"
    description = "A demo tool"
    args_model = _Args

    def run(self, args: _Args, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        return ToolResult(summary="ok")


def test_register_and_get() -> None:
    registry = ToolRegistry()
    tool = _Tool()
    registry.register(tool)
    assert registry.get("demo_tool") is tool
    assert "demo_tool" in registry
    assert len(registry) == 1


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry()
    registry.register(_Tool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_Tool())
    registry.register(_Tool(), replace=True)  # explicit replacement allowed


def test_invalid_name_rejected() -> None:
    registry = ToolRegistry()

    class Bad(_Tool):
        name = "bad name!"

    with pytest.raises(ValueError, match="invalid tool name"):
        registry.register(Bad())


def test_description_required() -> None:
    registry = ToolRegistry()

    class Quiet(_Tool):
        description = "   "

    with pytest.raises(ValueError, match="description"):
        registry.register(Quiet())


def test_require_missing_raises() -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        ToolRegistry().require("nope")


def test_default_registry_is_complete() -> None:
    registry = default_registry()
    names = set(registry.names())
    expected = {
        "list_files",
        "search_files",
        "read_file",
        "write_file",
        "copy_file",
        "move_file",
        "delete_file",
        "system_info",
        "cpu_info",
        "memory_info",
        "disk_usage",
        "process_list",
        "network_info",
        "uptime",
        "run_command",
        "git_status",
        "git_log",
        "git_branches",
        "git_diff",
        "git_info",
        "index_documents",
        "search_documents",
        "remember",
        "recall",
    }
    assert names == expected


def test_every_tool_has_schema_and_permission() -> None:
    registry = default_registry()
    for tool in registry.list():
        schema = tool.schema()
        assert schema["name"] == tool.name
        assert schema["description"].strip()
        assert "properties" in schema["parameters"]
        assert isinstance(tool.permission, PermissionLevel)
        assert tool.action


def test_destructive_tools_are_not_safe() -> None:
    registry = default_registry()
    assert registry.require("delete_file").permission >= PermissionLevel.CONFIRM
    assert registry.require("move_file").permission >= PermissionLevel.CONFIRM
    assert registry.require("run_command").permission >= PermissionLevel.CONFIRM
    assert registry.require("read_file").permission == PermissionLevel.SAFE
    assert registry.require("system_info").permission == PermissionLevel.SAFE
