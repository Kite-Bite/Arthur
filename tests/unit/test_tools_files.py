"""File tools lifecycle inside the sandbox (direct tool execution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.config.schema import Config
from arthur.execution.errors import ToolError
from arthur.security.paths import PathPolicy
from arthur.security.permissions import PermissionLevel
from arthur.tools.base import ToolContext
from arthur.tools.files import (
    CopyFileTool,
    DeleteFileTool,
    ListFilesTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    config = Config()
    config.security.allowed_roots = [str(tmp_path)]
    return ToolContext(config=config, paths=PathPolicy(config.security, cwd=tmp_path))


def test_write_read_roundtrip(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    result = WriteFileTool().run(
        WriteFileTool.args_model(path=str(target), content="hello arthur"), ctx
    )
    assert result.ok and target.read_text() == "hello arthur"

    read = ReadFileTool().run(ReadFileTool.args_model(path=str(target)), ctx)
    assert "hello arthur" in read.data["content"]
    assert read.data["total_lines"] == 1


def test_write_refuses_overwrite_without_flag(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(ToolError, match="overwrite=true"):
        WriteFileTool().run(
            WriteFileTool.args_model(path=str(target), content="new"), ctx
        )
    assert target.read_text() == "original"


def test_write_permission_upgrades_on_overwrite(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    tool = WriteFileTool()
    fresh = tool.args_model(path=str(target), content="x")
    assert tool.permission_for(fresh) == PermissionLevel.SAFE
    target.write_text("data", encoding="utf-8")
    existing = tool.args_model(path=str(target), content="x", overwrite=True)
    assert tool.permission_for(existing) == PermissionLevel.CONFIRM


def test_read_offset_limit(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "lines.txt"
    target.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    result = ReadFileTool().run(
        ReadFileTool.args_model(path=str(target), offset=3, limit=2), ctx
    )
    assert result.data["returned_lines"] == 2
    assert "line3" in result.data["content"] and "line4" in result.data["content"]
    assert "line5" not in result.data["content"]


def test_read_binary_refused(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(ToolError, match="binary"):
        ReadFileTool().run(ReadFileTool.args_model(path=str(target)), ctx)


def test_read_missing_file(ctx: ToolContext, tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        ReadFileTool().run(ReadFileTool.args_model(path=str(tmp_path / "nope")), ctx)


def test_list_files(ctx: ToolContext, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    result = ListFilesTool().run(
        ListFilesTool.args_model(directory=str(tmp_path), pattern="*.py"), ctx
    )
    assert result.data["count"] == 1
    assert result.data["entries"][0]["name"] == "a.py"


def test_list_hides_denied_dirs(ctx: ToolContext, tmp_path: Path) -> None:
    (tmp_path / ".ssh").mkdir()
    result = ListFilesTool().run(ListFilesTool.args_model(directory=str(tmp_path)), ctx)
    names = [entry["name"] for entry in result.data["entries"]]
    assert ".ssh" not in names


def test_list_recursive_respects_limit(ctx: ToolContext, tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    result = ListFilesTool().run(
        ListFilesTool.args_model(directory=str(tmp_path), recursive=True, limit=5), ctx
    )
    assert result.data["count"] == 5
    assert result.data["truncated"] is True


def test_search_by_name(ctx: ToolContext, tmp_path: Path) -> None:
    (tmp_path / "ProjectAlpha").mkdir()
    (tmp_path / "ProjectAlpha" / "main.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    result = SearchFilesTool().run(
        SearchFilesTool.args_model(root=str(tmp_path), glob="*.py"), ctx
    )
    assert result.data["count"] == 1
    assert result.data["files"][0]["name"] == "main.py"


def test_search_by_content(ctx: ToolContext, tmp_path: Path) -> None:
    (tmp_path / "linux.md").write_text(
        "# Notes\n\nbtrfs snapshot delete is dangerous\n", encoding="utf-8"
    )
    (tmp_path / "other.md").write_text("nothing here", encoding="utf-8")
    result = SearchFilesTool().run(
        SearchFilesTool.args_model(root=str(tmp_path), content_query="SNAPSHOT delete"),
        ctx
    )
    assert result.data["count"] == 1
    assert "linux.md" in result.data["files"][0]["path"]
    assert result.data["files"][0]["matches"]


def test_search_requires_criteria(ctx: ToolContext, tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="at least one"):
        SearchFilesTool().run(SearchFilesTool.args_model(root=str(tmp_path)), ctx)


def test_copy_and_move(ctx: ToolContext, tmp_path: Path) -> None:
    source = tmp_path / "src.txt"
    source.write_text("payload", encoding="utf-8")

    CopyFileTool().run(
        CopyFileTool.args_model(source=str(source), destination=str(tmp_path / "copy.txt")),
        ctx,
    )
    assert (tmp_path / "copy.txt").read_text() == "payload"
    assert source.exists()

    MoveFileTool().run(
        MoveFileTool.args_model(source=str(source), destination=str(tmp_path / "moved.txt")),
        ctx,
    )
    assert not source.exists()
    assert (tmp_path / "moved.txt").read_text() == "payload"


def test_move_permission_is_confirm(ctx: ToolContext, tmp_path: Path) -> None:
    args = MoveFileTool.args_model(source="a", destination="b")
    assert MoveFileTool().permission_for(args) == PermissionLevel.CONFIRM


def test_delete_file(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "trash.txt"
    target.write_text("bye", encoding="utf-8")
    DeleteFileTool().run(DeleteFileTool.args_model(path=str(target)), ctx)
    assert not target.exists()


def test_delete_directory_requires_recursive(ctx: ToolContext, tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "f.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="recursive=true"):
        DeleteFileTool().run(DeleteFileTool.args_model(path=str(folder)), ctx)
    DeleteFileTool().run(
        DeleteFileTool.args_model(path=str(folder), recursive=True), ctx
    )
    assert not folder.exists()
