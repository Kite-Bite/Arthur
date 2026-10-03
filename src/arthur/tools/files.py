"""File tools: list, search, read, write, copy, move, delete.

Every path is resolved through the :class:`PathPolicy` in the tool context
(the executor also pre-validates path arguments centrally). Deletion and moves
are CONFIRM-level; writing over an existing file upgrades itself to CONFIRM.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.security.permissions import PermissionLevel
from arthur.tools.base import Tool, ToolContext, ToolResult

MAX_READ_BYTES = 2_000_000
MAX_CONTENT_MATCHES = 100
MAX_SEARCH_FILE_BYTES = 512_000


def _walk_files(root: Path, denied_dir_names: set[str]) -> Iterator[Path]:
    """Yield files under ``root``, pruning denied directories and symlinks."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in denied_dir_names]
        for name in filenames:
            yield Path(dirpath) / name


def _entry(path: Path) -> dict[str, object]:
    entry: dict[str, object] = {"path": str(path), "name": path.name}
    try:
        stat = path.lstat()
    except OSError:
        entry["type"] = "unknown"
        return entry
    if path.is_symlink():
        entry["type"] = "symlink"
    elif path.is_dir():
        entry["type"] = "dir"
    else:
        entry["type"] = "file"
        entry["size"] = stat.st_size
    entry["modified"] = round(stat.st_mtime, 0)
    return entry


def _read_text(path: Path, max_bytes: int = MAX_READ_BYTES) -> tuple[str, bool]:
    """Read a UTF-ish text file, refusing obvious binaries."""
    try:
        raw = path.read_bytes()[: max_bytes + 1]
    except OSError as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    if b"\x00" in raw[:8192]:
        raise ToolError(f"{path} looks like a binary file; refusing to read it as text")
    return raw.decode("utf-8", errors="replace"), truncated


# --- list_files ----------------------------------------------------------------


class ListFilesArgs(BaseModel):
    directory: str = Field(description="Directory to list (absolute or relative to cwd)")
    pattern: str | None = Field(None, description="fnmatch glob for names, e.g. '*.py'")
    recursive: bool = Field(False, description="Recurse into subdirectories")
    limit: int = Field(100, ge=1, le=1000, description="Maximum entries returned")


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List the contents of a directory with file types, sizes and mtimes. "
        "Set recursive=true to walk subdirectories."
    )
    action = "List files in a directory"
    args_model = ListFilesArgs

    def run(self, args: ListFilesArgs, ctx: ToolContext) -> ToolResult:
        root = ctx.paths.resolve(args.directory)
        if not root.exists():
            raise ToolError(f"directory not found: {root}")
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")

        entries: list[dict[str, object]] = []
        truncated = False
        if args.recursive:
            for path in _walk_files(root, ctx.paths.denied_dir_names):
                if args.pattern and not fnmatch.fnmatch(path.name, args.pattern):
                    continue
                entries.append(_entry(path))
                if len(entries) >= args.limit:
                    truncated = True
                    break
        else:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
            for path in children:
                if path.name in ctx.paths.denied_dir_names:
                    continue  # never expose denied directories
                if args.pattern and not fnmatch.fnmatch(path.name, args.pattern):
                    continue
                entries.append(_entry(path))
                if len(entries) >= args.limit:
                    truncated = True
                    break

        return ToolResult(
            summary=f"{len(entries)} entries in {root}"
            + (" (truncated)" if truncated else ""),
            data={"directory": str(root), "count": len(entries), "truncated": truncated, "entries": entries},
        )


# --- search_files ---------------------------------------------------------------


class SearchFilesArgs(BaseModel):
    root: str = Field(".", description="Directory to search under")
    name_contains: str | None = Field(None, description="Case-insensitive name substring")
    glob: str | None = Field(None, description="fnmatch pattern for file names, e.g. '*.md'")
    content_query: str | None = Field(
        None, description="Case-insensitive text to find inside files"
    )
    max_results: int = Field(50, ge=1, le=500)


class SearchFilesTool(Tool):
    name = "search_files"
    description = (
        "Find files by name (substring or glob) and/or by text content inside files. "
        "Returns matching paths with sizes, plus line excerpts for content matches."
    )
    action = "Search files"
    args_model = SearchFilesArgs

    def run(self, args: SearchFilesArgs, ctx: ToolContext) -> ToolResult:
        if not (args.name_contains or args.glob or args.content_query):
            raise ToolError("provide at least one of: name_contains, glob, content_query")
        root = ctx.paths.resolve(args.root)
        if not root.exists():
            raise ToolError(f"directory not found: {root}")

        needle = args.name_contains.casefold() if args.name_contains else None
        content_needle = args.content_query.casefold() if args.content_query else None
        matches: list[dict[str, object]] = []
        total_content_hits = 0
        truncated = False

        candidates: Iterator[Path]
        candidates = (
            _walk_files(root, ctx.paths.denied_dir_names)
            if root.is_dir()
            else iter([root])
        )
        for path in candidates:
            if needle is not None and needle not in path.name.casefold():
                continue
            if args.glob and not fnmatch.fnmatch(path.name, args.glob):
                continue
            item: dict[str, object] = _entry(path)
            if content_needle is not None:
                try:
                    if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                        continue
                    text, _ = _read_text(path, MAX_SEARCH_FILE_BYTES)
                except (OSError, ToolError):
                    continue
                hits: list[str] = []
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if content_needle in line.casefold():
                        hits.append(f"{lineno}: {line.strip()[:240]}")
                        total_content_hits += 1
                        if len(hits) >= 5 or total_content_hits >= MAX_CONTENT_MATCHES:
                            break
                if not hits:
                    continue
                item["matches"] = hits
            matches.append(item)
            if len(matches) >= args.max_results:
                truncated = True
                break

        return ToolResult(
            summary=f"{len(matches)} matching files under {root}"
            + (" (truncated)" if truncated else ""),
            data={"root": str(root), "count": len(matches), "truncated": truncated, "files": matches},
        )


# --- read_file ------------------------------------------------------------------


class ReadFileArgs(BaseModel):
    path: str = Field(description="File to read")
    offset: int = Field(1, ge=1, description="1-based line number to start from")
    limit: int = Field(400, ge=1, le=5000, description="Maximum lines to return")


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file (with line numbers). Supports offset/limit for large files."
    action = "Read a file"
    args_model = ReadFileArgs

    def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.paths.resolve(args.path)
        if not path.exists():
            raise ToolError(f"file not found: {path}")
        if path.is_dir():
            raise ToolError(f"is a directory, not a file: {path} (use list_files)")
        text, truncated = _read_text(path)
        lines = text.splitlines()
        window = lines[args.offset - 1 : args.offset - 1 + args.limit]
        numbered = "\n".join(
            f"{args.offset + i:5d} | {line}" for i, line in enumerate(window)
        )
        return ToolResult(
            summary=(
                f"{path}: lines {args.offset}-{args.offset + len(window) - 1} "
                f"of {len(lines)}"
            ),
            data={
                "path": str(path),
                "total_lines": len(lines),
                "offset": args.offset,
                "returned_lines": len(window),
                "file_truncated": truncated,
                "content": numbered,
            },
            truncated=truncated,
        )


# --- write_file -----------------------------------------------------------------


class WriteFileArgs(BaseModel):
    path: str = Field(description="File to create or overwrite")
    content: str = Field(description="Text content to write")
    overwrite: bool = Field(False, description="Required to replace an existing file")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a text file. Refuses to replace an existing file unless "
        "overwrite=true (which requires confirmation)."
    )
    action = "Create or overwrite a file"
    risk = "medium"
    permission = PermissionLevel.SAFE
    args_model = WriteFileArgs

    def permission_for(self, args: WriteFileArgs) -> PermissionLevel:
        target = Path(getattr(args, "path", "")).expanduser()
        if target.exists():
            return PermissionLevel.CONFIRM
        return PermissionLevel.SAFE

    def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        path = ctx.paths.check_writable_location(args.path)
        if path.exists():
            if path.is_dir():
                raise ToolError(f"is a directory: {path}")
            if not args.overwrite:
                raise ToolError(f"{path} already exists; set overwrite=true to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(args.content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        return ToolResult(
            summary=f"wrote {len(args.content.encode('utf-8'))} bytes to {path}",
            data={"path": str(path), "bytes": len(args.content.encode("utf-8"))},
        )


# --- copy / move ----------------------------------------------------------------


class CopyFileArgs(BaseModel):
    source: str = Field(description="File to copy")
    destination: str = Field(description="Target path (file or directory)")
    overwrite: bool = Field(False, description="Replace an existing destination file")


class CopyFileTool(Tool):
    name = "copy_file"
    description = "Copy a file to a new location inside the allowed workspace."
    action = "Copy a file"
    args_model = CopyFileArgs

    def permission_for(self, args: CopyFileArgs) -> PermissionLevel:
        dest = Path(getattr(args, "destination", "")).expanduser()
        if dest.is_dir():
            dest = dest / Path(getattr(args, "source", "")).name
        return PermissionLevel.CONFIRM if dest.exists() else PermissionLevel.SAFE

    def run(self, args: CopyFileArgs, ctx: ToolContext) -> ToolResult:
        source = ctx.paths.resolve(args.source)
        destination = ctx.paths.check_writable_location(args.destination)
        if not source.exists():
            raise ToolError(f"source not found: {source}")
        if source.is_dir():
            raise ToolError("source is a directory; copy files individually")
        if destination.is_dir():
            destination = destination / source.name
        if destination.exists() and not args.overwrite:
            raise ToolError(f"{destination} exists; set overwrite=true to replace it")
        if destination == source:
            raise ToolError("source and destination are the same file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ToolResult(
            summary=f"copied {source} -> {destination}",
            data={"source": str(source), "destination": str(destination)},
        )


class MoveFileArgs(BaseModel):
    source: str = Field(description="File to move")
    destination: str = Field(description="Target path (file or directory)")
    overwrite: bool = Field(False, description="Replace an existing destination file")


class MoveFileTool(Tool):
    name = "move_file"
    description = "Move or rename a file inside the allowed workspace. Requires confirmation."
    action = "Move a file"
    permission = PermissionLevel.CONFIRM
    risk = "medium"
    args_model = MoveFileArgs

    def run(self, args: MoveFileArgs, ctx: ToolContext) -> ToolResult:
        source = ctx.paths.resolve(args.source)
        destination = ctx.paths.check_writable_location(args.destination)
        if not source.exists():
            raise ToolError(f"source not found: {source}")
        if source.is_dir():
            raise ToolError("source is a directory; move files individually")
        if destination.is_dir():
            destination = destination / source.name
        if destination.exists() and not args.overwrite:
            raise ToolError(f"{destination} exists; set overwrite=true to replace it")
        if destination == source:
            raise ToolError("source and destination are the same file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return ToolResult(
            summary=f"moved {source} -> {destination}",
            data={"source": str(source), "destination": str(destination)},
        )


# --- delete_file ----------------------------------------------------------------


class DeleteFileArgs(BaseModel):
    path: str = Field(description="File (or directory with recursive=true) to delete")
    recursive: bool = Field(False, description="Required to delete a directory")


class DeleteFileTool(Tool):
    name = "delete_file"
    description = (
        "Delete a file, or a directory when recursive=true. Always requires "
        "explicit confirmation; roots and denied paths can never be deleted."
    )
    action = "Delete a file or directory"
    permission = PermissionLevel.CONFIRM
    risk = "high"
    args_model = DeleteFileArgs

    def run(self, args: DeleteFileArgs, ctx: ToolContext) -> ToolResult:
        target = ctx.paths.check_deletable(args.path)
        if not target.exists() and not target.is_symlink():
            raise ToolError(f"not found: {target}")
        if target.is_dir() and not target.is_symlink():
            if not args.recursive:
                raise ToolError(
                    f"{target} is a directory; set recursive=true to delete it "
                    "(confirmation still required)"
                )
            count = sum(1 for _ in _walk_files(target, set()))
            shutil.rmtree(target)
            return ToolResult(
                summary=f"deleted directory {target} ({count} files)",
                data={"path": str(target), "kind": "directory", "files": count},
            )
        target.unlink()
        return ToolResult(
            summary=f"deleted file {target}",
            data={"path": str(target), "kind": "file"},
        )
