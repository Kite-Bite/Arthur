"""Read-only git tools: status, log, branches, diff summary, repo metadata.

All git invocations use the safe argv runner (no shell) with a ``-C <path>``
so repositories are inspected in place. Remote URLs are credential-redacted.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.execution.runner import CommandResult, run_argv
from arthur.tools.base import Tool, ToolContext, ToolResult

GIT_TIMEOUT = 20.0
_CREDENTIAL_RE = re.compile(r"(?P<scheme>https?://)[^/@\s]+@")


def redact_url(url: str) -> str:
    """Strip userinfo credentials from a remote URL before display/logging."""
    return _CREDENTIAL_RE.sub(r"\g<scheme>***@", url)


def _git(ctx: ToolContext, path: str, git_args: list[str]) -> tuple[Any, CommandResult]:
    root = ctx.paths.resolve(path)
    result = run_argv(["git", "-C", str(root), *git_args], timeout=GIT_TIMEOUT)
    combined = (result.stdout + result.stderr).lower()
    if "not a git repository" in combined:
        raise ToolError(f"not a git repository: {root}")
    if "fatal:" in result.stderr.lower() and not result.stdout.strip():
        raise ToolError(f"git error: {result.stderr.strip()[:300]}")
    return root, result


class GitStatusArgs(BaseModel):
    path: str = Field(".", description="Path inside the repository")


class GitStatusTool(Tool):
    name = "git_status"
    description = "Git repository status: branch, staged/changed/untracked file counts."
    action = "Show git status"
    args_model = GitStatusArgs

    def run(self, args: GitStatusArgs, ctx: ToolContext) -> ToolResult:
        root, result = _git(ctx, args.path, ["status", "--porcelain=v1", "--branch"])
        lines = result.stdout.splitlines()
        branch = ""
        if lines and lines[0].startswith("##"):
            branch = lines[0][2:].strip()
        staged = changed = untracked = 0
        files: list[dict[str, str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            x, y, name = line[:2], line[2:3], line[3:].strip()
            if x == "?" or y == "?":
                untracked += 1
                state = "untracked"
            elif x != " ":
                staged += 1
                state = "staged"
            else:
                changed += 1
                state = "modified"
            files.append({"state": state, "path": name})
        data: dict[str, Any] = {
            "repository": str(root),
            "branch": branch,
            "staged": staged,
            "changed": changed,
            "untracked": untracked,
            "files": files[:100],
        }
        return ToolResult(
            summary=(
                f"{branch or 'unknown branch'}: {staged} staged, "
                f"{changed} changed, {untracked} untracked"
            ),
            data=data,
        )


class GitLogArgs(BaseModel):
    path: str = Field(".", description="Path inside the repository")
    limit: int = Field(10, ge=1, le=50)


class GitLogTool(Tool):
    name = "git_log"
    description = "Recent commits: hash, author, date, subject."
    action = "Show git log"
    args_model = GitLogArgs

    def run(self, args: GitLogArgs, ctx: ToolContext) -> ToolResult:
        _, result = _git(
            ctx,
            args.path,
            [
                "log",
                f"-n{args.limit}",
                "--date=short",
                "--format=%H%x09%an%x09%ad%x09%s",
            ],
        )
        commits = []
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                commits.append(
                    {
                        "hash": parts[0][:12],
                        "author": parts[1],
                        "date": parts[2],
                        "subject": parts[3],
                    }
                )
        return ToolResult(
            summary=f"{len(commits)} commits", data={"commits": commits}
        )


class GitBranchesArgs(BaseModel):
    path: str = Field(".", description="Path inside the repository")


class GitBranchesTool(Tool):
    name = "git_branches"
    description = "List local branches and highlight the current one."
    action = "List git branches"
    args_model = GitBranchesArgs

    def run(self, args: GitBranchesArgs, ctx: ToolContext) -> ToolResult:
        _, result = _git(ctx, args.path, ["branch", "--list"])
        branches: list[dict[str, Any]] = []
        current = ""
        for line in result.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            is_current = name.startswith("* ")
            name = name[2:].strip() if is_current else name
            if is_current:
                current = name
            branches.append({"name": name, "current": is_current})
        return ToolResult(
            summary=f"{len(branches)} branches (current: {current or 'detached'})",
            data={"current": current, "branches": branches},
        )


class GitDiffArgs(BaseModel):
    path: str = Field(".", description="Path inside the repository")
    staged: bool = Field(False, description="Diff the index instead of the working tree")


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Diff summary (--stat): which files changed and how much."
    action = "Show git diff summary"
    args_model = GitDiffArgs

    def run(self, args: GitDiffArgs, ctx: ToolContext) -> ToolResult:
        git_args = ["diff", "--stat"]
        if args.staged:
            git_args.insert(1, "--cached")
        _, result = _git(ctx, args.path, git_args)
        stat = result.stdout.strip()
        return ToolResult(
            summary=stat.splitlines()[-1] if stat else "no changes",
            data={"staged": args.staged, "stat": stat or "(clean)"},
        )


class GitInfoArgs(BaseModel):
    path: str = Field(".", description="Path inside the repository")


class GitInfoTool(Tool):
    name = "git_info"
    description = "Repository metadata: root, current branch, remote URL (credentials redacted)."
    action = "Show git repository metadata"
    args_model = GitInfoArgs

    def run(self, args: GitInfoArgs, ctx: ToolContext) -> ToolResult:
        root, _ = _git(ctx, args.path, ["rev-parse", "--show-toplevel"])
        branch_result = run_argv(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=GIT_TIMEOUT,
        )
        remote: str | None = None
        remote_result = run_argv(
            ["git", "-C", str(root), "remote", "get-url", "origin"], timeout=GIT_TIMEOUT
        )
        if remote_result.ok and remote_result.stdout.strip():
            remote = redact_url(remote_result.stdout.strip())
        data = {
            "root": str(root),
            "branch": branch_result.stdout.strip() or None,
            "remote_url": remote,
        }
        return ToolResult(
            summary=f"repository at {root} (branch: {data['branch'] or 'detached'})",
            data=data,
        )
