"""Controlled Linux command execution.

There is no "run arbitrary shell" tool. :class:`RunCommandTool` only executes
allow-listed commands as argv lists (no shell), and the security policy gates
the whole tool behind ``security.default_shell_access``. The checks are
repeated inside the tool itself as defence in depth.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.execution.runner import run_argv
from arthur.security.permissions import PermissionLevel
from arthur.tools.base import Tool, ToolContext, ToolResult

COMMAND_NAME_RE = re.compile(r"[A-Za-z0-9._+-]{1,64}\Z")


class RunCommandArgs(BaseModel):
    command: str = Field(description="Allow-listed command name, e.g. 'df' or 'systemctl'")
    args: list[str] = Field(default_factory=list, max_length=32, description="Arguments (no shell)")
    timeout: float = Field(30.0, gt=0, le=120, description="Seconds before the command is killed")


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run one allow-listed Linux command without a shell (argv only). "
        "Disabled unless security.default_shell_access=true; commands outside "
        "the allow-list are always denied."
    )
    action = "Run a Linux command"
    permission = PermissionLevel.CONFIRM
    risk = "medium"
    args_model = RunCommandArgs
    timeout = 140.0

    def run(self, args: RunCommandArgs, ctx: ToolContext) -> ToolResult:
        security = ctx.config.security

        if not security.default_shell_access:
            raise ToolError(
                "shell access is disabled (security.default_shell_access=false)"
            )
        if not COMMAND_NAME_RE.match(args.command):
            raise ToolError(f"invalid command name: {args.command!r}")
        if args.command not in security.allowed_commands:
            raise ToolError(
                f"{args.command!r} is not in the command allow-list; "
                "arbitrary command execution is denied"
            )

        subcommands = security.allowed_commands[args.command]
        positionals = [a for a in args.args if not a.startswith("-")]
        if subcommands is not None and positionals and positionals[0] not in subcommands:
            raise ToolError(
                f"subcommand {positionals[0]!r} is not allowed for {args.command!r} "
                f"(allowed: {', '.join(subcommands)})"
            )

        result = run_argv(
            [args.command, *args.args],
            timeout=min(args.timeout, 120.0),
            max_output=32_000,
        )
        summary = (
            f"$ {args.command} {' '.join(args.args)} -> exit {result.returncode}"
            + (" (timed out)" if result.timed_out else "")
        )
        return ToolResult(
            ok=not result.timed_out,
            summary=summary,
            data={
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "duration_ms": round(result.duration_ms, 1),
            },
        )
