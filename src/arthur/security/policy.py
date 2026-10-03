"""The security policy: turns (tool, arguments) into a permission decision.

Central evaluation order:

1. Explicit per-tool overrides from configuration (highest precedence).
2. The tool's own dynamic level (e.g. overwriting an existing file upgrades
   a write from SAFE to CONFIRM).
3. ``run_command`` is gated entirely by ``security.default_shell_access`` and
   the command allow-list - arbitrary shell is always denied.
4. Filesystem arguments are validated against :class:`PathPolicy` before the
   tool ever runs (tools validate again as defence in depth).
5. RESTRICTED requires opt-in; CONFIRM requires a human; DENIED never runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arthur.config.schema import DENIED_COMMAND_TOKENS, SecurityConfig
from arthur.security.paths import PathPolicy, PathViolation
from arthur.security.permissions import PermissionDecision, PermissionLevel

if TYPE_CHECKING:
    from pydantic import BaseModel

    from arthur.tools.base import Tool

#: Argument keys that are treated as filesystem paths for central validation.
PATH_ARG_KEYS = frozenset(
    {"path", "file_path", "source", "target", "destination", "root", "directory", "dir"}
)


class SecurityPolicy:
    """Configurable, testable security policy for all tool executions."""

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        self.paths = PathPolicy(config)
        self._tool_overrides: dict[str, PermissionLevel] = {
            name: PermissionLevel.parse(level) for name, level in config.tool_overrides.items()
        }

    def evaluate(
        self,
        tool: Tool,
        args: BaseModel,
        args_dict: dict[str, Any],
    ) -> PermissionDecision:
        """Decide whether ``tool`` may run with ``args``.

        Returns a :class:`PermissionDecision` - it never raises for policy
        violations (path violations are converted into denials with reasons).
        """
        name = tool.name

        if name == "run_command":
            return self._evaluate_command(tool, args, args_dict)

        level = self._tool_overrides.get(name, tool.permission_for(args))

        violation = self._path_violation(args_dict)
        if violation is not None:
            return PermissionDecision.deny(level, violation)

        return self._decision_for_level(name, level)

    def _decision_for_level(self, name: str, level: PermissionLevel) -> PermissionDecision:
        if level >= PermissionLevel.DENIED:
            return PermissionDecision.deny(level, f"tool {name!r} is denied by policy")
        if level >= PermissionLevel.RESTRICTED:
            if not self.config.allow_restricted:
                return PermissionDecision.deny(
                    level,
                    f"tool {name!r} is restricted; set security.allow_restricted=true to enable",
                )
            confirm = self.config.require_confirmation
            return PermissionDecision.allow(
                level, confirm=confirm, reason="restricted tool enabled by configuration"
            )
        if level >= PermissionLevel.CONFIRM:
            if self.config.require_confirmation:
                return PermissionDecision.allow(
                    level, confirm=True, reason="confirmation required for this operation"
                )
            return PermissionDecision.allow(
                level, confirm=False, reason="confirmation disabled by configuration"
            )
        return PermissionDecision.allow(level, confirm=False, reason="safe operation")

    def _evaluate_command(
        self,
        tool: Tool,
        args: BaseModel,
        args_dict: dict[str, Any],
    ) -> PermissionDecision:
        base_level = self._tool_overrides.get(
            "run_command", tool.permission_for(args)
        )

        if not self.config.default_shell_access:
            return PermissionDecision.deny(
                base_level,
                "shell access is disabled (security.default_shell_access=false); "
                "arbitrary commands never run",
            )

        command = str(args_dict.get("command", "") or "")
        if command not in self.config.allowed_commands:
            return PermissionDecision.deny(
                base_level,
                f"command {command!r} is not in the allow-list; "
                "arbitrary shell execution is denied",
            )

        violation = self._path_violation(args_dict)
        if violation is not None:
            return PermissionDecision.deny(base_level, violation)

        subcommands = self.config.allowed_commands[command]
        argv = [str(a) for a in args_dict.get("args") or []]
        positionals = [a for a in argv if not a.startswith("-")]
        if positionals and set(positionals) & DENIED_COMMAND_TOKENS:
            banned = ", ".join(sorted(set(positionals) & DENIED_COMMAND_TOKENS))
            return PermissionDecision.deny(
                base_level, f"argument(s) {banned} would modify system state"
            )
        if subcommands is not None and positionals and positionals[0] not in subcommands:
            return PermissionDecision.deny(
                base_level,
                f"subcommand {positionals[0]!r} is not allowed for {command!r} "
                f"(allowed: {', '.join(subcommands)})",
            )

        if command in self.config.safe_commands:
            return PermissionDecision.allow(
                PermissionLevel.SAFE, confirm=False, reason=f"{command!r} is a read-only command"
            )
        level = PermissionLevel.SAFE if not self.config.require_confirmation else PermissionLevel.CONFIRM
        return self._decision_for_level(f"run_command:{command}", level)

    def _path_violation(self, args_dict: dict[str, Any]) -> str | None:
        for key in PATH_ARG_KEYS:
            value = args_dict.get(key)
            if isinstance(value, str) and value:
                try:
                    self.paths.resolve(value)
                except PathViolation as exc:
                    return f"{key}: {exc.reason}"
        return None
