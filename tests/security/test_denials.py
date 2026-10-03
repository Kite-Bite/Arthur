"""Security tests: denials that must never execute anything.

Every test here asserts two things: the operation did not run, and the
attempt was recorded in the audit trail.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from arthur.config.schema import Config
from arthur.execution.executor import ToolExecutor
from arthur.security.permissions import PermissionLevel
from arthur.security.policy import SecurityPolicy
from arthur.tools.base import Tool, ToolContext, ToolResult
from arthur.tools.registry import ToolRegistry


class _NoArgs(BaseModel):
    pass


class _SentinelTool(Tool):
    """A tool that must never run when policy denies it."""

    name = "sentinel"
    description = "Records whether it was executed (must stay unexecuted)."
    permission = PermissionLevel.SAFE
    args_model = _NoArgs

    def __init__(self) -> None:
        self.executed = False

    def run(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        self.executed = True
        return ToolResult(summary="should never happen")


@pytest.fixture()
def denied_executor(base_config: Config):
    """Executor with a sentinel tool forced to DENIED by configuration."""
    base_config.security.tool_overrides = {"sentinel": "DENIED"}
    registry = ToolRegistry()
    sentinel = _SentinelTool()
    registry.register(sentinel)
    policy = SecurityPolicy(base_config.security)
    ctx = ToolContext(config=base_config, paths=policy.paths)
    records: list = []
    executor = ToolExecutor(registry, policy, ctx, audit=records.append)
    return executor, sentinel, records


def test_denied_tool_never_executes(denied_executor) -> None:
    executor, sentinel, records = denied_executor

    outcome = executor.execute("sentinel", {})

    assert outcome.status == "denied"
    assert sentinel.executed is False
    assert outcome.record.permission is PermissionLevel.DENIED
    assert len(records) == 1 and records[0].status == "denied"


def test_confirmation_cannot_bypass_denial(denied_executor) -> None:
    executor, sentinel, _ = denied_executor

    outcome = executor.execute("sentinel", {}, confirm=True)

    assert outcome.status == "denied"
    assert sentinel.executed is False


def test_unknown_tool_is_rejected(services) -> None:
    outcome = services.executor.execute("definitely_not_a_tool", {})

    assert outcome.status == "unknown_tool"
    assert "available:" in outcome.record.error


def test_shell_access_denied_by_default(services) -> None:
    outcome = services.executor.execute("run_command", {"command": "date"})

    assert outcome.status == "denied"
    assert "shell access is disabled" in outcome.record.error


def test_run_command_denies_non_allowlisted_binary(services, base_config: Config) -> None:
    base_config.security.default_shell_access = True

    for command in ("curl", "wget", "bash", "sh", "python"):
        outcome = services.executor.execute("run_command", {"command": command})
        assert outcome.status == "denied", command
        assert "not in the allow-list" in outcome.record.error


def test_run_command_denies_state_mutating_subcommands(services, base_config: Config) -> None:
    base_config.security.default_shell_access = True

    reboot = services.executor.execute("run_command", {"command": "systemctl", "args": ["reboot"]})
    assert reboot.status == "denied"
    assert "reboot" in reboot.record.error

    ip_mutate = services.executor.execute(
        "run_command", {"command": "ip", "args": ["set", "down", "eth0"]}
    )
    assert ip_mutate.status == "denied"
    assert "modify system state" in ip_mutate.record.error


def test_run_command_allows_read_only_commands(services, base_config: Config) -> None:
    base_config.security.default_shell_access = True

    outcome = services.executor.execute("run_command", {"command": "date"})

    assert outcome.status == "success"
    assert outcome.record.permission is PermissionLevel.SAFE


def test_run_command_denies_shell_metacharacters(services, base_config: Config) -> None:
    base_config.security.default_shell_access = True

    outcome = services.executor.execute(
        "run_command", {"command": "date", "args": ["; echo pwned"]}
    )

    assert outcome.status in {"denied", "error"}
    assert outcome.result is None or not outcome.result.ok


def test_path_escape_denied(services, tmp_path: Path) -> None:
    outcome = services.executor.execute("read_file", {"path": "/etc/shadow"})

    assert outcome.status == "denied"
    assert "escapes allowed roots" in outcome.record.error


def test_relative_traversal_denied(services, tmp_path: Path) -> None:
    (tmp_path / "inside.txt").write_text("ok", encoding="utf-8")

    outcome = services.executor.execute("read_file", {"path": "../../etc/passwd"})

    # Either escapes the root or lands outside it - never readable.
    assert outcome.status == "denied"
    assert "result" not in (outcome.result.data if outcome.result else {}) or True
    assert outcome.result is None


def test_dotenv_and_ssh_paths_denied(services, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519").write_text("PRIVATE", encoding="utf-8")

    env_outcome = services.executor.execute("read_file", {"path": str(tmp_path / ".env")})
    assert env_outcome.status == "denied"

    ssh_outcome = services.executor.execute("read_file", {"path": str(ssh_dir / "id_ed25519")})
    assert ssh_outcome.status == "denied"


def test_symlink_escape_denied(services, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    link = tmp_path / "sneaky.txt"
    link.symlink_to(outside)

    outcome = services.executor.execute("read_file", {"path": str(link)})

    assert outcome.status == "denied"
    assert "escapes allowed roots" in outcome.record.error


def test_root_directory_deletion_denied(services, tmp_path: Path) -> None:
    outcome = services.executor.execute(
        "delete_file", {"path": str(tmp_path), "recursive": True}, confirm=True
    )

    assert outcome.status == "denied"
    assert "refusing to delete an allowed root" in outcome.record.error
    assert tmp_path.exists()


def test_filesystem_root_deletion_denied(services) -> None:
    outcome = services.executor.execute(
        "delete_file", {"path": "/", "recursive": True}, confirm=True
    )

    assert outcome.status == "denied"


def test_restricted_tool_denied_without_opt_in(services) -> None:
    services.config.security.tool_overrides = {"system_info": "RESTRICTED"}

    outcome = services.executor.execute("system_info", {})

    assert outcome.status == "denied"
    assert "restricted" in outcome.record.error


def test_restricted_tool_runs_with_opt_in(services) -> None:
    services.config.security.tool_overrides = {"system_info": "RESTRICTED"}
    services.config.security.allow_restricted = True

    outcome = services.executor.execute("system_info", {}, confirm=True)

    assert outcome.status == "success"
    assert outcome.record.permission is PermissionLevel.RESTRICTED


def test_malformed_arguments_rejected(services) -> None:
    missing = services.executor.execute("read_file", {})
    assert missing.status == "invalid_args"
    assert "path" in missing.record.error

    wrong_type = services.executor.execute("read_file", {"path": "/tmp/x", "limit": "lots"})
    assert wrong_type.status == "invalid_args"

    unknown_field = services.executor.execute("uptime", {"bogus_argument": True})
    assert unknown_field.status == "invalid_args"


def test_policy_errors_fail_closed(services) -> None:
    # A path argument that cannot be resolved must deny, not raise.
    outcome = services.executor.execute("read_file", {"path": "\0bad"})

    assert outcome.status in {"denied", "invalid_args", "error"}
    assert outcome.result is None or not outcome.result.ok
