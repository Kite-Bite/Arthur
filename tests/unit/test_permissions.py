"""SecurityPolicy decisions across permission levels and command gating."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from arthur.config.schema import SecurityConfig
from arthur.security.permissions import PermissionLevel
from arthur.security.policy import SecurityPolicy
from arthur.tools.base import Tool, ToolContext, ToolResult


class _Args(BaseModel):
    path: str = ""


class _Stub(Tool):
    name = "stub_tool"
    description = "stub"
    args_model = _Args

    def run(self, args: _Args, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        return ToolResult(summary="ok")


class _ConfirmStub(_Stub):
    name = "confirm_tool"
    permission = PermissionLevel.CONFIRM


class _RestrictedStub(_Stub):
    name = "restricted_tool"
    permission = PermissionLevel.RESTRICTED


class _DeniedStub(_Stub):
    name = "denied_tool"
    permission = PermissionLevel.DENIED


class _RunCommand(_Stub):
    name = "run_command"
    permission = PermissionLevel.CONFIRM


@pytest.fixture()
def policy(tmp_path: Path) -> SecurityPolicy:
    return SecurityPolicy(SecurityConfig(allowed_roots=[str(tmp_path)]))


def _evaluate(policy: SecurityPolicy, tool: Tool, path: str | None = None):
    args = _Args(**({"path": path} if path else {}))
    return policy.evaluate(tool, args, args.model_dump())


def test_safe_tool_allowed(policy: SecurityPolicy) -> None:
    decision = _evaluate(policy, _Stub())
    assert decision.allowed and not decision.requires_confirmation
    assert decision.level == PermissionLevel.SAFE


def test_confirm_tool_requires_confirmation(policy: SecurityPolicy) -> None:
    decision = _evaluate(policy, _ConfirmStub())
    assert decision.allowed and decision.requires_confirmation


def test_require_confirmation_false_disables_prompt(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], require_confirmation=False)
    decision = SecurityPolicy(config).evaluate(_ConfirmStub(), _Args(), _Args().model_dump())
    assert decision.allowed and not decision.requires_confirmation


def test_restricted_denied_by_default(policy: SecurityPolicy) -> None:
    decision = _evaluate(policy, _RestrictedStub())
    assert not decision.allowed
    assert "allow_restricted" in decision.reason


def test_restricted_allowed_when_enabled(tmp_path: Path) -> None:
    config = SecurityConfig(
        allowed_roots=[str(tmp_path)], allow_restricted=True, require_confirmation=True
    )
    decision = SecurityPolicy(config).evaluate(_RestrictedStub(), _Args(), _Args().model_dump())
    assert decision.allowed and decision.requires_confirmation


def test_denied_never_allowed(policy: SecurityPolicy) -> None:
    decision = _evaluate(policy, _DeniedStub())
    assert not decision.allowed and decision.level == PermissionLevel.DENIED


def test_tool_override_from_config(tmp_path: Path) -> None:
    config = SecurityConfig(
        allowed_roots=[str(tmp_path)],
        tool_overrides={"stub_tool": "DENIED"},
    )
    decision = SecurityPolicy(config).evaluate(_Stub(), _Args(), _Args().model_dump())
    assert not decision.allowed and decision.level == PermissionLevel.DENIED


def test_path_argument_outside_roots_denied(policy: SecurityPolicy) -> None:
    decision = _evaluate(policy, _Stub(), path="/etc/shadow")
    assert not decision.allowed
    assert "escapes allowed roots" in decision.reason


def test_path_argument_denied_pattern(policy: SecurityPolicy, tmp_path: Path) -> None:
    decision = _evaluate(policy, _Stub(), path=str(tmp_path / ".env"))
    assert not decision.allowed
    assert "denied pattern" in decision.reason


# --- run_command gating --------------------------------------------------------


def test_shell_disabled_denies_all_commands(policy: SecurityPolicy) -> None:
    args = {"command": "df", "args": []}
    decision = policy.evaluate(_RunCommand(), _Args(), args)
    assert not decision.allowed
    assert "default_shell_access" in decision.reason


def test_shell_enabled_allows_allowlisted_command(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], default_shell_access=True)
    policy = SecurityPolicy(config)
    decision = policy.evaluate(_RunCommand(), _Args(), {"command": "df", "args": ["-h"]})
    assert decision.allowed
    assert decision.level == PermissionLevel.SAFE  # df is on the safe list


def test_shell_enabled_denies_unknown_command(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], default_shell_access=True)
    policy = SecurityPolicy(config)
    decision = policy.evaluate(_RunCommand(), _Args(), {"command": "curl", "args": []})
    assert not decision.allowed
    assert "allow-list" in decision.reason


def test_systemctl_subcommand_restricted(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], default_shell_access=True)
    policy = SecurityPolicy(config)
    bad = policy.evaluate(
        _RunCommand(), _Args(), {"command": "systemctl", "args": ["restart", "sshd"]}
    )
    assert not bad.allowed
    good = policy.evaluate(
        _RunCommand(), _Args(), {"command": "systemctl", "args": ["status", "sshd"]}
    )
    assert good.allowed


def test_ip_destructive_verbs_denied(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], default_shell_access=True)
    policy = SecurityPolicy(config)
    decision = policy.evaluate(
        _RunCommand(), _Args(), {"command": "ip", "args": ["link", "set", "down"]}
    )
    assert not decision.allowed


def test_unsafe_allowlisted_command_needs_confirmation(tmp_path: Path) -> None:
    config = SecurityConfig(
        allowed_roots=[str(tmp_path)],
        default_shell_access=True,
        safe_commands=[],
        require_confirmation=True,
    )
    policy = SecurityPolicy(config)
    decision = policy.evaluate(_RunCommand(), _Args(), {"command": "df", "args": []})
    assert decision.allowed and decision.requires_confirmation
