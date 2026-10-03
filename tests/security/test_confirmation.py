"""Confirmation semantics: pending, approved, declined - all audited."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.security.confirmation import ConfirmationRequest


@pytest.fixture()
def target(tmp_path: Path) -> Path:
    path = tmp_path / "victim.txt"
    path.write_text("contents", encoding="utf-8")
    return path


def test_pending_confirmation_when_no_handler(services, target: Path) -> None:
    outcome = services.executor.execute("delete_file", {"path": str(target)})

    assert outcome.status == "confirmation_required"
    assert outcome.request is not None
    assert outcome.request.tool_name == "delete_file"
    assert outcome.request.target == str(target)
    assert outcome.request.risk == "high"
    assert target.exists()
    assert services.executions.recent(1)[0].status == "confirmation_required"


def test_callback_receives_described_request(services, target: Path) -> None:
    seen: list[ConfirmationRequest] = []

    def handler(request: ConfirmationRequest) -> bool:
        seen.append(request)
        return True

    outcome = services.executor.execute("delete_file", {"path": str(target)}, confirm=handler)

    assert outcome.status == "success"
    assert len(seen) == 1
    assert "Delete a file" in seen[0].action
    assert str(target) in seen[0].describe()
    assert not target.exists()


def test_declined_callback_keeps_file(services, target: Path) -> None:
    outcome = services.executor.execute(
        "delete_file", {"path": str(target)}, confirm=lambda _r: False
    )

    assert outcome.status == "declined"
    assert target.exists()
    row = services.executions.recent(1)[0]
    assert row.status == "declined"
    assert row.confirmed is False


def test_true_flag_approves_and_audits(services, target: Path) -> None:
    outcome = services.executor.execute("delete_file", {"path": str(target)}, confirm=True)

    assert outcome.status == "success"
    assert outcome.record.confirmed is True
    row = services.executions.recent(1)[0]
    assert row.confirmed is True
    assert row.permission == "CONFIRM"


def test_false_flag_declines_without_prompt(services, target: Path) -> None:
    outcome = services.executor.execute("delete_file", {"path": str(target)}, confirm=False)

    assert outcome.status == "declined"
    assert target.exists()


def test_safe_operation_needs_no_confirmation(services) -> None:
    outcome = services.executor.execute("system_info", {})

    assert outcome.status == "success"
    assert outcome.request is None
    assert outcome.record.confirmed is False


def test_overwriting_file_upgrades_to_confirm(services, tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("old", encoding="utf-8")
    args = {"path": str(existing), "content": "new", "overwrite": True}

    pending = services.executor.execute("write_file", args)
    assert pending.status == "confirmation_required"

    approved = services.executor.execute("write_file", args, confirm=True)
    assert approved.status == "success"
    assert existing.read_text(encoding="utf-8") == "new"


def test_confirmation_disabled_by_config_runs_directly(services, target: Path) -> None:
    services.config.security.require_confirmation = False

    outcome = services.executor.execute("delete_file", {"path": str(target)})

    assert outcome.status == "success"
    assert not target.exists()


def test_pending_confirmation_surfaces_through_agent(
    services, scripted, decision, target: Path
) -> None:
    scripted([decision("delete_file", {"path": str(target)})])

    result = services.agent.run(f"Delete {target}")

    assert result.stopped_reason == "confirmation_required"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.reason
    assert target.exists()
