"""CLI integration via Typer's CliRunner (no real LLM required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from arthur import __version__
from arthur.cli.main import app

runner = CliRunner()


@pytest.fixture()
def sandbox_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Isolate every CLI invocation: own DB, own log file, own allowed roots."""
    env = {
        "ARTHUR_MEMORY_DATABASE": str(tmp_path / "cli.db"),
        "ARTHUR_LOGGING_FILE": str(tmp_path / "arthur.log"),
        "ARTHUR_SECURITY_ALLOWED_ROOTS": str(tmp_path),
        # Dependency-free retrieval so no Ollama is required.
        "ARTHUR_RETRIEVAL_EMBEDDER": "hash",
        # Never pick up a developer's real config file.
        "ARTHUR_CONFIG": str(tmp_path / "absent.toml"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def test_version(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"arthur {__version__}" in result.output


def test_tools_list_shows_permissions(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["tools", "list"])

    assert result.exit_code == 0
    assert "read_file" in result.output
    assert "run_command" in result.output
    assert "SAFE" in result.output


def test_tools_show_unknown_exits_nonzero(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["tools", "show", "does_not_exist"])

    assert result.exit_code == 1
    assert "unknown tool" in result.output


def test_tools_show_known_prints_schema(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["tools", "show", "read_file"])

    assert result.exit_code == 0
    assert "CONFIRM" in result.output or "SAFE" in result.output
    assert "properties" in result.output


def test_memory_add_search_delete(sandbox_env: dict[str, str]) -> None:
    added = runner.invoke(
        app, ["memory", "add", "The metric backend is Prometheus", "--category", "infra"]
    )
    assert added.exit_code == 0
    assert "stored" in added.output

    found = runner.invoke(app, ["memory", "search", "metric backend"])
    assert found.exit_code == 0
    assert "Prometheus" in found.output

    listed = runner.invoke(app, ["memory", "list"])
    assert listed.exit_code == 0
    assert "1 of 1 memories" in listed.output

    deleted = runner.invoke(app, ["memory", "delete", "1"])
    assert deleted.exit_code == 0
    assert "deleted" in deleted.output


def test_memory_delete_missing_id_fails(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["memory", "delete", "99"])

    assert result.exit_code == 1
    assert "no such memory" in result.output


def test_memory_delete_requires_id_or_all(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["memory", "delete"])

    assert result.exit_code == 2
    assert "provide a memory id or --all" in result.output


def test_system_uptime_runs_without_shell(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["system", "uptime"])

    assert result.exit_code == 0
    assert "uptime_seconds" in result.output


def test_system_info_reports_kernel(sandbox_env: dict[str, str]) -> None:
    result = runner.invoke(app, ["system", "info"])

    assert result.exit_code == 0
    assert "kernel" in result.output
    assert "architecture" in result.output


def test_config_show_redacts_token(sandbox_env: dict[str, str], monkeypatch) -> None:
    monkeypatch.setenv("ARTHUR_API_TOKEN", "super-secret")
    monkeypatch.setenv("ARTHUR_AGENT_MAX_STEPS", "4")

    result = runner.invoke(app, ["config-show"])

    assert result.exit_code == 0
    assert "***redacted***" in result.output
    assert "super-secret" not in result.output
    assert '"max_steps": 4' in result.output


def test_logs_empty_then_populated(sandbox_env: dict[str, str]) -> None:
    empty = runner.invoke(app, ["logs"])
    assert empty.exit_code == 0

    runner.invoke(app, ["system", "uptime"])
    populated = runner.invoke(app, ["logs"])
    assert populated.exit_code == 0
    assert "uptime" in populated.output


def test_documents_index_and_search_cli(sandbox_env: dict[str, str], tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Guide\n\nCache invalidation happens on publish via the purge endpoint.\n",
        encoding="utf-8",
    )

    indexed = runner.invoke(app, ["documents", "index", str(docs)])
    assert indexed.exit_code == 0
    assert "files" in indexed.output

    # Hash embedder is configured via env so no Ollama is needed.
    monkeyless = runner.invoke(app, ["documents", "search", "cache invalidation", "-n", "3"])
    assert monkeyless.exit_code == 0
    assert "guide.md" in monkeyless.output


def test_ask_requires_model_but_fails_cleanly(sandbox_env: dict[str, str], monkeypatch) -> None:
    # Point at an unreachable Ollama so the failure path is deterministic.
    monkeypatch.setenv("ARTHUR_LLM_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("ARTHUR_LLM_TIMEOUT_SECONDS", "2")

    result = runner.invoke(app, ["ask", "hello", "--quiet"])

    assert result.exit_code == 1
    assert "Error" in result.output or "error" in result.output.lower()


def test_doctor_reports_llm_state(sandbox_env: dict[str, str], monkeypatch) -> None:
    monkeypatch.setenv("ARTHUR_LLM_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("ARTHUR_LLM_TIMEOUT_SECONDS", "2")

    result = runner.invoke(app, ["doctor"])

    # Exit code 1 when Ollama is unreachable, 0 when it happens to be up.
    assert result.exit_code in {0, 1}
    assert "ollama" in result.output
    assert "tools" in result.output
    # The sandbox uses the hash embedder, so no embedding model is required.
    assert "embed" in result.output
    assert "offline" in result.output
