"""Runner safety: no shell, metacharacter rejection, timeouts, scrubbed env."""

from __future__ import annotations

import time

import pytest

from arthur.execution.runner import SHELL_METACHARACTERS, run_argv, validate_argv


def test_runs_argv_without_shell() -> None:
    result = run_argv(["echo", "hello world"])
    assert result.ok
    assert result.stdout.strip() == "hello world"
    assert result.duration_ms >= 0


def test_echo_does_not_interpret_quotes() -> None:
    # No shell means quote characters arrive literally (no quote stripping).
    result = run_argv(["echo", 'it\'s a "quoted" value'])
    assert result.stdout.strip() == 'it\'s a "quoted" value'


@pytest.mark.parametrize(
    "bad",
    ["a;b", "a|b", "a&b", "$(id)", "`id`", "a>b", "a<b", "a\nb", "a*b"],
)
def test_metacharacters_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="metacharacter"):
        validate_argv(["echo", bad])
    assert any(ch in SHELL_METACHARACTERS for ch in bad)


def test_empty_argv_rejected() -> None:
    with pytest.raises(ValueError):
        validate_argv([])
    with pytest.raises(ValueError):
        validate_argv([""])


def test_too_many_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="too many"):
        validate_argv(["cmd"] + ["a"] * 40)


def test_missing_binary_reported_not_raised() -> None:
    result = run_argv(["definitely-not-a-real-binary-xyz"])
    assert result.returncode == 127
    assert not result.ok
    assert "not found" in result.stderr


def test_timeout_is_a_result() -> None:
    started = time.perf_counter()
    result = run_argv(["sleep", "5"], timeout=0.3)
    elapsed = time.perf_counter() - started
    assert result.timed_out
    assert not result.ok
    assert elapsed < 3.0


def test_output_truncated() -> None:
    result = run_argv(["echo", "x" * 500], max_output=50)
    assert "[stdout truncated]" in result.stderr or "[stdout truncated]" in result.stdout
    assert len(result.stdout) < 200


def test_nonzero_exit_captured() -> None:
    result = run_argv(["ls", "/definitely/not/here"])
    assert result.returncode != 0
    assert not result.ok


def test_environment_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTHUR_TEST_SECRET", "s3cret-value")
    result = run_argv(["printenv", "ARTHUR_TEST_SECRET"])
    assert "s3cret-value" not in result.stdout
    assert result.returncode != 0  # variable was not passed through
