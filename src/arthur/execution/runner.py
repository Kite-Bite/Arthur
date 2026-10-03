"""Safe subprocess execution.

Hard rules enforced here:

* ``shell=False`` always - arguments are passed as an argv list, never a
  string interpreted by a shell.
* Shell metacharacters are rejected outright as defence in depth.
* A scrubbed environment is passed to child processes (no secrets inherited).
* Output is capped and the process is killed on timeout.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Characters that would be meaningful to a shell; rejected even though no
#: shell is ever invoked.
SHELL_METACHARACTERS = frozenset(";|&`$><\n\r*?{}[]()~")

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT = 64_000


@dataclass(frozen=True)
class CommandResult:
    """Structured result of one controlled command execution."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def validate_argv(argv: Sequence[str]) -> list[str]:
    """Validate an argv list: non-empty, no shell metacharacters, sane size."""
    cleaned = [str(a) for a in argv]
    if not cleaned or not cleaned[0]:
        raise ValueError("command must be a non-empty argv list")
    if len(cleaned) > 33:
        raise ValueError("too many arguments (max 32)")
    for arg in cleaned:
        bad = SHELL_METACHARACTERS.intersection(arg)
        if bad:
            chars = "".join(sorted(bad))
            raise ValueError(f"argument {arg!r} contains forbidden shell metacharacter(s): {chars}")
    return cleaned


def _scrubbed_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Minimal environment for child processes: no tokens or secrets leak in."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TERM": os.environ.get("TERM", "dumb"),
    }
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def run_argv(
    argv: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output: int = DEFAULT_MAX_OUTPUT,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run a validated argv list without a shell.

    Args:
        argv: Program and its arguments (never a shell string).
        timeout: Seconds before the child is killed.
        max_output: Maximum characters kept from stdout/stderr each.
        cwd: Optional working directory.
        env: Optional extra environment variables (merged over a scrubbed base).

    Returns:
        A :class:`CommandResult`; timeouts are results, not exceptions.

    Raises:
        ValueError: if the argv fails validation.
    """
    cleaned = validate_argv(argv)
    started = time.perf_counter()
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, shell=False, validated
            cleaned,
            shell=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=_scrubbed_env(env),
        )
    except subprocess.TimeoutExpired as exc:
        duration = (time.perf_counter() - started) * 1000
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            argv=tuple(cleaned),
            returncode=-1,
            stdout=stdout[:max_output],
            stderr=(stderr or f"timed out after {timeout}s")[:max_output],
            timed_out=True,
            duration_ms=duration,
        )
    except FileNotFoundError as exc:
        duration = (time.perf_counter() - started) * 1000
        return CommandResult(
            argv=tuple(cleaned),
            returncode=127,
            stdout="",
            stderr=f"command not found: {cleaned[0]} ({exc})",
            timed_out=False,
            duration_ms=duration,
        )
    except PermissionError as exc:
        duration = (time.perf_counter() - started) * 1000
        return CommandResult(
            argv=tuple(cleaned),
            returncode=126,
            stdout="",
            stderr=f"permission denied: {exc}",
            timed_out=False,
            duration_ms=duration,
        )

    duration = (time.perf_counter() - started) * 1000
    stdout = proc.stdout[:max_output]
    stderr = proc.stderr[:max_output]
    if len(proc.stdout) > max_output:
        stdout += "\n... [stdout truncated]"
    if len(proc.stderr) > max_output:
        stderr += "\n... [stderr truncated]"
    return CommandResult(
        argv=tuple(cleaned),
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_ms=duration,
    )
