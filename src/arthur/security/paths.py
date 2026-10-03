"""Filesystem boundary enforcement.

All file-touching tools resolve paths through :class:`PathPolicy` before any
operation happens. The policy enforces:

* containment within configured roots (default: the user's home directory),
* symlink safety (paths are resolved *before* containment checks),
* deny-listed patterns (SSH keys, env files, cloud credentials, ...),
* deny-listed directory names anywhere in the path,
* extra rules for destructive operations (a root can never be deleted).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from arthur.config.schema import SecurityConfig


class PathViolation(Exception):
    """Raised when a path escapes the configured sandbox or hits a deny rule."""

    def __init__(self, path: Path | str, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{reason}: {path}")


class PathPolicy:
    """Resolves and validates paths against the security configuration."""

    def __init__(self, config: SecurityConfig, *, cwd: Path | None = None) -> None:
        raw_roots = [Path(p).expanduser() for p in config.allowed_roots] or [Path.home()]
        self.roots: list[Path] = [root.resolve() for root in raw_roots]
        self.denied_patterns: list[str] = list(config.denied_patterns)
        self.denied_dir_names: set[str] = set(config.denied_dir_names)
        self._cwd = cwd

    @property
    def cwd(self) -> Path:
        """Base directory for relative paths (defaults to the process cwd)."""
        return (self._cwd or Path.cwd()).resolve()

    def resolve(self, raw: str | Path) -> Path:
        """Resolve ``raw`` to an absolute, symlink-free path inside the sandbox.

        Raises:
            PathViolation: if the path leaves allowed roots or matches a rule.
        """
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:  # pragma: no cover - rare FS errors
            raise PathViolation(candidate, f"cannot resolve path ({exc})") from exc

        if not any(resolved.is_relative_to(root) for root in self.roots):
            roots = ", ".join(str(root) for root in self.roots)
            raise PathViolation(resolved, f"path escapes allowed roots ({roots})")

        for part in resolved.parts:
            if part in self.denied_dir_names:
                raise PathViolation(resolved, f"path contains denied directory {part!r}")

        full = str(resolved)
        for pattern in self.denied_patterns:
            if fnmatch.fnmatch(full, pattern) or fnmatch.fnmatch(resolved.name, pattern):
                raise PathViolation(resolved, f"path matches denied pattern {pattern!r}")

        return resolved

    def check_writable_location(self, raw: str | Path) -> Path:
        """Resolve a path whose parent may not exist yet (creation targets)."""
        return self.resolve(raw)

    def check_deletable(self, raw: str | Path) -> Path:
        """Resolve a deletion target, refusing roots and the filesystem root.

        Raises:
            PathViolation: for sandbox escapes or attempts to delete a root.
        """
        resolved = self.resolve(raw)
        if resolved in self.roots:
            raise PathViolation(resolved, "refusing to delete an allowed root directory")
        if resolved == Path("/"):
            raise PathViolation(resolved, "refusing to delete the filesystem root")
        if resolved.parent == resolved:
            raise PathViolation(resolved, "refusing to delete a filesystem root")
        return resolved

    def describe(self) -> dict[str, object]:
        """Human-readable summary for explainability (``arthur config show``)."""
        return {
            "roots": [str(root) for root in self.roots],
            "denied_patterns": self.denied_patterns,
            "denied_dir_names": sorted(self.denied_dir_names),
        }
