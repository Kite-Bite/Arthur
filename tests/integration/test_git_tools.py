"""Git tools against a real temporary repository (argv-only, no shell)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git binary not available")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [GIT, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(Path.home()),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_AUTHOR_NAME": "Arthur Test",
            "GIT_AUTHOR_EMAIL": "arthur@example.com",
            "GIT_COMMITTER_NAME": "Arthur Test",
            "GIT_COMMITTER_EMAIL": "arthur@example.com",
        },
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A one-commit repository inside the sandboxed allowed root."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "initial import")
    return root


def test_git_status_reports_branch_and_files(services, repo: Path) -> None:
    outcome = services.executor.execute("git_status", {"path": str(repo)})

    assert outcome.status == "success"
    data = outcome.result.data
    assert data["branch"].startswith("main")
    assert data["untracked"] == 0

    (repo / "scratch.txt").write_text("wip", encoding="utf-8")
    outcome = services.executor.execute("git_status", {"path": str(repo)})
    assert outcome.result.data["untracked"] == 1


def test_git_log_lists_commit(services, repo: Path) -> None:
    outcome = services.executor.execute("git_log", {"path": str(repo), "limit": 5})

    assert outcome.status == "success"
    commits = outcome.result.data["commits"]
    assert len(commits) == 1
    assert commits[0]["subject"] == "initial import"
    assert len(commits[0]["hash"]) == 12


def test_git_branches_highlights_current(services, repo: Path) -> None:
    outcome = services.executor.execute("git_branches", {"path": str(repo)})

    assert outcome.status == "success"
    data = outcome.result.data
    assert data["current"] == "main"
    assert data["branches"] == [{"name": "main", "current": True}]


def test_git_info_reports_root_and_branch(services, repo: Path) -> None:
    outcome = services.executor.execute("git_info", {"path": str(repo)})

    assert outcome.status == "success"
    data = outcome.result.data
    assert Path(data["root"]) == repo.resolve()
    assert data["branch"] == "main"
    assert data["remote_url"] is None


def test_git_remote_credentials_are_redacted(services, repo: Path) -> None:
    _git(repo, "remote", "add", "origin", "https://user:hunter2@example.com/repo.git")

    outcome = services.executor.execute("git_info", {"path": str(repo)})

    remote = outcome.result.data["remote_url"]
    assert remote == "https://***@example.com/repo.git"
    assert "hunter2" not in remote


def test_git_diff_clean_then_dirty(services, repo: Path) -> None:
    outcome = services.executor.execute("git_diff", {"path": str(repo)})
    assert outcome.status == "success"
    assert "no changes" in outcome.result.summary

    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    outcome = services.executor.execute("git_diff", {"path": str(repo)})
    assert "app.py" in outcome.result.data["stat"]


def test_git_outside_repository_fails_cleanly(services, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    outcome = services.executor.execute("git_status", {"path": str(plain)})

    assert outcome.status == "error"
    assert "not a git repository" in outcome.record.error


def test_git_path_escape_is_denied(services) -> None:
    outcome = services.executor.execute("git_status", {"path": "/etc"})

    assert outcome.status == "denied"
