"""PathPolicy: sandbox containment, deny rules, deletion guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.config.schema import SecurityConfig
from arthur.security.paths import PathPolicy, PathViolation


@pytest.fixture()
def policy(tmp_path: Path) -> PathPolicy:
    config = SecurityConfig(allowed_roots=[str(tmp_path)])
    return PathPolicy(config, cwd=tmp_path)


def test_inside_root_allowed(policy: PathPolicy, tmp_path: Path) -> None:
    target = tmp_path / "projects" / "a.txt"
    assert policy.resolve(target) == target.resolve()


def test_relative_path_resolves_against_cwd(policy: PathPolicy, tmp_path: Path) -> None:
    assert policy.resolve("sub/file.txt") == (tmp_path / "sub" / "file.txt").resolve()


def test_escape_denied(policy: PathPolicy) -> None:
    with pytest.raises(PathViolation, match="escapes allowed roots"):
        policy.resolve("/etc/passwd")
    with pytest.raises(PathViolation, match="escapes allowed roots"):
        policy.resolve("../../outside.txt")


def test_dotdot_traversal_normalised_then_checked(policy: PathPolicy, tmp_path: Path) -> None:
    # tmp/../.. escapes even though the string starts inside the root
    with pytest.raises(PathViolation):
        policy.resolve(f"{tmp_path}/../../etc/hostname")


def test_denied_dir_name_anywhere(policy: PathPolicy, tmp_path: Path) -> None:
    secret = tmp_path / ".ssh" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("key", encoding="utf-8")
    with pytest.raises(PathViolation, match="denied directory"):
        policy.resolve(secret)
    # The directory itself is also protected.
    with pytest.raises(PathViolation, match="denied directory"):
        policy.resolve(tmp_path / ".ssh")


def test_denied_pattern_env_file(policy: PathPolicy, tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("TOKEN=abc", encoding="utf-8")
    with pytest.raises(PathViolation, match="denied pattern"):
        policy.resolve(env)


def test_symlink_escape_denied(policy: PathPolicy, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("data", encoding="utf-8")
    link = tmp_path / "sneaky.txt"
    link.symlink_to(outside)
    with pytest.raises(PathViolation, match="escapes allowed roots"):
        policy.resolve(link)


def test_delete_root_refused(policy: PathPolicy, tmp_path: Path) -> None:
    with pytest.raises(PathViolation, match="allowed root"):
        policy.check_deletable(tmp_path)
    with pytest.raises(PathViolation):
        policy.check_deletable("/")


def test_delete_inside_allowed(policy: PathPolicy, tmp_path: Path) -> None:
    target = tmp_path / "junk.txt"
    target.write_text("x", encoding="utf-8")
    assert policy.check_deletable(target) == target.resolve()


def test_denied_dir_names_configurable(tmp_path: Path) -> None:
    config = SecurityConfig(allowed_roots=[str(tmp_path)], denied_dir_names=["private"])
    policy = PathPolicy(config, cwd=tmp_path)
    with pytest.raises(PathViolation, match="denied directory"):
        policy.resolve(tmp_path / "private" / "x.txt")


def test_describe_is_json_ish(policy: PathPolicy) -> None:
    described = policy.describe()
    assert described["roots"] and isinstance(described["denied_patterns"], list)
