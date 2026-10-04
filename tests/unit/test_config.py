"""Configuration loading: precedence, coercion and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.config.loader import load_config, parse_env_file
from arthur.config.schema import Config


def test_defaults_when_no_sources(tmp_path: Path) -> None:
    cfg = load_config(path=tmp_path / "missing.toml", env={})
    assert isinstance(cfg, Config)
    assert cfg.llm.host == "http://localhost:11434"
    assert cfg.security.default_shell_access is False
    assert cfg.security.require_confirmation is True
    assert cfg.api.host == "127.0.0.1"


def test_config_file_applies(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[llm]\nmodel = "qwen3:4b"\ntemperature = 0.7\n[security]\ndefault_shell_access = true\n',
        encoding="utf-8",
    )
    cfg = load_config(path=config_file, env={})
    assert cfg.llm.model == "qwen3:4b"
    assert cfg.llm.temperature == 0.7
    assert cfg.security.default_shell_access is True


def test_environment_beats_config_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('[llm]\nmodel = "from-file"\n', encoding="utf-8")
    cfg = load_config(path=config_file, env={"ARTHUR_LLM_MODEL": "from-env", "MODEL_NAME": "alias"})
    # Ollama alias and ARTHUR_ var both map to llm.model; ARTHUR_ wins last.
    assert cfg.llm.model == "from-env"


def test_ollama_host_alias(tmp_path: Path) -> None:
    cfg = load_config(path=tmp_path / "none.toml", env={"OLLAMA_HOST": "http://gpu:11434"})
    assert cfg.llm.host == "http://gpu:11434"


def test_type_coercion_from_env(tmp_path: Path) -> None:
    cfg = load_config(
        path=tmp_path / "none.toml",
        env={
            "ARTHUR_AGENT_MAX_STEPS": "9",
            "ARTHUR_SECURITY_REQUIRE_CONFIRMATION": "false",
            "ARTHUR_RETRIEVAL_FORMATS": "md,txt",
            "ARTHUR_LLM_TEMPERATURE": "0.9",
        },
    )
    assert cfg.agent.max_steps == 9
    assert cfg.security.require_confirmation is False
    assert cfg.retrieval.formats == ["md", "txt"]
    assert cfg.llm.temperature == pytest.approx(0.9)


def test_invalid_toml_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "broken.toml"
    config_file.write_text("not [ valid toml", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read config file"):
        load_config(path=config_file, env={})


def test_unknown_tool_override_level_fails_at_load(tmp_path: Path) -> None:
    """A typo'd level must not survive until the tool is first called."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[security]\ntool_overrides = { delete_file = "RESTRICTEDD" }\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown permission level"):
        load_config(path=config_file, env={})


def test_valid_tool_override_is_accepted(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[security]\ntool_overrides = { read_file = "CONFIRM" }\n',
        encoding="utf-8",
    )

    cfg = load_config(path=config_file, env={})

    assert cfg.security.tool_overrides == {"read_file": "CONFIRM"}


def test_parse_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "PLAIN=value\n"
        'QUOTED="with space"\n'
        "SINGLE='single quoted'\n"
        "WITH_COMMENT=abc # trailing\n"
        "EMPTY=\n"
        "NO_EQUALS\n",
        encoding="utf-8",
    )
    parsed = parse_env_file(env_file)
    assert parsed["PLAIN"] == "value"
    assert parsed["QUOTED"] == "with space"
    assert parsed["SINGLE"] == "single quoted"
    assert parsed["WITH_COMMENT"] == "abc"
    assert parsed["EMPTY"] == ""
    assert "NO_EQUALS" not in parsed


def test_dotenv_lower_precedence_than_process_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("ARTHUR_ENV_FILE", str(env_file))
    monkeypatch.setenv("MODEL_NAME", "from-process")
    cfg = load_config(path=tmp_path / "none.toml")
    assert cfg.llm.model == "from-process"
