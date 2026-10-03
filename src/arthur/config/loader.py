"""Configuration loading with a strict precedence order.

Precedence (lowest to highest)::

    built-in defaults  <  config.toml  <  .env file  <  process environment

The config file defaults to ``$XDG_CONFIG_HOME/arthur/config.toml`` (usually
``~/.config/arthur/config.toml``) and can be pointed elsewhere with the
``ARTHUR_CONFIG`` environment variable. Environment overrides use the
``ARTHUR_<SECTION>_<FIELD>`` naming scheme (e.g. ``ARTHUR_LLM_MODEL``), plus
the conventional ``OLLAMA_HOST`` / ``MODEL_NAME`` aliases.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from arthur.config.schema import Config

ENV_PREFIX = "ARTHUR_"

#: Convenience aliases so `.env` files can use familiar Ollama names.
LEGACY_ENV_MAP: dict[str, str] = {
    "OLLAMA_HOST": "llm.host",
    "MODEL_NAME": "llm.model",
}


def default_config_path() -> Path:
    """Return the default config file location, honouring XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "arthur" / "config.toml"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` dotenv file without third-party deps.

    Lines starting with ``#`` and blank lines are ignored. Values may be
    optionally wrapped in single or double quotes. ``#`` inside an unquoted
    value starts a comment.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def _flatten(defaults: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested section dicts into dotted leaf paths."""
    leaves: dict[str, Any] = {}
    for key, value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            leaves.update(_flatten(value, path))
        else:
            leaves[path] = value
    return leaves


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _coerce(raw: str, sample: Any) -> Any:
    """Coerce an environment string to the type of the default value."""
    if isinstance(sample, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(sample, int):
        try:
            return int(raw)
        except ValueError:
            return sample
    if isinstance(sample, float):
        try:
            return float(raw)
        except ValueError:
            return sample
    if isinstance(sample, list):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, list) else raw
            except json.JSONDecodeError:
                return sample
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(sample, dict):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else sample
        except json.JSONDecodeError:
            return sample
    return raw


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load and validate the effective configuration.

    Args:
        path: Explicit config file. When omitted, ``ARTHUR_CONFIG`` is checked
            and then the default location.
        env: Environment mapping to read overrides from (defaults to
            ``os.environ``). Process env beats ``.env`` beats config file.
    """
    environ: Mapping[str, str] = os.environ if env is None else env

    # 1. Defaults as a nested dict.
    defaults = Config().model_dump()
    leaves = _flatten(defaults)

    # 2. Config file.
    if path is not None:
        config_file: Path | None = Path(path).expanduser()
    elif environ.get("ARTHUR_CONFIG"):
        config_file = Path(environ["ARTHUR_CONFIG"]).expanduser()
    else:
        candidate = default_config_path()
        config_file = candidate if candidate.is_file() else None

    file_data: dict[str, Any] = {}
    if config_file is not None and config_file.is_file():
        try:
            file_data = tomllib.loads(config_file.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read config file {config_file}: {exc}") from exc

    merged = _deep_merge(defaults, file_data)

    # 3. .env file (lower precedence than the real environment).
    dotenv: dict[str, str] = {}
    env_file = environ.get("ARTHUR_ENV_FILE")
    dotenv_path = Path(env_file).expanduser() if env_file else Path.cwd() / ".env"
    dotenv = parse_env_file(dotenv_path)

    # 4. Environment overlays: dotenv first, then the real environment wins.
    for source in (dotenv, environ):
        for key, raw in source.items():
            target = LEGACY_ENV_MAP.get(key)
            if target is None:
                if not key.startswith(ENV_PREFIX):
                    continue
                target = key[len(ENV_PREFIX) :].lower().replace("_", ".")
            if target in leaves:
                _set_path(merged, target, _coerce(raw, leaves[target]))

    return Config.model_validate(merged)
