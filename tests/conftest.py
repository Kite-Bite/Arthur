"""Shared fixtures: sandboxed config, service graph, scripted decisions."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arthur.config.schema import Config
from arthur.execution.executor import ToolExecutor
from arthur.llm.testing import ScriptedLLM
from arthur.security.policy import SecurityPolicy
from arthur.services import Services
from arthur.tools.base import ToolContext
from arthur.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _reset_arthur_logging() -> None:
    """Drop handlers installed by CLI/API code so captures stay clean."""
    yield
    logger = logging.getLogger("arthur")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


@pytest.fixture()
def base_config(tmp_path: Path) -> Config:
    """Config that is safe and deterministic inside pytest's tmp sandbox."""
    cfg = Config()
    cfg.memory.database = ":memory:"
    cfg.retrieval.embedder = "hash"
    cfg.logging.file = None
    cfg.security.allowed_roots = [str(tmp_path)]
    return cfg


@pytest.fixture()
def services(base_config: Config):
    """A full Services graph with an in-memory database and hash embeddings."""
    svc = Services.create(config=base_config, configure_logs=False)
    try:
        yield svc
    finally:
        svc.close()


@pytest.fixture()
def decision() -> Callable[..., str]:
    """Factory for well-formed decision JSON (what a cooperative model emits)."""

    def _make(
        tool: str | None,
        args: dict[str, Any] | None = None,
        *,
        thought: str = "test step",
        answer: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "thought": thought,
            "plan": [],
            "tool": tool,
            "args": args,
            "answer": answer,
        }
        return json.dumps(payload)

    return _make


@pytest.fixture()
def executor_factory(base_config: Config):
    """Build an executor around a custom tool set with a capturing audit sink."""

    def _make(tools: list, *, config: Config | None = None):
        cfg = config or base_config
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        policy = SecurityPolicy(cfg.security)
        ctx = ToolContext(config=cfg, paths=policy.paths)
        records: list = []
        executor = ToolExecutor(registry, policy, ctx, audit=records.append)
        return executor, records

    return _make


@pytest.fixture()
def scripted(services: Services):
    """Swap a scripted LLM into the services graph; returns the double."""

    def _make(responses: list[str] | None = None, **kwargs: Any) -> ScriptedLLM:
        llm = ScriptedLLM(responses, **kwargs)
        services.swap_llm(llm)
        return llm

    return _make
