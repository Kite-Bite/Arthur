"""Health reporting: model availability, embedder status.

`OllamaClient.health()` is driven through `httpx.MockTransport` so the
reachable branches are covered without a live server - previously they only
executed when Ollama happened to be running, so CI never saw them.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from arthur.cli.commands import _embed_status
from arthur.config.schema import LLMConfig
from arthur.llm.ollama import OllamaClient

HOST = "http://ollama.test"

Responder = Callable[[httpx.Request], httpx.Response]


def _server(names: list[str], *, version: dict[str, object] | None = None) -> Responder:
    version_body = {"version": "0.18.3"} if version is None else version

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json=version_body)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in names]})
        return httpx.Response(404)

    return handler


def _client(names: list[str], *, model: str = "llama3.2:1b") -> OllamaClient:
    config = LLMConfig(host=HOST, model=model)
    transport = httpx.MockTransport(_server(names))
    return OllamaClient(config, client=httpx.Client(transport=transport, base_url=HOST))


def test_unreachable_server_reports_it_and_stops() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    config = LLMConfig(host=HOST, model="a", embedding_model="b")
    client = OllamaClient(
        config,
        client=httpx.Client(transport=httpx.MockTransport(refuse), base_url=HOST),
    )

    health = client.health()

    assert health.reachable is False
    assert f"cannot reach {HOST}" in health.detail
    # The tags lookup is skipped entirely, so neither model is claimed present.
    assert health.model_available is False
    assert health.embedding_available is False


def test_both_models_present() -> None:
    health = _client(["llama3.2:1b", "nomic-embed-text"], model="llama3.2:1b").health()

    assert health.reachable is True
    assert health.detail == "ollama 0.18.3"
    assert health.model_available is True
    assert health.embedding_available is True


def test_missing_embedding_model_is_reported_alongside_the_chat_model() -> None:
    """The case `doctor` used to miss: chat works, RAG does not."""
    health = _client(["llama3.2:1b"], model="llama3.2:1b").health()

    assert health.model_available is True
    assert health.embedding_available is False
    assert "embedding model 'nomic-embed-text' not pulled" in health.detail


def test_missing_chat_model_is_reported() -> None:
    health = _client(["nomic-embed-text"], model="llama3.2:1b").health()

    assert health.model_available is False
    assert health.embedding_available is True
    assert "model 'llama3.2:1b' not pulled" in health.detail


def test_both_missing_models_are_listed_in_one_detail() -> None:
    health = _client(["something-else"], model="llama3.2:1b").health()

    assert health.model_available is False
    assert health.embedding_available is False
    assert "model 'llama3.2:1b' not pulled" in health.detail
    assert "embedding model 'nomic-embed-text' not pulled" in health.detail


def test_both_models_are_checked_in_one_tags_request() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _server(["llama3.2:1b", "nomic-embed-text"])(request)

    config = LLMConfig(host=HOST, model="llama3.2:1b")
    client = OllamaClient(
        config, client=httpx.Client(transport=httpx.MockTransport(handler), base_url=HOST)
    )

    client.health()

    assert calls == ["/api/version", "/api/tags"]


def test_tagless_model_matches_the_latest_tag() -> None:
    """Ollama stores `llama3.2` as `llama3.2:latest`; exact match alone lied."""
    health = _client(["llama3.2:latest", "nomic-embed-text"], model="llama3.2").health()

    assert health.model_available is True


def test_tagless_model_is_not_satisfied_by_a_different_tag() -> None:
    """Only `llama3.2:3b` is pulled, so resolving `llama3.2` would fail."""
    health = _client(["llama3.2:3b", "nomic-embed-text"], model="llama3.2").health()

    assert health.model_available is False
    assert "not pulled" in health.detail


def test_explicit_tag_is_matched_exactly() -> None:
    health = _client(["llama3.2:latest"], model="llama3.2:3b").health()

    assert health.model_available is False


def test_version_body_without_a_version_key_is_tolerated() -> None:
    transport = httpx.MockTransport(_server([], version={"nope": 1}))
    config = LLMConfig(host=HOST, model="llama3.2:1b")
    client = OllamaClient(config, client=httpx.Client(transport=transport, base_url=HOST))

    health = client.health()

    assert health.reachable is True
    assert health.detail.startswith("ollama ?")


def test_tags_failure_leaves_health_best_effort() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.18.3"})
        return httpx.Response(500, text="tags exploded")

    config = LLMConfig(host=HOST, model="llama3.2:1b")
    client = OllamaClient(
        config, client=httpx.Client(transport=httpx.MockTransport(handler), base_url=HOST)
    )

    health = client.health()

    # Reachability is the primary signal and must survive a tags failure.
    assert health.reachable is True
    assert health.detail == "ollama 0.18.3"
    assert health.model_available is False


# --- _embed_status: the doctor row -----------------------------------------


def _status(**kwargs: object) -> tuple[str, str]:
    health: dict[str, object] = {
        "llm_reachable": True,
        "embedder": "ollama:nomic-embed-text",
        "embedding_model": "nomic-embed-text",
        "embedding_available": True,
        **kwargs,
    }
    return _embed_status(health)


def test_hash_embedder_needs_no_model() -> None:
    status, detail = _status(embedder="hash:384", embedding_available=False)

    assert status == "ok"
    assert "offline" in detail
    assert "nomic" not in detail


def test_ollama_embedder_with_model_present_is_ok() -> None:
    status, detail = _status()

    assert status == "ok"
    assert detail == "nomic-embed-text"


def test_ollama_embedder_with_model_missing_is_missing_with_a_fix() -> None:
    status, detail = _status(embedding_available=False)

    assert status == "MISSING"
    assert detail == "nomic-embed-text (run: ollama pull nomic-embed-text)"


def test_unreachable_ollama_does_not_give_a_misleading_pull_hint() -> None:
    status, detail = _status(llm_reachable=False, embedding_available=False)

    assert status == "UNAVAILABLE"
    assert "ollama pull" not in detail


@pytest.mark.parametrize(
    ("embedder", "reachable", "available", "expected"),
    [
        ("hash:384", False, False, "ok"),
        ("ollama:nomic-embed-text", True, True, "ok"),
        ("ollama:nomic-embed-text", True, False, "MISSING"),
        ("ollama:nomic-embed-text", False, False, "UNAVAILABLE"),
    ],
)
def test_embed_status_covers_every_combination(
    embedder: str, reachable: bool, available: bool, expected: str
) -> None:
    status, _ = _status(embedder=embedder, llm_reachable=reachable, embedding_available=available)
    assert status == expected
