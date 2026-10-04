"""Embedding providers: determinism, normalisation, selection.

The Ollama embedder is exercised through ``httpx.MockTransport`` so every
failure mode - server down, model missing, outdated server, garbage payload -
is covered without a running Ollama. ``OllamaEmbedder`` takes an injectable
client precisely for this.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import httpx
import pytest

from arthur.config.schema import LLMConfig, RetrievalConfig
from arthur.retrieval.embeddings import (
    Embedder,
    EmbeddingError,
    HashEmbedder,
    OllamaEmbedder,
    build_embedder,
)

HOST = "http://ollama.test"

Responder = Callable[[httpx.Request], httpx.Response]


def test_hash_embedder_deterministic() -> None:
    embedder = HashEmbedder(dimensions=64)
    first = embedder.embed(["btrfs snapshots"])[0]
    second = embedder.embed(["btrfs snapshots"])[0]
    assert first == second


def test_hash_embedder_normalized() -> None:
    embedder = HashEmbedder(dimensions=64)
    vector = embedder.embed(["some interesting text about linux"])[0]
    norm = math.sqrt(sum(v * v for v in vector))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hash_embedder_different_texts_differ() -> None:
    embedder = HashEmbedder(dimensions=64)
    a, b = embedder.embed(["alpha beta gamma", "delta epsilon zeta"])
    assert a[0] != b[0] or a != b


def test_hash_embedder_empty_text_zero_vector() -> None:
    embedder = HashEmbedder(dimensions=32)
    vector = embedder.embed([""])[0]
    assert all(v == 0.0 for v in vector)
    assert len(vector) == 32


def test_hash_embedder_batch() -> None:
    embedder = HashEmbedder(dimensions=16)
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3 and all(len(v) == 16 for v in vectors)


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimensions=4)


def test_build_embedder_hash() -> None:
    embedder = build_embedder(RetrievalConfig(embedder="hash"), LLMConfig())
    assert isinstance(embedder, HashEmbedder)


def test_build_embedder_ollama_default() -> None:
    embedder = build_embedder(RetrievalConfig(), LLMConfig(model="m"))
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "nomic-embed-text"


def test_ollama_embedder_unreachable_raises() -> None:
    embedder = OllamaEmbedder("http://127.0.0.1:1", "nomic-embed-text", timeout=0.5)
    with pytest.raises(Exception, match="cannot reach Ollama"):
        embedder.embed(["text"])


# --- OllamaEmbedder against a mock transport -------------------------------


def _embedder(responder: Responder) -> OllamaEmbedder:
    """An embedder whose HTTP goes to ``responder`` instead of the network."""
    client = httpx.Client(transport=httpx.MockTransport(responder), base_url=HOST)
    return OllamaEmbedder(HOST, "nomic-embed-text", client=client)


def _record(calls: list[httpx.Request], responder: Responder) -> Responder:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responder(request)

    return handler


def test_ollama_embedder_name_and_trailing_slash() -> None:
    embedder = OllamaEmbedder("http://ollama.test/", "m", client=httpx.Client())
    assert embedder.name == "ollama:m"
    assert embedder.host == "http://ollama.test"
    embedder.close()


def test_ollama_embed_returns_l2_normalized_vectors() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [[3.0, 4.0]]}))

    vectors = embedder.embed(["hello"])

    assert vectors == [pytest.approx([0.6, 0.8])]


def test_ollama_embed_posts_model_and_input() -> None:
    calls: list[httpx.Request] = []
    embedder = _embedder(
        _record(calls, lambda req: httpx.Response(200, json={"embeddings": [[1.0], [1.0]]}))
    )

    embedder.embed(["alpha", "beta"])

    assert len(calls) == 1
    assert calls[0].url.path == "/api/embed"
    body = json.loads(calls[0].content)
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["alpha", "beta"]


def test_ollama_embed_empty_input_never_contacts_the_server() -> None:
    calls: list[httpx.Request] = []
    embedder = _embedder(_record(calls, lambda req: httpx.Response(500)))

    assert embedder.embed([]) == []
    assert calls == []


def test_ollama_embed_allows_a_zero_vector() -> None:
    """The degenerate norm must not divide by zero."""
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [[0.0, 0.0]]}))

    assert embedder.embed(["nothing here"]) == [[0.0, 0.0]]


def test_ollama_embed_rejects_non_list_payload() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": "oops"}))

    with pytest.raises(EmbeddingError, match="unexpected embedding payload"):
        embedder.embed(["hello"])


def test_ollama_embed_rejects_missing_payload() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"other": []}))

    with pytest.raises(EmbeddingError, match="unexpected embedding payload"):
        embedder.embed(["hello"])


def test_ollama_embed_rejects_wrong_vector_count() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [[1.0, 2.0]]}))

    with pytest.raises(EmbeddingError, match="unexpected embedding payload"):
        embedder.embed(["one", "two"])


def test_ollama_embed_rejects_non_numeric_components() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [["a", "b"]]}))

    with pytest.raises(EmbeddingError, match="malformed embedding vector"):
        embedder.embed(["hello"])


def test_ollama_embed_rejects_null_components() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [[None]]}))

    with pytest.raises(EmbeddingError, match="malformed embedding vector"):
        embedder.embed(["hello"])


def test_ollama_embed_surfaces_the_servers_error_detail() -> None:
    embedder = _embedder(lambda req: httpx.Response(500, json={"error": "model is loading"}))

    with pytest.raises(EmbeddingError, match="model is loading"):
        embedder.embed(["hello"])


def test_ollama_embed_falls_back_to_the_body_when_there_is_no_error_field() -> None:
    embedder = _embedder(lambda req: httpx.Response(500, text="upstream exploded"))

    with pytest.raises(EmbeddingError, match="upstream exploded"):
        embedder.embed(["hello"])


def test_ollama_embed_falls_back_to_the_reason_phrase_when_the_body_is_empty() -> None:
    embedder = _embedder(lambda req: httpx.Response(503))

    with pytest.raises(EmbeddingError, match="Service Unavailable"):
        embedder.embed(["hello"])


def test_ollama_embed_truncates_a_huge_error_body() -> None:
    embedder = _embedder(lambda req: httpx.Response(500, text="x" * 5000))

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed(["hello"])

    detail = str(excinfo.value)
    # _http_detail caps at 300 characters, so 301 identical ones must not appear.
    assert "x" * 300 in detail
    assert "x" * 301 not in detail
    assert len(detail) < 400


def test_ollama_embed_reports_invalid_json() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, text="{not json"))

    with pytest.raises(EmbeddingError, match="invalid JSON"):
        embedder.embed(["hello"])


def test_ollama_embed_reports_a_refused_connection() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    embedder = _embedder(refuse)

    with pytest.raises(EmbeddingError, match="cannot reach Ollama") as excinfo:
        embedder.embed(["hello"])

    assert "is `ollama serve` running?" in str(excinfo.value)


def test_ollama_embed_reports_a_timeout() -> None:
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    embedder = _embedder(stall)

    with pytest.raises(EmbeddingError, match="cannot reach Ollama at " + HOST):
        embedder.embed(["hello"])


def test_ollama_embed_close_closes_the_client() -> None:
    embedder = _embedder(lambda req: httpx.Response(200, json={"embeddings": [[1.0]]}))

    embedder.close()

    assert embedder._client.is_closed


# --- legacy fallback for servers without /api/embed ------------------------


def _legacy(endpoint: Responder, legacy: Responder) -> Responder:
    """Route modern vs legacy endpoints."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return endpoint(request)
        return legacy(request)

    return handler


def test_legacy_fallback_used_when_embed_endpoint_is_missing() -> None:
    calls: list[httpx.Request] = []
    embedder = _embedder(
        _record(
            calls,
            _legacy(
                lambda req: httpx.Response(404, json={"error": "Not Found"}),
                lambda req: httpx.Response(200, json={"embedding": [3.0, 4.0]}),
            ),
        )
    )

    vectors = embedder.embed(["alpha", "beta"])

    # One request per prompt: that is the whole point of the legacy endpoint.
    assert vectors == [pytest.approx([0.6, 0.8]), pytest.approx([0.6, 0.8])]
    assert [r.url.path for r in calls] == ["/api/embed", "/api/embeddings", "/api/embeddings"]
    assert json.loads(calls[1].content) == {"model": "nomic-embed-text", "prompt": "alpha"}


def test_legacy_reports_a_missing_embedding_model() -> None:
    embedder = _embedder(
        _legacy(
            lambda req: httpx.Response(404, json={"error": "Not Found"}),
            lambda req: httpx.Response(404, json={"error": "model 'nomic' not found"}),
        )
    )

    with pytest.raises(EmbeddingError, match="ollama pull nomic-embed-text"):
        embedder.embed(["alpha"])


def test_legacy_404_without_a_not_found_marker_is_a_status_error() -> None:
    embedder = _embedder(
        _legacy(
            lambda req: httpx.Response(404, json={"error": "Not Found"}),
            lambda req: httpx.Response(404, json={"error": "route missing"}),
        )
    )

    with pytest.raises(EmbeddingError, match="route missing"):
        embedder.embed(["alpha"])


def test_legacy_rejects_a_missing_embedding_field() -> None:
    embedder = _embedder(
        _legacy(
            lambda req: httpx.Response(404, json={"error": "Not Found"}),
            lambda req: httpx.Response(200, json={"embedding": None}),
        )
    )

    with pytest.raises(EmbeddingError, match="unexpected embedding payload"):
        embedder.embed(["alpha"])


def test_legacy_surfaces_http_errors() -> None:
    embedder = _embedder(
        _legacy(
            lambda req: httpx.Response(404, json={"error": "Not Found"}),
            lambda req: httpx.Response(500, json={"error": "legacy exploded"}),
        )
    )

    with pytest.raises(EmbeddingError, match="legacy exploded"):
        embedder.embed(["alpha"])


def test_legacy_reports_invalid_json() -> None:
    embedder = _embedder(
        _legacy(
            lambda req: httpx.Response(404, json={"error": "Not Found"}),
            lambda req: httpx.Response(200, text="garbage"),
        )
    )

    with pytest.raises(EmbeddingError, match="invalid JSON"):
        embedder.embed(["alpha"])


def test_legacy_reports_an_unreachable_server() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    embedder = _embedder(_legacy(lambda req: httpx.Response(404), refuse))

    with pytest.raises(EmbeddingError, match="cannot reach Ollama"):
        embedder.embed(["alpha"])


def test_both_providers_satisfy_the_embedder_protocol() -> None:
    """The runtime-checkable contract the orchestrator dispatches on."""
    assert isinstance(HashEmbedder(dimensions=8), Embedder)
    assert isinstance(
        _embedder(lambda req: httpx.Response(200, json={"embeddings": [[1.0]]})), Embedder
    )
