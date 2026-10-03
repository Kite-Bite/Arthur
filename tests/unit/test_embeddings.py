"""Embedding providers: determinism, normalisation, selection."""

from __future__ import annotations

import math

import pytest

from arthur.config.schema import LLMConfig, RetrievalConfig
from arthur.retrieval.embeddings import HashEmbedder, OllamaEmbedder, build_embedder


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
