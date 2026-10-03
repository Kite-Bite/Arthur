"""Embedding providers - all local, no cloud APIs.

Two implementations:

* :class:`OllamaEmbedder` - high-quality embeddings from a locally running
  Ollama server (recommended; default configuration).
* :class:`HashEmbedder` - deterministic feature-hashing embeddings with zero
  dependencies. Used for tests and as an explicit offline fallback; quality is
  lexical (bag-of-words), not semantic.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from arthur.config.schema import LLMConfig, RetrievalConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """Embedding generation failed (server down, model missing, bad input)."""


@runtime_checkable
class Embedder(Protocol):
    """Contract for embedding providers."""

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input string."""
        ...


class OllamaEmbedder:
    """Embeddings via a local Ollama server (``/api/embed`` with legacy fallback)."""

    def __init__(
        self,
        host: str,
        model: str,
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self._client = client or httpx.Client(base_url=self.host, timeout=timeout)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = list(texts)
        try:
            response = self._client.post(
                "/api/embed", json={"model": self.model, "input": payload}
            )
            if response.status_code == 404:
                return self._embed_legacy(payload)
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed: {_http_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(_unreachable(self.host, exc)) from exc
        except ValueError as exc:
            raise EmbeddingError(f"Ollama returned invalid JSON: {exc}") from exc

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(payload):
            raise EmbeddingError("Ollama returned an unexpected embedding payload")
        try:
            return [_normalize([float(x) for x in vec]) for vec in embeddings]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(f"malformed embedding vector: {exc}") from exc

    def _embed_legacy(self, payload: list[str]) -> list[list[float]]:
        """Older Ollama servers expose only per-prompt ``/api/embeddings``."""
        vectors: list[list[float]] = []
        try:
            for text in payload:
                response = self._client.post(
                    "/api/embeddings", json={"model": self.model, "prompt": text}
                )
                if response.status_code == 404 and "not found" in _http_detail(response).lower():
                    raise EmbeddingError(
                        f"embedding model {self.model!r} is not available in Ollama "
                        f"(run: ollama pull {self.model})"
                    )
                response.raise_for_status()
                body: dict[str, Any] = response.json()
                vector = body.get("embedding")
                if not isinstance(vector, list):
                    raise EmbeddingError("Ollama returned an unexpected embedding payload")
                vectors.append(_normalize([float(x) for x in vector]))
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed: {_http_detail(exc.response)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(_unreachable(self.host, exc)) from exc
        except ValueError as exc:
            raise EmbeddingError(f"Ollama returned invalid JSON: {exc}") from exc
        return vectors

    def close(self) -> None:
        self._client.close()


class HashEmbedder:
    """Dependency-free feature-hashing embedder (lexical, deterministic).

    Words and word bigrams are hashed into a fixed-size vector which is then
    L2-normalized. This gives useful keyword-level retrieval for tests and
    offline use, but it does not capture synonymy the way a real embedding
    model does - configure ``retrieval.embedder = "ollama"`` for semantics.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions
        self.name = f"hash:{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_embed_hash(text, self.dimensions) for text in texts]


def _embed_hash(text: str, dimensions: int) -> list[float]:
    tokens = _TOKEN_RE.findall(text.casefold())
    vector = [0.0] * dimensions
    features = list(tokens)
    features.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False))
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _unreachable(host: str, exc: httpx.HTTPError) -> str:
    return (
        f"cannot reach Ollama at {host} ({exc.__class__.__name__}); "
        "is `ollama serve` running?"
    )


def _http_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            return str(body["error"])
    except ValueError:  # pragma: no cover - non-JSON error bodies
        pass
    return response.text[:300] or response.reason_phrase


def build_embedder(retrieval: RetrievalConfig, llm: LLMConfig) -> Embedder:
    """Construct the configured embedding provider."""
    if retrieval.embedder == "hash":
        return HashEmbedder()
    return OllamaEmbedder(llm.host, llm.embedding_model, timeout=llm.timeout_seconds)
