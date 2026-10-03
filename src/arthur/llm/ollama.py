"""Ollama chat client (local, no cloud APIs).

Uses ``/api/chat`` for completions and streaming, ``/api/version`` +
``/api/tags`` for health. All failure modes are mapped to typed errors so the
agent and API can return useful messages instead of stack traces.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from arthur.config.schema import LLMConfig
from arthur.llm.base import (
    ChatMessage,
    Health,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotFoundError,
)


class OllamaClient:
    """Thin typed client for a local Ollama server."""

    def __init__(self, config: LLMConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(
            base_url=config.host.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_seconds, connect=5.0),
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = self._payload(
            messages, stream=False, max_tokens=max_tokens, temperature=temperature
        )
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(response)
        body = self._json(response)
        message = body.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError(f"Ollama returned no message content: {str(body)[:200]}")
        return content

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        payload = self._payload(
            messages, stream=True, max_tokens=max_tokens, temperature=temperature
        )
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    response.read()  # buffer body so the error detail is readable
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(f"malformed stream chunk: {line[:120]}") from exc
                    if chunk.get("error"):
                        raise LLMError(str(chunk["error"]))
                    message = chunk.get("message") or {}
                    content = message.get("content")
                    if content:
                        yield str(content)
                    if chunk.get("done"):
                        return
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc

    def health(self) -> Health:
        health = Health(model=self.config.model)
        try:
            version = self._client.get("/api/version", timeout=3.0)
            version.raise_for_status()
            health.reachable = True
            health.detail = f"ollama {version.json().get('version', '?')}"
        except (httpx.HTTPError, ValueError) as exc:
            health.reachable = False
            health.detail = f"cannot reach {self.config.host}: {exc.__class__.__name__}"
            return health
        try:
            tags = self._client.get("/api/tags", timeout=3.0)
            tags.raise_for_status()
            names = {m.get("name", "") for m in tags.json().get("models", [])}
            health.model_available = self.config.model in names
            if not health.model_available:
                health.detail += f"; model {self.config.model!r} not pulled"
        except (httpx.HTTPError, ValueError):  # pragma: no cover - health is best-effort
            pass
        return health

    def close(self) -> None:
        self._client.close()

    # --- internals ---------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: bool,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "num_predict": self.config.answer_max_tokens if max_tokens is None else max_tokens,
        }
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": options,
            "keep_alive": self.config.keep_alive,
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = _detail(response)
        if response.status_code == 404 and "not found" in detail.lower():
            raise ModelNotFoundError(
                f"model {self.config.model!r} is not available in Ollama "
                f"(run: ollama pull {self.config.model})"
            )
        if response.status_code == 404:
            raise LLMUnavailableError(f"Ollama endpoint not found at {self.config.host} ({detail})")
        raise LLMError(f"Ollama error (HTTP {response.status_code}): {detail}")

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {exc}") from exc
        if isinstance(body, dict) and body.get("error"):
            raise LLMError(str(body["error"]))
        if not isinstance(body, dict):
            raise LLMError("Ollama returned an unexpected payload")
        return body

    @staticmethod
    def _unreachable(exc: httpx.HTTPError) -> LLMError:
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(f"Ollama request timed out ({exc})")
        return LLMUnavailableError(
            f"cannot reach Ollama ({exc.__class__.__name__}); is `ollama serve` running?"
        )


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            return str(body["error"])
    except ValueError:
        pass
    return (response.text or response.reason_phrase)[:300]
