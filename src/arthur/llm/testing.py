"""Scripted LLM for tests and offline demos.

Implements the same protocol as :class:`OllamaClient` but returns canned
responses (or a callable's output), recording every call so tests can assert
on prompts. This is how the agent loop is tested without a real model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from arthur.llm.base import ChatMessage, Health


class ScriptedLLM:
    """Deterministic LLM double: pops queued responses or calls a responder."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        responder: Callable[[list[ChatMessage]], str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self._error = error
        self.calls: list[list[ChatMessage]] = []
        #: Temperature passed to every ``complete`` call, in call order.
        self.temperatures: list[float | None] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(list(messages))
        self.temperatures.append(temperature)
        if self._error is not None:
            raise self._error
        if self._responder is not None:
            return self._responder(list(messages))
        if not self._responses:
            raise AssertionError(
                f"ScriptedLLM exhausted after {len(self.calls)} call(s); "
                "the agent made more LLM calls than the test scripted"
            )
        return self._responses.pop(0)

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        text = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        for word in text.split(" "):
            yield (word + " ") if word else ""

    def health(self) -> Health:
        return Health(reachable=True, detail="scripted", model="scripted", model_available=True)
