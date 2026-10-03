"""LLM client contract and errors.

The agent depends only on this protocol, so alternative backends (llama.cpp,
vLLM, or a cloud adapter) can be added later without touching the agent.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class LLMError(Exception):
    """Base class for LLM communication failures."""


class LLMUnavailableError(LLMError):
    """The inference server cannot be reached."""


class ModelNotFoundError(LLMError):
    """The configured model is not available on the server."""


class LLMTimeoutError(LLMError):
    """The request exceeded the configured timeout."""


#: The three roles accepted by the LLM chat protocol.
ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One chat message."""

    role: ChatRole
    content: str


class Health(BaseModel):
    """LLM backend health snapshot (used by ``/health`` and ``arthur doctor``)."""

    reachable: bool = False
    detail: str = ""
    model: str = ""
    model_available: bool = False


@runtime_checkable
class LLMClient(Protocol):
    """What the agent needs from a language model backend."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Synchronous single-shot completion."""
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream completion tokens as they are generated."""
        ...

    def health(self) -> Health:
        """Report server reachability and configured-model availability."""
        ...
