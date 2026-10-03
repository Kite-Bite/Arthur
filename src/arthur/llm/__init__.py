"""LLM package: backend-agnostic contract, Ollama client, prompts, test double."""

from arthur.llm.base import (
    ChatMessage,
    Health,
    LLMClient,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotFoundError,
)
from arthur.llm.ollama import OllamaClient

__all__ = [
    "ChatMessage",
    "Health",
    "LLMClient",
    "LLMError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "ModelNotFoundError",
    "OllamaClient",
]
