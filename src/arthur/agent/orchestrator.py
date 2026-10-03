"""The agent loop: decide -> act -> observe -> repeat -> answer.

Reliability features (Phase 6):

* JSON parse repairs with feedback to the model (bounded by ``max_repairs``),
* bounded steps (``max_steps``) with a graceful fallback answer,
* structured per-step records for explainability,
* confirmation-aware execution (pending requests stop the loop cleanly),
* citation sanitisation against actually retrieved sources,
* conversation persistence and memory injection.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from arthur.agent.parser import Decision, DecisionParseError, parse_decision
from arthur.agent.types import AgentResult, AgentStep
from arthur.config.schema import Config
from arthur.database.repos import ConversationRepository
from arthur.execution.executor import ConfirmationMode, ToolExecutor
from arthur.llm.base import ChatMessage, LLMClient, LLMError
from arthur.llm.prompts import build_answer_system_prompt, build_system_prompt
from arthur.memory.store import MemoryStore
from arthur.retrieval.citations import sanitize_citations
from arthur.retrieval.service import Retriever

logger = logging.getLogger("arthur.agent")

StepCallback = Callable[[AgentStep], None]
TokenCallback = Callable[[str], None]

OBSERVATION_CAP = 4000


class Agent:
    """Orchestrates one request through tools to a grounded final answer."""

    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        config: Config,
        *,
        memory: MemoryStore | None = None,
        retrieval: Retriever | None = None,
        conversations: ConversationRepository | None = None,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.config = config
        self.memory = memory
        self.retrieval = retrieval
        self.conversations = conversations

    def run(
        self,
        request: str,
        *,
        conversation_id: str | None = None,
        history: list[ChatMessage] | None = None,
        confirm: ConfirmationMode = None,
        on_step: StepCallback | None = None,
        on_token: TokenCallback | None = None,
    ) -> AgentResult:
        """Run the full agent loop for one user request.

        Args:
            request: The user's natural-language request.
            conversation_id: Continue an existing conversation (persisted).
            history: Explicit history (tests/advanced callers); overrides DB.
            confirm: Confirmation behaviour forwarded to the executor.
            on_step: Called after every tool step (live trace in the CLI).
            on_token: Called with each streamed answer token.
        """
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        request = request.strip()
        result = AgentResult(request=request, request_id=request_id)

        messages = self._opening_messages(request, conversation_id, history, result)
        memories = self._recall_memories(request)
        indexed = self._indexed_sources()
        system_prompt = build_system_prompt(
            self.config,
            self.executor.registry.schemas(),
            memories=memories,
            indexed_sources=indexed,
        )
        messages = [ChatMessage(role="system", content=system_prompt), *messages]

        retrieved_sources: set[str] = set()
        repairs = 0
        answer = ""

        try:
            for _step_index in range(self.config.agent.max_steps):
                raw = self.llm.complete(
                    messages, max_tokens=self.config.llm.decide_max_tokens
                )
                try:
                    decision = parse_decision(raw)
                except DecisionParseError as exc:
                    repairs += 1
                    logger.debug("decision parse failure %d: %s", repairs, exc)
                    if repairs > self.config.agent.max_repairs:
                        result.stopped_reason = "parse_failure"
                        break
                    messages.append(ChatMessage(role="assistant", content=raw))
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                f"INVALID ACTION: {exc}. Respond with exactly one valid "
                                "JSON object following the decision protocol."
                            ),
                        )
                    )
                    continue

                if decision.tool is None:
                    answer = self._generate_answer(messages, on_token)
                    result.stopped_reason = "answered"
                    break

                outcome = self.executor.execute(
                    decision.tool,
                    decision.args or {},
                    request_id=request_id,
                    request_text=request,
                    confirm=confirm,
                )
                observation = outcome.observation()
                step = AgentStep(
                    index=len(result.steps) + 1,
                    thought=decision.thought,
                    plan=decision.plan,
                    tool=decision.tool,
                    args=decision.args or {},
                    status=outcome.status,
                    permission=outcome.record.permission.name,
                    observation=observation[:2000],
                    duration_ms=outcome.record.duration_ms,
                    error=outcome.record.error,
                    confirmation=outcome.request,
                )
                result.steps.append(step)
                result.used_tools.append(decision.tool)
                if on_step is not None:
                    on_step(step)

                messages.append(ChatMessage(role="assistant", content=raw))
                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"OBSERVATION ({decision.tool}):\n{observation[:OBSERVATION_CAP]}",
                    )
                )

                if decision.tool == "search_documents":
                    retrieved_sources.update(_sources_from_outcome(outcome))

                if outcome.status == "confirmation_required":
                    result.stopped_reason = "confirmation_required"
                    result.pending_confirmation = outcome.request
                    break
            else:
                result.stopped_reason = "max_steps"

            if result.stopped_reason == "max_steps":
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Step limit reached. Give your best final answer now using only "
                            "the observations above."
                        ),
                    )
                )
                answer = self._generate_answer(messages, on_token)
            elif result.stopped_reason == "parse_failure":
                try:
                    answer = self._generate_answer(messages, on_token)
                except LLMError:  # pragma: no cover - already degraded
                    answer = ""
                if not answer:
                    answer = (
                        "I could not produce a valid action plan for this request "
                        "(the model repeatedly returned malformed decisions)."
                    )
        except LLMError as exc:
            result.stopped_reason = "llm_error"
            result.error = str(exc)
            logger.warning("agent run %s failed: %s", request_id, exc)
            answer = (
                f"I could not complete that request: {exc}"
            )

        if retrieved_sources and answer:
            cleaned, kept, removed = sanitize_citations(answer, retrieved_sources)
            answer = cleaned
            result.citations = kept
            result.removed_citations = removed
            if removed:
                logger.warning(
                    "removed %d fabricated citation(s): %s", len(removed), ", ".join(removed)
                )
        result.sources = sorted(retrieved_sources)
        result.answer = answer
        result.duration_ms = (time.perf_counter() - started) * 1000

        if self.conversations is not None:
            result.conversation_id = self._persist(
                conversation_id, request, answer, result
            )
        return result

    # --- helpers -----------------------------------------------------------

    def _opening_messages(
        self,
        request: str,
        conversation_id: str | None,
        history: list[ChatMessage] | None,
        result: AgentResult,
    ) -> list[ChatMessage]:
        if history is not None:
            return [*history, ChatMessage(role="user", content=request)]
        if conversation_id and self.conversations is not None:
            stored = self.conversations.history(
                conversation_id, limit=self.config.llm.history_messages
            )
            messages = [ChatMessage(role=r, content=c) for r, c in stored]
            return [*messages, ChatMessage(role="user", content=request)]
        return [ChatMessage(role="user", content=request)]

    def _recall_memories(self, request: str) -> list[Any]:
        if not (
            self.config.agent.memory_injection
            and self.memory is not None
            and self.config.memory.auto_recall
        ):
            return []
        try:
            return self.memory.search(request, limit=self.config.memory.recall_limit)
        except Exception as exc:  # noqa: BLE001 - recall is best-effort
            logger.debug("memory recall failed: %s", exc)
            return []

    def _indexed_sources(self) -> list[str]:
        if self.retrieval is None:
            return []
        try:
            return self.retrieval.indexed_sources()
        except Exception as exc:  # noqa: BLE001 - listing is best-effort
            logger.debug("indexed source listing failed: %s", exc)
            return []

    def _generate_answer(
        self, messages: list[ChatMessage], on_token: TokenCallback | None
    ) -> str:
        answer_messages = [
            ChatMessage(role="system", content=build_answer_system_prompt(self.config)),
            *messages[1:],  # drop the decision-protocol system prompt
            ChatMessage(
                role="user",
                content="Provide the final answer now as plain text (no JSON, no tools).",
            ),
        ]
        if on_token is not None:
            buffer: list[str] = []
            for token in self.llm.stream(
                answer_messages, max_tokens=self.config.llm.answer_max_tokens
            ):
                buffer.append(token)
                on_token(token)
            return "".join(buffer).strip()
        return self.llm.complete(
            answer_messages, max_tokens=self.config.llm.answer_max_tokens
        ).strip()

    def _persist(
        self,
        conversation_id: str | None,
        request: str,
        answer: str,
        result: AgentResult,
    ) -> str | None:
        assert self.conversations is not None
        try:
            conversation = None
            if conversation_id:
                conversation = self.conversations.get(conversation_id)
            if conversation is None:
                conversation = self.conversations.create(title=request[:80])
            self.conversations.add_message(conversation.id, "user", request)
            if answer:
                self.conversations.add_message(conversation.id, "assistant", answer)
            elif result.pending_confirmation is not None:
                self.conversations.add_message(
                    conversation.id,
                    "assistant",
                    f"confirmation required: {result.pending_confirmation.action} "
                    f"-> {result.pending_confirmation.target}",
                )
            return conversation.id
        except Exception as exc:  # noqa: BLE001 - persistence must not fail the reply
            logger.error("failed to persist conversation: %s", exc)
            return conversation_id


def _sources_from_outcome(outcome: Any) -> set[str]:
    sources: set[str] = set()
    data = outcome.result.data if outcome.result is not None else None
    if isinstance(data, dict):
        for item in data.get("results", []) or []:
            if isinstance(item, dict) and item.get("source"):
                sources.add(str(item["source"]))
    return sources
