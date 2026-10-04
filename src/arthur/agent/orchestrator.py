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
from typing import Any, cast

from arthur.agent.parser import DecisionParseError, decision_issue, parse_decision
from arthur.agent.types import AgentResult, AgentStep
from arthur.config.schema import Config
from arthur.database.repos import ConversationRepository
from arthur.execution.executor import ConfirmationMode, ToolExecutor
from arthur.llm.base import ChatMessage, ChatRole, LLMClient, LLMError
from arthur.llm.prompts import build_answer_system_prompt, build_system_prompt
from arthur.memory.store import MemoryStore
from arthur.retrieval.citations import sanitize_citations
from arthur.retrieval.service import Retriever

logger = logging.getLogger("arthur.agent")

StepCallback = Callable[[AgentStep], None]
TokenCallback = Callable[[str], None]

OBSERVATION_CAP = 4000


def _looks_like_decision(content: str) -> bool:
    """True when an assistant message is a decision-protocol transcript.

    The answer phase must not be fed our own JSON decisions back - small models
    then reply in JSON as well. A leading object carrying decision keys counts,
    including truncated output; prose answers from earlier turns are kept.
    """
    stripped = content.lstrip()
    if not stripped.startswith("{"):
        return False
    head = stripped[:400]
    return any(f'"{key}"' in head for key in ("thought", "tool", "args", "plan"))


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
        context = self._auto_retrieve(request)
        if context is not None:
            passage_text, sources = context
            retrieved_sources.update(sources)
            messages.append(ChatMessage(role="user", content=passage_text))
            example = sorted(retrieved_sources)[0]
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Use the passages above when they answer the request. To cite one, "
                        "copy its exact path from the passage into a marker like this: "
                        f"[source: {example}]. Use the real path, never a placeholder. "
                        "Otherwise call a tool."
                    ),
                )
            )

        repairs = 0
        answer = ""

        try:
            for _step_index in range(self.config.agent.max_steps):
                raw = self.llm.complete(
                    messages,
                    max_tokens=self.config.llm.decide_max_tokens,
                    temperature=self.config.llm.decide_temperature,
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

                issue = decision_issue(decision)
                if issue is not None:
                    repairs += 1
                    logger.debug("decision inconsistency %d: %s", repairs, issue)
                    if repairs > self.config.agent.max_repairs:
                        result.stopped_reason = "parse_failure"
                        break
                    messages.append(ChatMessage(role="assistant", content=raw))
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                f"INVALID ACTION: {issue}. Respond with exactly one valid "
                                "JSON object following the decision protocol."
                            ),
                        )
                    )
                    continue

                if decision.tool is None:
                    answer = self._generate_answer(messages, on_token, retrieved_sources)
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
                answer = self._generate_answer(messages, on_token, retrieved_sources)
            elif result.stopped_reason == "parse_failure":
                try:
                    answer = self._generate_answer(messages, on_token, retrieved_sources)
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
            answer = f"I could not complete that request: {exc}"

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
            result.conversation_id = self._persist(conversation_id, request, answer, result)
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
            messages = [ChatMessage(role=cast(ChatRole, r), content=c) for r, c in stored]
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

    def _auto_retrieve(self, request: str) -> tuple[str, list[str]] | None:
        """Ground the request in indexed passages before the first decision.

        Small local models frequently skip the ``search_documents`` tool and
        answer from parametric memory, which produces invented facts. Retrieving
        up-front fixes that; the tool remains available for follow-up queries.

        Returns:
            ``(passage_block, sources)`` or ``None`` when disabled/empty.
        """
        if not self.config.agent.auto_retrieve or self.retrieval is None:
            return None
        try:
            if self.retrieval.store.count() == 0:
                return None
            passages = self.retrieval.search(request, k=self.config.agent.auto_retrieve_limit)
        except Exception as exc:  # noqa: BLE001 - grounding is best-effort
            logger.debug("auto retrieval failed: %s", exc)
            return None
        floor = self.config.retrieval.auto_retrieve_min_score
        passages = [passage for passage in passages if passage.score >= floor]
        if not passages:
            return None
        lines = []
        sources: list[str] = []
        for i, passage in enumerate(passages, start=1):
            sources.append(passage.source)
            snippet = " ".join(passage.text.split())[:600]
            lines.append(f"{i}. [source: {passage.source}] (score {passage.score:.2f}) {snippet}")
        block = "## Retrieved passages (local document index)\n" + "\n".join(lines)
        return block, sources

    def _generate_answer(
        self,
        messages: list[ChatMessage],
        on_token: TokenCallback | None,
        sources: set[str] | None = None,
    ) -> str:
        # The conversation carries our own JSON decisions; echoing them into the
        # answer context makes small models answer in JSON too. Keep the user
        # turns and observations, drop the decision transcripts.
        context = [
            message
            for message in messages[1:]
            if not (message.role == "assistant" and _looks_like_decision(message.content))
        ]
        answer_messages = [
            ChatMessage(role="system", content=build_answer_system_prompt(self.config)),
            *context,
            ChatMessage(
                role="user",
                content=(
                    "Write the final answer for the user now, as a short prose or "
                    "bullet-point summary in natural language. Do NOT echo raw JSON, "
                    "tool names or observation dumps - translate them. No tools."
                ),
            ),
        ]
        if sources:
            answer_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Your answer draws on retrieved documents. End it with a "
                        "citation marker for the document you used, written exactly "
                        "as [source: <path>] using one of these paths: "
                        + ", ".join(sorted(sources))
                    ),
                )
            )
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
