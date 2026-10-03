"""Composition root: wires configuration into every subsystem exactly once.

Both the CLI and the FastAPI app consume :class:`Services`, so there is a
single place where components are constructed and no business logic is
duplicated between interfaces.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from arthur.agent.orchestrator import Agent
from arthur.config import Config, load_config
from arthur.database.repos import ConversationRepository, DocumentRepository, ExecutionRepository
from arthur.database.session import create_engine, create_session_factory, init_db
from arthur.execution.executor import ToolExecutor
from arthur.llm.base import LLMClient
from arthur.llm.ollama import OllamaClient
from arthur.logging.audit import AuditLogger
from arthur.logging.setup import configure_logging
from arthur.memory.store import MemoryStore
from arthur.retrieval.embeddings import Embedder, build_embedder
from arthur.retrieval.service import Retriever
from arthur.retrieval.store import VectorStore, build_vector_store
from arthur.security.policy import SecurityPolicy
from arthur.tools.base import ToolContext
from arthur.tools.registry import ToolRegistry, default_registry


class Services:
    """Holds every runtime component; created via :meth:`create`."""

    def __init__(
        self,
        *,
        config: Config,
        engine: Engine,
        session_factory: sessionmaker[Session],
        llm: LLMClient,
        embedder: Embedder,
        vector_store: VectorStore,
        memory: MemoryStore,
        retrieval: Retriever,
        registry: ToolRegistry,
        policy: SecurityPolicy,
        executor: ToolExecutor,
        audit: AuditLogger,
        conversations: ConversationRepository,
        documents: DocumentRepository,
        executions: ExecutionRepository,
        agent: Agent,
    ) -> None:
        self.config = config
        self.engine = engine
        self.session_factory = session_factory
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.memory = memory
        self.retrieval = retrieval
        self.registry = registry
        self.policy = policy
        self.executor = executor
        self.audit = audit
        self.conversations = conversations
        self.documents = documents
        self.executions = executions
        self.agent = agent

    @classmethod
    def create(
        cls,
        config_path: str | Path | None = None,
        *,
        config: Config | None = None,
        configure_logs: bool = True,
    ) -> Services:
        """Build the full application graph from configuration.

        Args:
            config_path: Explicit config file (see :func:`load_config`).
            config: Pre-built configuration (tests / API overrides).
            configure_logs: Initialise logging (disabled for repeated builds).
        """
        cfg = config if config is not None else load_config(config_path)
        if configure_logs:
            configure_logging(cfg.logging)

        engine = create_engine(cfg.memory.database)
        init_db(engine)
        session_factory = create_session_factory(engine)

        audit = AuditLogger(session_factory)
        llm = OllamaClient(cfg.llm)
        registry = default_registry()
        policy = SecurityPolicy(cfg.security)
        memory = MemoryStore(session_factory, default_importance=cfg.memory.default_importance)
        documents = DocumentRepository(session_factory)
        embedder = build_embedder(cfg.retrieval, cfg.llm)
        vector_store = build_vector_store(cfg.retrieval, session_factory)
        retrieval = Retriever(cfg.retrieval, embedder, vector_store, documents, policy.paths)
        ctx = ToolContext(config=cfg, paths=policy.paths, memory=memory, retrieval=retrieval)
        executor = ToolExecutor(registry, policy, ctx, audit=audit)
        conversations = ConversationRepository(session_factory)
        executions = ExecutionRepository(session_factory)
        agent = Agent(
            llm,
            executor,
            cfg,
            memory=memory,
            retrieval=retrieval,
            conversations=conversations,
        )
        return cls(
            config=cfg,
            engine=engine,
            session_factory=session_factory,
            llm=llm,
            embedder=embedder,
            vector_store=vector_store,
            memory=memory,
            retrieval=retrieval,
            registry=registry,
            policy=policy,
            executor=executor,
            audit=audit,
            conversations=conversations,
            documents=documents,
            executions=executions,
            agent=agent,
        )

    def swap_llm(self, llm: LLMClient) -> None:
        """Replace the LLM client and rebuild the agent around it (tests)."""
        self.llm = llm
        self.agent = Agent(
            llm,
            self.executor,
            self.config,
            memory=self.memory,
            retrieval=self.retrieval,
            conversations=self.conversations,
        )

    def health(self) -> dict[str, object]:
        """Aggregate health information for CLI/API health endpoints."""
        llm_health = self.llm.health()
        return {
            "llm_reachable": llm_health.reachable,
            "llm_detail": llm_health.detail,
            "model": llm_health.model,
            "model_available": llm_health.model_available,
            "tools": len(self.registry),
            "indexed_documents": len(self.documents.list_all()),
            "memories": self.memory.count(),
            "audit_records": self.executions.count(),
        }

    def close(self) -> None:
        """Release long-lived resources."""
        for component in (self.llm, self.embedder):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - best-effort shutdown
                    pass
        self.engine.dispose()

    def __enter__(self) -> Services:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
