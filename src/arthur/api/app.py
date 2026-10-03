"""FastAPI application - a thin HTTP shell over the same services the CLI uses.

No business logic lives here: every endpoint delegates to ``Services`` so the
CLI and API cannot drift apart. When ``ARTHUR_API_TOKEN`` is configured, every
endpoint except ``/health`` requires ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response

from arthur import __version__
from arthur.agent.types import AgentResult
from arthur.api.schemas import (
    ChatRequest,
    IndexRequest,
    LogEntry,
    MemoryCreateRequest,
    MemoryItemResponse,
    ToolExecuteRequest,
    ToolInfo,
)
from arthur.execution.types import ExecutionOutcome
from arthur.llm.base import LLMError
from arthur.security.confirmation import approve
from arthur.services import Services

OPENAPI_TAGS = ["meta", "chat", "tools", "memory", "documents", "logs"]


def _item(item: object) -> dict[str, object]:
    """Serialize a MemoryItem for API responses."""
    return {
        "id": item.id,  # type: ignore[attr-defined]
        "content": item.content,  # type: ignore[attr-defined]
        "category": item.category,  # type: ignore[attr-defined]
        "source": item.source,  # type: ignore[attr-defined]
        "importance": item.importance,  # type: ignore[attr-defined]
        "created_at": item.created_at,  # type: ignore[attr-defined]
        "expires_at": item.expires_at,  # type: ignore[attr-defined]
    }


def create_app(services: Services | None = None) -> FastAPI:
    """Build the FastAPI app around a (possibly shared) Services graph.

    Args:
        services: Pre-built services; when omitted, ones created here are
            disposed on shutdown.
    """
    owned = services is None
    state = {"services": services}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state["services"] is None:
            state["services"] = Services.create()
        yield
        if owned and state["services"] is not None:
            state["services"].close()

    app = FastAPI(
        title="Arthur API",
        version=__version__,
        description=(
            "Local-first AI engineering assistant. All operations go through the "
            "same tool registry, security policy and audit trail as the CLI."
        ),
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
    )

    def svc() -> Services:
        services_now = state["services"]
        if services_now is None:  # pragma: no cover - lifespan guarantees init
            state["services"] = Services.create()
            services_now = state["services"]
        return services_now  # type: ignore[return-value]

    async def require_auth(request: Request) -> None:
        token = svc().config.api.token
        if not token:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {token}":
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # --- meta ---------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        """LLM/storage health; intentionally unauthenticated for monitoring."""
        services = svc()
        info = services.health()
        return {
            "status": "ok" if info["llm_reachable"] else "degraded",
            "version": __version__,
            **info,
        }

    # --- chat ---------------------------------------------------------------

    @app.post("/chat", response_model=AgentResult, tags=["chat"],
              dependencies=[Depends(require_auth)])
    def chat(payload: ChatRequest) -> AgentResult:
        """Run one agent request (may execute tools under the security policy).

        When an operation needs confirmation and ``confirm`` is false, the
        response has ``stopped_reason="confirmation_required"`` and a
        ``pending_confirmation`` payload; re-post with ``confirm=true``.
        """
        services = svc()
        try:
            result = services.agent.run(
                payload.message,
                conversation_id=payload.conversation_id,
                confirm=approve if payload.confirm else None,
            )
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if result.stopped_reason == "llm_error":
            raise HTTPException(status_code=503, detail=result.error or "LLM error")
        return result

    # --- tools --------------------------------------------------------------

    @app.get("/tools", response_model=list[ToolInfo], tags=["tools"],
             dependencies=[Depends(require_auth)])
    def list_tools() -> list[ToolInfo]:
        """List registered tools with permissions and schemas."""
        return [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                permission=tool.permission.name,
                risk=tool.risk,
                action=tool.action or tool.description,
                parameters=tool.schema()["parameters"],
            )
            for tool in svc().registry.list()
        ]

    @app.post("/tools/{tool_name}/execute", response_model=ExecutionOutcome,
              tags=["tools"], dependencies=[Depends(require_auth)])
    def execute_tool(tool_name: str, payload: ToolExecuteRequest) -> ExecutionOutcome:
        """Execute a tool directly (same pipeline as agent-driven calls)."""
        return svc().executor.execute(
            tool_name,
            payload.arguments,
            confirm=approve if payload.confirm else None,
        )

    # --- memory -------------------------------------------------------------

    @app.get("/memory", tags=["memory"], dependencies=[Depends(require_auth)])
    def list_memory(
        query: str | None = Query(None, description="Keyword search"),
        category: str | None = Query(None),
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, object]:
        """List or search long-term memories."""
        memory = svc().memory
        if query:
            items = memory.search(query, limit=limit, category=category)
        else:
            items = memory.list(limit=limit, category=category)
        return {"count": len(items), "memories": [_item(i) for i in items]}

    @app.post("/memory", response_model=MemoryItemResponse, status_code=201,
              tags=["memory"], dependencies=[Depends(require_auth)])
    def create_memory(payload: MemoryCreateRequest) -> MemoryItemResponse:
        """Store a long-term memory."""
        item = svc().memory.add(
            payload.content, category=payload.category, importance=payload.importance
        )
        return MemoryItemResponse(**_item(item))

    @app.delete("/memory/{memory_id}", status_code=204, tags=["memory"],
                dependencies=[Depends(require_auth)])
    def delete_memory(memory_id: int) -> Response:
        """Delete one memory."""
        if not svc().memory.delete(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return Response(status_code=204)

    # --- documents ----------------------------------------------------------

    @app.post("/documents/index", tags=["documents"], dependencies=[Depends(require_auth)])
    def index_documents(payload: IndexRequest) -> dict[str, object]:
        """Index documents into the local vector store."""
        outcome = svc().executor.execute(
            "index_documents",
            {"path": payload.path, "recursive": payload.recursive},
        )
        if outcome.status != "success":
            raise HTTPException(status_code=400, detail=outcome.record.error)
        return {
            "summary": outcome.record.result_summary,
            "report": outcome.result.data if outcome.result else None,
        }

    @app.get("/documents/search", tags=["documents"], dependencies=[Depends(require_auth)])
    def search_documents(
        q: str = Query(..., min_length=1, description="Search query"),
        limit: int = Query(5, ge=1, le=20),
    ) -> dict[str, object]:
        """Semantic search over indexed documents."""
        outcome = svc().executor.execute("search_documents", {"query": q, "limit": limit})
        if outcome.status != "success":
            raise HTTPException(status_code=400, detail=outcome.record.error)
        return outcome.result.data if outcome.result else {"count": 0, "results": []}

    # --- logs ---------------------------------------------------------------

    @app.get("/logs", response_model=list[LogEntry], tags=["logs"],
             dependencies=[Depends(require_auth)])
    def logs(limit: int = Query(50, ge=1, le=500)) -> list[LogEntry]:
        """Recent tool executions from the audit trail."""
        rows = svc().executions.recent(limit)
        return [
            LogEntry(
                id=row.id,
                created_at=row.created_at,
                request_id=row.request_id,
                request=row.request,
                tool_name=row.tool_name,
                arguments=row.arguments,
                permission=row.permission,
                confirmed=row.confirmed,
                status=row.status,
                result_summary=row.result_summary,
                error=row.error,
                duration_ms=row.duration_ms,
            )
            for row in rows
        ]

    return app
