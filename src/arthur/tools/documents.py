"""Document tools: build and query the local RAG index."""

from __future__ import annotations

from pydantic import BaseModel, Field

from arthur.execution.errors import ToolError
from arthur.tools.base import Tool, ToolContext, ToolResult


class IndexDocumentsArgs(BaseModel):
    path: str = Field(description="File or directory to index")
    recursive: bool = Field(True, description="Recurse into subdirectories")


class IndexDocumentsTool(Tool):
    name = "index_documents"
    description = (
        "Ingest text documents under a path into the local vector index so "
        "they can be found with search_documents. Safe: only reads files and "
        "writes Arthur's own index."
    )
    action = "Index documents for semantic search"
    args_model = IndexDocumentsArgs

    def run(self, args: IndexDocumentsArgs, ctx: ToolContext) -> ToolResult:
        if ctx.retrieval is None:
            raise ToolError("retrieval service is not available")
        target = ctx.paths.resolve(args.path)
        if not target.exists():
            raise ToolError(f"not found: {target}")
        report = ctx.retrieval.index_path(target, recursive=args.recursive)
        return ToolResult(
            summary=(
                f"indexed {report.files_indexed} files "
                f"({report.chunks_added} chunks, {report.skipped} skipped)"
            ),
            data=report.model_dump(),
        )


class SearchDocumentsArgs(BaseModel):
    query: str = Field(description="Natural-language search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum passages to return")


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = (
        "Semantic search over indexed documents. Returns the most relevant "
        "passages with their source paths - cite these sources verbatim."
    )
    action = "Search indexed documents"
    args_model = SearchDocumentsArgs

    def run(self, args: SearchDocumentsArgs, ctx: ToolContext) -> ToolResult:
        if ctx.retrieval is None:
            raise ToolError("retrieval service is not available")
        passages = ctx.retrieval.search(args.query, k=args.limit)
        if not passages:
            return ToolResult(
                ok=True,
                summary="no matching passages found (are the documents indexed?)",
                data={"count": 0, "results": []},
            )
        results = [
            {
                "source": p.source,
                "chunk_index": p.chunk_index,
                "score": round(p.score, 4),
                "text": p.text,
            }
            for p in passages
        ]
        sources = sorted({p.source for p in passages})
        return ToolResult(
            summary=f"{len(results)} passages from: {', '.join(sources)}",
            data={"count": len(results), "results": results},
        )
