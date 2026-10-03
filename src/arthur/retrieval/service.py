"""RAG service: discovery -> extraction -> chunking -> embedding -> storage.

:class:`Retriever` is the single façade used by tools, CLI and API, so the
ingestion pipeline stays testable and the vector backend stays swappable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker

from arthur.config.schema import RetrievalConfig
from arthur.database.repos import DocumentRepository
from arthur.retrieval.chunking import chunk_text
from arthur.retrieval.embeddings import Embedder, EmbeddingError
from arthur.retrieval.extractors import (
    ExtractionError,
    UnsupportedFormatError,
    discover_files,
    extract_text,
)
from arthur.retrieval.store import VectorItem, VectorStore
from arthur.security.paths import PathPolicy


class Passage(BaseModel):
    """One retrieved, citable text passage."""

    source: str
    chunk_index: int
    score: float
    text: str
    title: str = ""


class IndexReport(BaseModel):
    """Outcome of an ingestion run."""

    files_indexed: int = 0
    chunks_added: int = 0
    skipped: int = 0
    errors: list[str] = Field(default_factory=list)


class Retriever:
    """Document ingestion and semantic search over the configured backend."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Embedder,
        store: VectorStore,
        documents: DocumentRepository,
        paths: PathPolicy,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.store = store
        self.documents = documents
        self.paths = paths

    def index_path(self, root: Path, *, recursive: bool = True) -> IndexReport:
        """Index a file or directory tree.

        Re-indexing a path replaces its previous chunks, so ingestion is
        idempotent. Embedding failures abort with a clear error instead of
        silently writing an inconsistent index.
        """
        report = IndexReport()
        formats = set(self.config.formats)
        files = sorted(
            set(
                discover_files(
                    root,
                    formats=formats,
                    recursive=recursive,
                    denied_dir_names=self.paths.denied_dir_names,
                )
            )
        )
        for path in files:
            try:
                text, ext = extract_text(
                    path, formats=formats, max_bytes=self.config.max_file_bytes
                )
            except (UnsupportedFormatError, ExtractionError, OSError) as exc:
                report.skipped += 1
                report.errors.append(f"{path}: {exc}")
                continue
            chunks = chunk_text(
                text, size=self.config.chunk_size, overlap=self.config.chunk_overlap
            )
            if not chunks:
                report.skipped += 1
                continue
            try:
                vectors = self.embedder.embed(chunks)
            except EmbeddingError as exc:
                raise EmbeddingError(
                    f"cannot embed {path}: {exc}. Fix the embedding setup or set "
                    'retrieval.embedder = "hash" for dependency-free retrieval.'
                ) from exc
            source = str(path)
            self.store.delete_source(source)
            self.store.add(
                [
                    VectorItem(
                        id=f"{source}:{index}",
                        source=source,
                        chunk_index=index,
                        text=chunk,
                        embedding=vector,
                        meta={"title": path.name},
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ]
            )
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - race with deletion
                stat = None
            self.documents.upsert(
                path=source,
                title=path.name,
                format=ext,
                size_bytes=stat.st_size if stat else 0,
                mtime=stat.st_mtime if stat else 0.0,
                chunk_count=len(chunks),
            )
            report.files_indexed += 1
            report.chunks_added += len(chunks)
        return report

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Semantic search over all indexed documents."""
        limit = k or self.config.max_passages
        if not query.strip():
            return []
        if self.store.count() == 0:
            return []
        vector = self.embedder.embed([query])[0]
        hits = self.store.query(vector, limit)
        return [
            Passage(
                source=hit.source,
                chunk_index=hit.chunk_index,
                score=hit.score,
                text=hit.text,
                title=str(hit.meta.get("title", "")),
            )
            for hit in hits
        ]

    def indexed_sources(self) -> list[str]:
        """Paths of every document currently in the index."""
        return [record.path for record in self.documents.list_all()]
