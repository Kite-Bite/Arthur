"""Vector stores behind a small, swappable interface.

Two backends:

* :class:`SqliteVectorStore` (default) - embeddings as BLOBs in Arthur's own
  SQLite database with NumPy cosine scoring. Linear scan is fine for the
  personal-document scale this tool targets (tens of thousands of chunks).
* :class:`ChromaVectorStore` (optional extra) - for when a dedicated vector
  DB is genuinely needed; installed via ``uv sync --extra chroma``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from arthur.config.schema import RetrievalConfig
from arthur.database.models import DocumentChunk
from arthur.database.session import rowcount


@dataclass
class VectorItem:
    """One chunk + embedding to store."""

    id: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    """One search hit."""

    id: str
    source: str
    chunk_index: int
    text: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Contract for vector backends (cosine similarity, higher = better)."""

    def add(self, items: Sequence[VectorItem]) -> int: ...

    def query(self, vector: Sequence[float], k: int) -> list[VectorHit]: ...

    def delete_source(self, source: str) -> int: ...

    def count(self) -> int: ...


class SqliteVectorStore:
    """Built-in store: DocumentChunk rows + NumPy dot-product ranking."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def add(self, items: Sequence[VectorItem]) -> int:
        if not items:
            return 0
        with self._sf() as session:
            for item in items:
                session.merge(
                    DocumentChunk(
                        id=item.id,
                        source=item.source,
                        chunk_index=item.chunk_index,
                        text=item.text,
                        embedding=_to_blob(item.embedding),
                        dimensions=len(item.embedding),
                        meta=item.meta or None,
                    )
                )
            session.commit()
        return len(items)

    def query(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        if k <= 0:
            return []
        with self._sf() as session:
            rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.source,
                    DocumentChunk.chunk_index,
                    DocumentChunk.text,
                    DocumentChunk.embedding,
                    DocumentChunk.meta,
                    DocumentChunk.dimensions,
                )
            ).all()
        if not rows:
            return []
        dimensions = int(rows[0].dimensions)
        query = np.asarray(vector, dtype=np.float32)
        if query.size != dimensions or dimensions == 0:
            return []
        matrix = np.vstack([_from_blob(row.embedding, dimensions) for row in rows]).astype(
            np.float32
        )
        scores = matrix @ query
        order = np.argsort(scores)[::-1][:k]
        return [
            VectorHit(
                id=rows[i].id,
                source=rows[i].source,
                chunk_index=int(rows[i].chunk_index),
                text=rows[i].text,
                score=float(scores[i]),
                meta=dict(rows[i].meta or {}),
            )
            for i in order
        ]

    def delete_source(self, source: str) -> int:
        with self._sf() as session:
            result = session.execute(delete(DocumentChunk).where(DocumentChunk.source == source))
            session.commit()
            return rowcount(result)

    def count(self) -> int:
        with self._sf() as session:
            return int(session.execute(select(func.count(DocumentChunk.id))).scalar() or 0)


class ChromaVectorStore:
    """Optional ChromaDB backend (``uv sync --extra chroma``)."""

    def __init__(self, path: str) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ChromaDB backend requested but chromadb is not installed; "
                "run: uv sync --extra chroma"
            ) from exc
        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            name="arthur_documents", metadata={"hnsw:space": "cosine"}
        )

    def add(self, items: Sequence[VectorItem]) -> int:
        if not items:
            return 0
        self._collection.upsert(
            ids=[i.id for i in items],
            documents=[i.text for i in items],
            embeddings=[i.embedding for i in items],
            metadatas=[{"source": i.source, "chunk_index": i.chunk_index, **i.meta} for i in items],
        )
        return len(items)

    def query(self, vector: Sequence[float], k: int) -> list[VectorHit]:
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorHit] = []
        ids = result.get("ids") or [[]]
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        for idx, doc, meta, dist in zip(ids[0], docs[0], metas[0], dists[0], strict=True):
            hits.append(
                VectorHit(
                    id=idx,
                    source=str(meta.get("source", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc or "",
                    score=1.0 - float(dist),
                    meta=dict(meta),
                )
            )
        return hits

    def delete_source(self, source: str) -> int:
        self._collection.delete(where={"source": source})
        return 0

    def count(self) -> int:
        return int(self._collection.count())


def build_vector_store(
    config: RetrievalConfig, session_factory: sessionmaker[Session]
) -> VectorStore:
    """Construct the configured vector backend."""
    if config.backend == "chroma":
        from pathlib import Path

        chroma_path = Path(config.chroma_path).expanduser()
        chroma_path.mkdir(parents=True, exist_ok=True)
        return ChromaVectorStore(str(chroma_path))
    return SqliteVectorStore(session_factory)


def _to_blob(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dimensions: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dimensions)
