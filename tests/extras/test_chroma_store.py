"""ChromaDB vector backend (optional: ``uv sync --extra chroma``).

Chroma must honour the same ``VectorStore`` contract as the built-in SQLite
backend, which the unit suite pins down. These tests run only when the extra
is installed; CI has a dedicated job so the backend is never untested there.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from arthur.config.schema import RetrievalConfig, SecurityConfig
from arthur.database.repos import DocumentRepository
from arthur.database.session import create_engine, create_session_factory, init_db
from arthur.retrieval.embeddings import HashEmbedder
from arthur.retrieval.service import Retriever
from arthur.retrieval.store import ChromaVectorStore, VectorItem, build_vector_store
from arthur.security.paths import PathPolicy

pytest.importorskip("chromadb", reason="chroma extra not installed: uv sync --extra chroma")


def _vec(values: dict[int, float], dimensions: int = 8) -> list[float]:
    vector = [0.0] * dimensions
    for index, value in values.items():
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture()
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(str(tmp_path / "chroma"))


def test_add_and_count(store: ChromaVectorStore) -> None:
    items = [
        VectorItem(id="a:0", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0})),
        VectorItem(id="b:0", source="b", chunk_index=0, text="beta", embedding=_vec({1: 1.0})),
    ]

    assert store.add(items) == 2
    assert store.count() == 2


def test_add_of_nothing_is_a_no_op(store: ChromaVectorStore) -> None:
    assert store.add([]) == 0
    assert store.count() == 0


def test_query_ranks_by_cosine_and_scores_like_sqlite(store: ChromaVectorStore) -> None:
    store.add(
        [
            VectorItem(id="near", source="s", chunk_index=0, text="near", embedding=_vec({0: 1.0})),
            VectorItem(
                id="far", source="s", chunk_index=1, text="far", embedding=_vec({0: 0.2, 7: 0.9})
            ),
        ]
    )

    hits = store.query(_vec({0: 1.0}), k=2)

    assert [h.id for h in hits] == ["near", "far"]
    # Score must be 1 - cosine distance so it is comparable with SQLite's.
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score < hits[0].score
    assert 0.0 <= hits[1].score <= 1.0


def test_query_with_no_room_returns_nothing(store: ChromaVectorStore) -> None:
    assert store.query(_vec({0: 1.0}), k=0) == []


def test_delete_source_reports_the_number_removed(store: ChromaVectorStore) -> None:
    """Regression: this used to return 0 unconditionally."""
    store.add(
        [
            VectorItem(id="a:0", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0})),
            VectorItem(id="a:1", source="a", chunk_index=1, text="gamma", embedding=_vec({1: 1.0})),
            VectorItem(id="b:0", source="b", chunk_index=0, text="beta", embedding=_vec({2: 1.0})),
        ]
    )

    assert store.delete_source("a") == 2
    assert store.count() == 1
    assert store.delete_source("a") == 0  # already gone
    assert store.delete_source("never-seen") == 0


def test_build_vector_store_returns_the_chroma_backend(tmp_path: Path) -> None:
    config = RetrievalConfig(backend="chroma", chroma_path=str(tmp_path / "chroma"))

    engine = create_engine(":memory:")
    init_db(engine)

    built = build_vector_store(config, create_session_factory(engine))

    assert isinstance(built, ChromaVectorStore)


def test_retriever_ingestion_round_trip_is_idempotent(tmp_path: Path) -> None:
    """Re-indexing must replace, not accumulate - which needs delete_source."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text(
        "# Deploy\nDeploys run through shipd. Rollback is automatic.\n", encoding="utf-8"
    )
    (docs / "net.md").write_text(
        "# Network\nThe load balancer listens on port 8443.\n", encoding="utf-8"
    )

    config = RetrievalConfig(backend="chroma", chroma_path=str(tmp_path / "chroma"))
    engine = create_engine(":memory:")
    init_db(engine)
    session_factory = create_session_factory(engine)
    retriever = Retriever(
        config,
        HashEmbedder(),
        ChromaVectorStore(str(tmp_path / "chroma")),
        DocumentRepository(session_factory),
        PathPolicy(SecurityConfig(allowed_roots=[str(tmp_path)])),
    )

    first = retriever.index_path(docs)
    assert first.files_indexed == 2
    assert retriever.store.count() == first.chunks_added

    # Re-index the same tree: chunks are replaced, so the total must not grow.
    second = retriever.index_path(docs)
    assert second.chunks_added == first.chunks_added
    assert retriever.store.count() == first.chunks_added

    hits = retriever.search("rollback after a deploy", k=2)
    assert hits
    assert any("rollback" in hit.text.lower() for hit in hits)

    # Deleting a source clears it from both the index and the store.
    assert retriever.store.delete_source(str(docs / "deploy.md")) >= 1
    assert retriever.store.count() < first.chunks_added
