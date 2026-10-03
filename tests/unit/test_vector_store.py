"""SQLite vector store: add, cosine query, source deletion."""

from __future__ import annotations

import math

import pytest

from arthur.database.session import create_engine, create_session_factory, init_db
from arthur.retrieval.store import SqliteVectorStore, VectorItem


def _vec(values: dict[int, float], dimensions: int = 8) -> list[float]:
    vector = [0.0] * dimensions
    for index, value in values.items():
        vector[index] = value
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture()
def store() -> SqliteVectorStore:
    engine = create_engine(":memory:")
    init_db(engine)
    return SqliteVectorStore(create_session_factory(engine))


def test_add_and_count(store: SqliteVectorStore) -> None:
    items = [
        VectorItem(id="a:0", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0})),
        VectorItem(id="b:0", source="b", chunk_index=0, text="beta", embedding=_vec({1: 1.0})),
    ]
    assert store.add(items) == 2
    assert store.count() == 2


def test_query_orders_by_similarity(store: SqliteVectorStore) -> None:
    store.add(
        [
            VectorItem(id="a", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0})),
            VectorItem(id="b", source="b", chunk_index=0, text="beta", embedding=_vec({1: 1.0})),
            VectorItem(id="c", source="c", chunk_index=0, text="gamma", embedding=_vec({2: 1.0})),
        ]
    )
    hits = store.query(_vec({1: 1.0, 0: 0.2}), k=2)
    assert [hit.id for hit in hits] == ["b", "a"]
    assert hits[0].score >= hits[1].score


def test_query_respects_k(store: SqliteVectorStore) -> None:
    store.add(
        [
            VectorItem(id=f"s{i}", source="s", chunk_index=i, text="t", embedding=_vec({0: 1.0}))
            for i in range(10)
        ]
    )
    assert len(store.query(_vec({0: 1.0}), k=3)) == 3


def test_query_dimension_mismatch_returns_empty(store: SqliteVectorStore) -> None:
    store.add(
        [VectorItem(id="a", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0}))]
    )
    assert store.query([1.0, 0.0, 0.0], k=5) == []  # 3 dims vs stored 8


def test_delete_source(store: SqliteVectorStore) -> None:
    store.add(
        [
            VectorItem(id="a:0", source="a", chunk_index=0, text="alpha", embedding=_vec({0: 1.0})),
            VectorItem(id="b:0", source="b", chunk_index=0, text="beta", embedding=_vec({1: 1.0})),
        ]
    )
    assert store.delete_source("a") == 1
    assert store.count() == 1
    hits = store.query(_vec({0: 1.0}), k=5)
    assert all(hit.source != "a" for hit in hits)


def test_upsert_replaces_same_id(store: SqliteVectorStore) -> None:
    item = VectorItem(id="a:0", source="a", chunk_index=0, text="v1", embedding=_vec({0: 1.0}))
    store.add([item])
    item.text = "v2"
    store.add([item])
    assert store.count() == 1
    hits = store.query(_vec({0: 1.0}), k=1)
    assert hits[0].text == "v2"
