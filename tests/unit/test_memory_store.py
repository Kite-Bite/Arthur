"""Memory store: CRUD, search ranking, expiry."""

from __future__ import annotations

from datetime import timedelta

import pytest

from arthur.database.models import utcnow
from arthur.database.session import create_engine, create_session_factory, init_db
from arthur.memory.store import MemoryStore


@pytest.fixture()
def store() -> MemoryStore:
    engine = create_engine(":memory:")
    init_db(engine)
    return MemoryStore(create_session_factory(engine))


def test_add_and_get(store: MemoryStore) -> None:
    item = store.add("Main projects live under ~/Projects", category="preference")
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.content == "Main projects live under ~/Projects"
    assert fetched.category == "preference"
    assert fetched.source == "user"


def test_empty_content_rejected(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.add("   ")


def test_search_ranks_relevant_first(store: MemoryStore) -> None:
    store.add("Btrfs snapshot commands are btrfs subvolume snapshot", category="linux")
    store.add("Preferred editor is Neovim", category="preference")
    store.add("btrfs send receive for backups", category="linux")
    results = store.search("btrfs snapshots", limit=5)
    assert len(results) == 2
    assert "btrfs" in results[0].content.casefold()


def test_search_no_overlap_returns_empty(store: MemoryStore) -> None:
    store.add("completely unrelated content here")
    assert store.search("quantum entanglement") == []


def test_search_respects_limit_and_category(store: MemoryStore) -> None:
    for i in range(5):
        store.add(f"btrfs note {i}", category="linux" if i % 2 else "misc")
    results = store.search("btrfs", limit=2, category="misc")
    assert len(results) <= 2
    assert all(item.category == "misc" for item in results)


def test_delete(store: MemoryStore) -> None:
    item = store.add("temp memory")
    assert store.delete(item.id) is True
    assert store.get(item.id) is None
    assert store.delete(item.id) is False


def test_delete_all(store: MemoryStore) -> None:
    store.add("a")
    store.add("b")
    assert store.delete_all() == 2
    assert store.count() == 0


def test_purge_expired(store: MemoryStore) -> None:
    store.add("evergreen")
    store.add(
        "stale",
        expires_at=utcnow() - timedelta(minutes=1),
    )
    store.purge_expired()
    remaining = store.list()
    assert len(remaining) == 1
    assert remaining[0].content == "evergreen"


def test_list_newest_first(store: MemoryStore) -> None:
    first = store.add("older")
    second = store.add("newer")
    items = store.list()
    assert [item.id for item in items] == [second.id, first.id]
