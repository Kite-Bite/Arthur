"""Long-term memory store: explicit, inspectable, deletable.

Search uses lightweight lexical scoring (token overlap weighted by importance)
rather than embeddings: memory entries are short and few, so a linear scan is
fast, deterministic and works with zero model dependencies. Semantic memory
recall can reuse the retrieval layer later if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from arthur.database.models import MemoryRecord, utcnow
from arthur.database.session import rowcount

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MemoryItem:
    """One memory entry as seen by callers."""

    id: int
    content: str
    category: str
    source: str
    importance: float
    created_at: datetime
    expires_at: datetime | None = None
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens of length >= 2."""
    return [t for t in _TOKEN_RE.findall(text.casefold()) if len(t) >= 2]


#: ``MemoryStore.list`` shadows the builtin inside the class body, so return
#: annotations reference this module-level alias instead of ``list[...]``.
MemoryItemList = list[MemoryItem]


class MemoryStore:
    """CRUD + search over the ``memory_records`` table."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        default_importance: float = 0.5,
    ) -> None:
        self._sf = session_factory
        self.default_importance = default_importance

    def add(
        self,
        content: str,
        *,
        category: str = "general",
        source: str = "user",
        importance: float | None = None,
        expires_at: datetime | None = None,
        extra: dict | None = None,
    ) -> MemoryItem:
        content = content.strip()
        if not content:
            raise ValueError("memory content must not be empty")
        with self._sf() as session:
            record = MemoryRecord(
                content=content,
                category=category.strip() or "general",
                source=source,
                importance=self.default_importance if importance is None else importance,
                expires_at=expires_at,
                extra=extra,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return _to_item(record)

    def get(self, memory_id: int) -> MemoryItem | None:
        with self._sf() as session:
            record = session.get(MemoryRecord, memory_id)
            return _to_item(record) if record else None

    def list(self, *, limit: int = 50, category: str | None = None) -> MemoryItemList:
        self.purge_expired()
        with self._sf() as session:
            stmt = select(MemoryRecord)
            if category:
                stmt = stmt.where(MemoryRecord.category == category)
            records = session.execute(
                stmt.order_by(MemoryRecord.created_at.desc()).limit(limit)
            ).scalars()
            return [_to_item(r) for r in records]

    def search(self, query: str, *, limit: int = 5, category: str | None = None) -> MemoryItemList:
        """Rank memories by token overlap, weighted by importance."""
        self.purge_expired()
        tokens = tokenize(query)
        if not tokens:
            return []
        with self._sf() as session:
            stmt = select(MemoryRecord)
            if category:
                stmt = stmt.where(MemoryRecord.category == category)
            records = list(session.execute(stmt).scalars())

        scored: MemoryItemList = []
        for record in records:
            haystack = tokenize(f"{record.content} {record.category}")
            if not haystack:
                continue
            counts = {t: haystack.count(t) for t in set(tokens)}
            overlap = sum(counts.values())
            if overlap == 0:
                continue
            score = overlap + 0.5 * record.importance
            scored.append(_to_item(record, score=score))
        scored.sort(key=lambda item: (item.score, item.created_at), reverse=True)
        return scored[:limit]

    def delete(self, memory_id: int) -> bool:
        with self._sf() as session:
            result = session.execute(delete(MemoryRecord).where(MemoryRecord.id == memory_id))
            session.commit()
            return rowcount(result) > 0

    def delete_all(self, *, category: str | None = None) -> int:
        with self._sf() as session:
            stmt = delete(MemoryRecord)
            if category:
                stmt = stmt.where(MemoryRecord.category == category)
            result = session.execute(stmt)
            session.commit()
            return rowcount(result)

    def count(self) -> int:
        with self._sf() as session:
            return int(session.execute(select(func.count(MemoryRecord.id))).scalar() or 0)

    def purge_expired(self) -> int:
        """Delete entries whose expiration has passed."""
        with self._sf() as session:
            result = session.execute(
                delete(MemoryRecord).where(
                    MemoryRecord.expires_at.is_not(None),
                    MemoryRecord.expires_at < utcnow(),
                )
            )
            session.commit()
            return rowcount(result)


def _to_item(record: MemoryRecord, *, score: float = 0.0) -> MemoryItem:
    return MemoryItem(
        id=record.id,
        content=record.content,
        category=record.category,
        source=record.source,
        importance=record.importance,
        created_at=record.created_at,
        expires_at=record.expires_at,
        score=score,
    )
