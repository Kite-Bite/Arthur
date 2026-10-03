"""Repository classes: thin, typed data access over the SQLAlchemy session."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

from arthur.database.models import Conversation, DocumentRecord, Message, ToolExecution, utcnow
from arthur.execution.types import ExecutionRecord


class ConversationRepository:
    """Chat history persistence (the short-term memory layer)."""

    def __init__(self, session_factory: sessionmaker[object]) -> None:
        self._sf = session_factory

    def create(self, title: str | None = None) -> Conversation:
        with self._sf() as session:  # type: ignore[operator]
            conversation = Conversation(id=_new_id(), title=title)
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        with self._sf() as session:  # type: ignore[operator]
            return session.get(Conversation, conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        with self._sf() as session:  # type: ignore[operator]
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise KeyError(f"unknown conversation: {conversation_id}")
            message = Message(conversation_id=conversation_id, role=role, content=content)
            session.add(message)
            conversation.updated_at = utcnow()
            session.commit()
            session.refresh(message)
            return message

    def history(
        self, conversation_id: str, limit: int = 20
    ) -> list[tuple[str, str]]:
        """Return (role, content) pairs, oldest first, capped at ``limit``."""
        with self._sf() as session:  # type: ignore[operator]
            rows = session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc())
                .limit(limit)
            ).scalars()
            messages = [(m.role, m.content) for m in rows]
        return list(reversed(messages))

    def recent(self, limit: int = 20) -> list[Conversation]:
        with self._sf() as session:  # type: ignore[operator]
            return list(
                session.execute(
                    select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
                ).scalars()
            )


class ExecutionRepository:
    """Read side of the audit trail (``arthur logs`` / ``GET /logs``)."""

    def __init__(self, session_factory: sessionmaker[object]) -> None:
        self._sf = session_factory

    def recent(self, limit: int = 20) -> list[ToolExecution]:
        with self._sf() as session:  # type: ignore[operator]
            return list(
                session.execute(
                    select(ToolExecution)
                    .order_by(ToolExecution.id.desc())
                    .limit(limit)
                ).scalars()
            )

    def since(self, moment: datetime) -> list[ToolExecution]:
        with self._sf() as session:  # type: ignore[operator]
            return list(
                session.execute(
                    select(ToolExecution)
                    .where(ToolExecution.created_at >= moment)
                    .order_by(ToolExecution.id.desc())
                ).scalars()
            )

    def by_request(self, request_id: str) -> list[ToolExecution]:
        with self._sf() as session:  # type: ignore[operator]
            return list(
                session.execute(
                    select(ToolExecution)
                    .where(ToolExecution.request_id == request_id)
                    .order_by(ToolExecution.id)
                ).scalars()
            )

    def count(self) -> int:
        with self._sf() as session:  # type: ignore[operator]
            return int(session.execute(select(func.count(ToolExecution.id))).scalar() or 0)


class DocumentRepository:
    """Metadata for indexed documents (chunks live in the vector store)."""

    def __init__(self, session_factory: sessionmaker[object]) -> None:
        self._sf = session_factory

    def upsert(
        self,
        *,
        path: str,
        title: str,
        format: str,
        size_bytes: int,
        mtime: float,
        chunk_count: int,
    ) -> DocumentRecord:
        with self._sf() as session:  # type: ignore[operator]
            record = session.execute(
                select(DocumentRecord).where(DocumentRecord.path == path)
            ).scalar_one_or_none()
            if record is None:
                record = DocumentRecord(path=path)
                session.add(record)
            record.title = title
            record.format = format
            record.size_bytes = size_bytes
            record.mtime = mtime
            record.chunk_count = chunk_count
            record.indexed_at = utcnow()
            session.commit()
            session.refresh(record)
            return record

    def list_all(self) -> list[DocumentRecord]:
        with self._sf() as session:  # type: ignore[operator]
            return list(
                session.execute(select(DocumentRecord).order_by(DocumentRecord.path)).scalars()
            )

    def delete(self, path: str) -> bool:
        with self._sf() as session:  # type: ignore[operator]
            result = session.execute(delete(DocumentRecord).where(DocumentRecord.path == path))
            session.commit()
            return bool(result.rowcount)


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex
