"""SQLAlchemy models for Arthur's local SQLite database.

Tables:

* ``conversations`` / ``messages`` - chat history (short-term memory)
* ``memory_records``               - explicit long-term memory
* ``tool_executions``              - audit trail of tool activity
* ``documents`` / ``document_chunks`` - RAG metadata and the built-in
  vector backend (unused when the ChromaDB backend is selected)
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite-friendly, consistent across the app)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base for all Arthur models."""


class Conversation(Base):
    """A conversation thread with its ordered messages."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    """A single chat message belonging to a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class MemoryRecord(Base):
    """An explicit long-term memory entry."""

    __tablename__ = "memory_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), default="general", index=True)
    source: Mapped[str] = mapped_column(String(128), default="user")
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ToolExecution(Base):
    """Audit record for one tool invocation."""

    __tablename__ = "tool_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(64), index=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    permission: Mapped[str] = mapped_column(String(16), default="SAFE")
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(24), default="success", index=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)


class DocumentRecord(Base):
    """Metadata for an indexed document."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(400), default="")
    format: Mapped[str] = mapped_column(String(16), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    mtime: Mapped[float] = mapped_column(Float, default=0.0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DocumentChunk(Base):
    """A text chunk with its embedding (built-in SQLite vector backend)."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(Text, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    dimensions: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
