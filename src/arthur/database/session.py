"""Engine/session management for Arthur's SQLite database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event
from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def rowcount(result: Result[Any]) -> int:
    """Rows affected by a DML statement.

    ``rowcount`` lives on :class:`CursorResult` (what DML actually returns)
    but ``Session.execute`` is typed as the more general ``Result``.
    """
    if isinstance(result, CursorResult):
        return int(result.rowcount or 0)
    return 0


SessionFactory = sessionmaker[Session]


def create_engine(database: str | Path, *, echo: bool = False) -> Engine:
    """Create a SQLite engine with sensible pragmas.

    Args:
        database: Filesystem path, or ``":memory:"`` for an in-memory database
            (used by tests).
        echo: Forwarded to SQLAlchemy for SQL echo debugging.
    """
    db_str = str(database)
    if db_str == ":memory:":
        engine = sa_create_engine(
            "sqlite:///:memory:",
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        return engine

    path = Path(db_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = sa_create_engine(
        f"sqlite+pysqlite:///{path}",
        echo=echo,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _configure_connection(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> SessionFactory:
    """Bind a session factory to an engine (non-expiring for UI friendliness)."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables if they do not exist."""
    from arthur.database.models import Base

    Base.metadata.create_all(engine)
