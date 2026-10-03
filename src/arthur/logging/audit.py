"""Audit trail: every tool execution is logged and optionally persisted.

The audit logger is a plain callable so the executor stays decoupled from the
database: tests can pass a list-appending function, production wires the
:class:`AuditLogger` with a SQLAlchemy session factory.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from arthur.database.models import ToolExecution
from arthur.execution.types import ExecutionRecord
from arthur.security.permissions import PermissionLevel

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger("arthur.audit")


class AuditLogger:
    """Structured audit sink: emits a log line and stores the record in SQLite.

    Database failures are logged but never raised - auditing must not break
    the operation it is observing, and the log file remains a fallback trail.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory

    def __call__(self, record: ExecutionRecord) -> None:
        self.emit(record)

    def emit(self, record: ExecutionRecord) -> None:
        payload = record.model_dump(mode="json")
        logger.info(
            "tool=%s status=%s decision=%s confirmed=%s duration_ms=%.1f",
            record.tool_name,
            record.status,
            record.permission,
            record.confirmed,
            record.duration_ms,
            extra={"audit": payload},
        )
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as session:
                session.add(
                    ToolExecution(
                        request_id=record.request_id,
                        request=record.request,
                        tool_name=record.tool_name,
                        arguments=record.arguments,
                        permission=PermissionLevel(record.permission).name,
                        confirmed=record.confirmed,
                        status=record.status,
                        result_summary=record.result_summary,
                        error=record.error,
                        duration_ms=record.duration_ms,
                    )
                )
                session.commit()
        except Exception as exc:  # noqa: BLE001 - audit must never crash the caller
            logger.error("Failed to persist audit record: %s", exc)
            logger.debug("Audit payload: %s", json.dumps(payload, default=str))
