"""The tool executor: validate -> permission check -> confirm -> run -> audit.

The executor never raises for policy outcomes; every invocation produces an
:class:`ExecutionOutcome` with a structured status, and every invocation (even
a denial) is written to the audit trail.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from pydantic import ValidationError

from arthur.security.confirmation import ConfirmationCallback, ConfirmationRequest, approve, deny
from arthur.security.permissions import PermissionLevel
from arthur.security.policy import SecurityPolicy
from arthur.tools.base import Tool, ToolContext, ToolResult
from arthur.tools.registry import ToolRegistry
from arthur.execution.types import AuditSink, ExecutionOutcome, ExecutionRecord

ConfirmationMode = ConfirmationCallback | bool | None


class ToolExecutor:
    """Executes registered tools under the security policy with auditing."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: SecurityPolicy,
        ctx: ToolContext,
        *,
        audit: AuditSink | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.ctx = ctx
        self.audit = audit

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        request_id: str | None = None,
        request_text: str | None = None,
        confirm: ConfirmationMode = None,
    ) -> ExecutionOutcome:
        """Run one tool invocation through the full security pipeline.

        Args:
            tool_name: Registered tool name.
            arguments: Raw arguments (validated against the tool's schema).
            request_id: Correlates audit records across one user request.
            request_text: The user request that motivated this call (audit).
            confirm: Confirmation behaviour - a callback (interactive UI),
                ``True`` (approve all), ``False`` (decline all), or ``None``
                (leave confirmation unresolved and report it as pending).
        """
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()

        def finish(
            status: str,
            *,
            permission: PermissionLevel,
            confirmed: bool = False,
            result: ToolResult | None = None,
            error: str | None = None,
            request: ConfirmationRequest | None = None,
            args: dict[str, Any] | None = None,
        ) -> ExecutionOutcome:
            duration = (time.perf_counter() - started) * 1000
            record = ExecutionRecord(
                request_id=request_id,
                request=(request_text or None) and request_text[:500],
                tool_name=tool_name,
                arguments=args if args is not None else dict(arguments or {}),
                permission=permission,
                confirmed=confirmed,
                status=status,  # type: ignore[arg-type]
                result_summary=result.summary if result else None,
                error=error,
                duration_ms=duration,
            )
            if self.audit is not None:
                try:
                    self.audit(record)
                except Exception:  # noqa: BLE001 - auditing must not break execution
                    pass
            return ExecutionOutcome(record=record, result=result, request=request)

        # 1. Tool lookup -----------------------------------------------------
        tool = self.registry.get(tool_name)
        if tool is None:
            return finish(
                "unknown_tool",
                permission=PermissionLevel.DENIED,
                error=f"unknown tool {tool_name!r}; available: "
                f"{', '.join(self.registry.names())}",
            )

        # 2. Argument validation --------------------------------------------
        try:
            validated = tool.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()[:5]
            )
            return finish(
                "invalid_args",
                permission=PermissionLevel.DENIED,
                error=f"invalid arguments for {tool_name}: {details}",
            )
        args_dict = validated.model_dump()

        # 3. Security policy -------------------------------------------------
        try:
            decision = self.policy.evaluate(tool, validated, args_dict)
        except Exception as exc:  # noqa: BLE001 - policy errors fail closed
            return finish(
                "denied",
                permission=PermissionLevel.DENIED,
                error=f"policy evaluation failed (denied): {exc}",
                args=args_dict,
            )
        if not decision.allowed:
            return finish(
                "denied",
                permission=decision.level,
                error=decision.reason,
                args=args_dict,
            )

        # 4. Confirmation ----------------------------------------------------
        confirmed = False
        if decision.requires_confirmation:
            confirmation = ConfirmationRequest(
                tool_name=tool.name,
                action=tool.action or tool.description,
                target=self._target_of(args_dict),
                args=args_dict,
                reason=decision.reason,
                risk=tool.risk,
            )
            handler = self._confirm_handler(confirm)
            if handler is None:
                return finish(
                    "confirmation_required",
                    permission=decision.level,
                    request=confirmation,
                    args=args_dict,
                )
            if not handler(confirmation):
                return finish(
                    "declined",
                    permission=decision.level,
                    error="user declined the operation",
                    request=confirmation,
                    args=args_dict,
                )
            confirmed = True

        # 5. Execution -------------------------------------------------------
        try:
            result = self._run_with_timeout(tool, validated)
        except FutureTimeoutError:
            return finish(
                "timeout",
                permission=decision.level,
                confirmed=confirmed,
                error=f"{tool_name} exceeded its {tool.timeout:.0f}s time limit",
                args=args_dict,
            )
        except Exception as exc:  # noqa: BLE001 - tools fail with structured errors
            return finish(
                "error",
                permission=decision.level,
                confirmed=confirmed,
                error=f"{type(exc).__name__}: {exc}",
                args=args_dict,
            )

        if tool.result_model is not None and result.data is not None:
            try:
                result.data = tool.result_model.model_validate(result.data).model_dump()
            except ValidationError as exc:
                return finish(
                    "error",
                    permission=decision.level,
                    confirmed=confirmed,
                    error=f"tool returned an invalid result schema: {exc.errors()[:3]}",
                    args=args_dict,
                )

        return finish(
            "success" if result.ok else "error",
            permission=decision.level,
            confirmed=confirmed,
            result=result,
            error=None if result.ok else result.summary,
            args=args_dict,
        )

    @staticmethod
    def _target_of(args_dict: dict[str, Any]) -> str:
        for key in ("path", "target", "source", "destination", "directory", "root", "command"):
            value = args_dict.get(key)
            if isinstance(value, str) and value:
                return value
        return "(see arguments)"

    @staticmethod
    def _confirm_handler(confirm: ConfirmationMode) -> ConfirmationCallback | None:
        if confirm is None:
            return None
        if confirm is True:
            return approve
        if confirm is False:
            return deny
        return confirm

    def _run_with_timeout(self, tool: Tool, validated: Any) -> ToolResult:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{tool.name}")
        try:
            future = pool.submit(tool.run, validated, self.ctx)
            return future.result(timeout=tool.timeout)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
