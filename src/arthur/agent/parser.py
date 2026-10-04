"""Parsing of the model's JSON decision protocol.

Small models routinely wrap JSON in code fences or add stray prose, so the
parser is deliberately forgiving about *where* the JSON lives, but strict
about its structure (validated with Pydantic). Unparseable output raises
:class:`DecisionParseError`, which the orchestrator feeds back to the model as
a repair signal.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class DecisionParseError(Exception):
    """The model output did not contain a valid decision object."""


class Decision(BaseModel):
    """One agent decision step."""

    thought: str = ""
    plan: list[str] = Field(default_factory=list)
    tool: str | None = None
    args: dict[str, Any] | None = None
    answer: str | None = None


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _extract_json_object(text: str) -> str:
    """Return the first balanced top-level ``{...}`` object in ``text``."""
    start = text.find("{")
    if start == -1:
        raise DecisionParseError("no JSON object found in model output")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise DecisionParseError("unbalanced JSON object in model output")


def decision_issue(decision: Decision) -> str | None:
    """Semantic consistency problems that structural parsing cannot catch.

    Small models sometimes emit a plan of concrete steps but no tool and no
    answer, which carries no usable instruction. Reported as a repair signal
    rather than silently accepted.

    Args:
        decision: A structurally valid decision.

    Returns:
        A message to feed back to the model, or ``None`` when consistent.
    """
    if decision.tool is None and decision.answer is None and decision.plan:
        return (
            "your decision listed a plan but chose no tool and gave no answer; "
            "either call exactly one tool to carry out the plan, or drop the plan "
            "and answer directly"
        )
    if decision.tool is None and decision.args:
        return '"args" was set while "tool" was null; args are only valid together with a tool'
    if decision.tool is not None and decision.answer is not None:
        return (
            "both a tool and an answer were provided; pick exactly one - call the "
            "tool first, the answer comes in a later step"
        )
    return None


def parse_decision(raw: str) -> Decision:
    """Parse model output into a :class:`Decision`.

    Raises:
        DecisionParseError: output contains no structurally valid decision.
    """
    if not raw or not raw.strip():
        raise DecisionParseError("empty model output")
    candidate = _strip_fences(raw)
    parsed: object | None = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_extract_json_object(candidate))
        except (json.JSONDecodeError, DecisionParseError) as exc:
            raise DecisionParseError(
                f"no valid JSON object in output: {candidate[:120]!r}"
            ) from exc
    if not isinstance(parsed, dict):
        raise DecisionParseError("decision JSON must be an object")
    try:
        decision = Decision.model_validate(parsed)
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:3]
        )
        raise DecisionParseError(f"invalid decision fields: {issues}") from exc
    if decision.tool is not None and not isinstance(decision.tool, str):
        raise DecisionParseError("tool must be a string or null")
    if decision.tool is not None and not decision.tool.strip():
        decision.tool = None
    return decision
