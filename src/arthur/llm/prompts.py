"""Prompt construction for the agent's decision and answer phases.

Design choice: Arthur uses a JSON decision protocol instead of provider-specific
tool-calling APIs. That keeps the agent model-agnostic (any instruct model served
by Ollama works) and makes every decision parseable, testable and auditable. The
trade-off is one extra generation call for the final answer, which also enables
streaming output to the user.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arthur.config.schema import Config
from arthur.memory.store import MemoryItem

MAX_INDEXED_SOURCES = 40
MAX_MEMORIES = 6

#: Sentinel for "JSON-schema property had no default".
_MISSING = object()


def _type_of(spec: Any) -> str:
    """Human-readable type for one JSON-schema property (defensive)."""
    if not isinstance(spec, dict):
        return "any"
    if "enum" in spec and isinstance(spec["enum"], list):
        return "|".join(str(value) for value in spec["enum"])
    kind = spec.get("type")
    if isinstance(kind, list):  # e.g. ["string", "null"]
        kind = next((k for k in kind if k != "null"), None)
    if kind == "array":
        return f"{_type_of(spec.get('items', {}))}[]"
    if kind in {"string", "integer", "number", "boolean", "object"}:
        return str(kind)
    if "anyOf" in spec and isinstance(spec["anyOf"], list):  # Optional[X]
        inner = [o for o in spec["anyOf"] if not (isinstance(o, dict) and o.get("type") == "null")]
        return "|".join(_type_of(option) for option in inner) or "any"
    if "properties" in spec:
        return "object"
    return "any"


def _signature(schema: dict[str, Any]) -> str:
    """``(path: string, overwrite?: boolean=false)`` for one tool's parameters."""
    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    if not isinstance(properties, dict) or not properties:
        return "()"
    required = set(params.get("required") or [])
    parts: list[str] = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {}
        marker = "" if name in required else "?"
        default = spec.get("default", _MISSING)
        rendered = f"{name}{marker}: {_type_of(spec)}"
        if default is not _MISSING:
            rendered += f"={json.dumps(default, ensure_ascii=False)}"
        parts.append(rendered)
    return "(" + ", ".join(parts) + ")"


def render_tools(tools_schema: Sequence[dict[str, Any]]) -> str:
    """Render tools as one signature line each instead of raw JSON schemas.

    Prompt real estate is budgeted: local models run with a bounded context
    window and slow prompt evaluation, and small models follow a compact
    ``name(args) - description`` menu far more reliably than a multi-kilobyte
    JSON blob (they tend to ignore it and invent tools).
    """
    lines = []
    for schema in tools_schema:
        description = " ".join(str(schema.get("description", "")).split())
        lines.append(f"- {schema.get('name', '?')}{_signature(schema)}: {description}")
    return "\n".join(lines)


def build_system_prompt(
    config: Config,
    tools_schema: Sequence[dict[str, Any]],
    *,
    memories: Sequence[MemoryItem] = (),
    indexed_sources: Sequence[str] = (),
    cwd: Path | None = None,
) -> str:
    """Build the decision-phase system prompt."""
    today = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    workdir = str(cwd or Path.cwd())

    sections: list[str] = [
        "You are Arthur, a local-first AI engineering assistant running on the user's "
        "Linux machine.",
        f"Date: {today}. Working directory: {workdir}. Everything happens locally; "
        "do not suggest cloud APIs.",
        (
            "You act ONLY by selecting registered tools. You never generate shell commands, "
            "never invent tools, and never claim to have run something you did not run."
        ),
    ]

    if config.security.default_shell_access:
        sections.append(
            "A small allow-list of read-only Linux commands is available through run_command; "
            "anything else will be denied."
        )
    else:
        sections.append(
            "Shell access is disabled. Use the file/system/git/document tools instead "
            "of shell commands."
        )

    sections.append(
        """
## Decision protocol
At each step, respond with EXACTLY ONE JSON object and nothing else - no prose, no
markdown, no code fences:
{
  "thought": "<your brief reasoning>",
  "plan": ["<optional: steps you intend to follow>"],
  "tool": "<one tool name from the list below, or null>",
  "args": {<arguments matching that tool's schema, or null>},
  "answer": null
}
Rules:
- Arguments must match the shown signature. A parameter marked ``?:`` is optional
  (the ``=value`` after it is its default); every unmarked parameter is required.
- If you need information, pick exactly one tool and set "args".
- If you already know enough to answer, set "tool": null, "args": null, "answer": null.
  You will then be asked to write the final answer in plain text.
- Observations from tools arrive in later messages; read them carefully before deciding.
- "tool": null is ONLY for greetings or general knowledge. Any request to show, list,
  check, read, find or report anything about this machine or the user's files REQUIRES
  a tool call on this turn - pick the closest registered tool instead of answering
  from prior knowledge. Never fabricate command output.
- Prefer absolute paths under the user's home directory; relative paths resolve
  against the working directory.
- Some operations require user confirmation; the system handles that - never try to bypass it.
- If a tool reports PERMISSION DENIED or INVALID ARGUMENTS, adapt instead of
  repeating the same call.
"""
    )

    if memories:
        lines = "\n".join(f"- [{m.category}] {m.content}" for m in memories[:MAX_MEMORIES])
        sections.append(f"## Known memories (from previous conversations)\n{lines}")

    if indexed_sources:
        shown = sorted(indexed_sources)[:MAX_INDEXED_SOURCES]
        listing = "\n".join(f"- {source}" for source in shown)
        more = (
            ""
            if len(indexed_sources) <= MAX_INDEXED_SOURCES
            else f"\n- ...and {len(indexed_sources) - MAX_INDEXED_SOURCES} more"
        )
        sections.append(
            "## Indexed documents\nWhen you answer from these, copy the exact file "
            "path into a citation marker like [source: <that path>]:\n" + listing + more
        )

    sections.append("## Registered tools\n" + render_tools(tools_schema))
    sections.append(
        """## Examples
User: Show me my system information.
{"thought": "The user wants host details; system_info provides them.", "plan": [],
 "tool": "system_info", "args": {}, "answer": null}

User: What files are in ~/Projects?
{"thought": "Need the directory listing first.", "plan": ["list"],
 "tool": "list_files", "args": {"directory": "/home/USER/Projects"}, "answer": null}

User: Read the first 40 lines of NOTES.md in the working directory.
{"thought": "Read the file with a line limit.", "plan": [],
 "tool": "read_file", "args": {"path": "/home/USER/NOTES.md", "limit": 40},
 "answer": null}

User: Thanks, that is all.
{"thought": "A greeting needs no tool.", "plan": [], "tool": null, "args": null,
 "answer": "You're welcome."}
"""
    )
    return "\n".join(sections)


def build_answer_system_prompt(config: Config) -> str:
    """Build the system prompt for the final (streamed) answer phase."""
    return "\n".join(
        [
            "You are Arthur, a local-first AI engineering assistant running on the "
            "user's Linux machine.",
            "Write the final answer to the user's request based strictly on the conversation "
            "and tool observations provided.",
            "Rules:",
            "- Plain text only: no JSON, no tool calls, no code fences around the whole answer.",
            "  Markdown formatting inside the answer is fine.",
            "- Never output raw JSON or copy an observation verbatim. Restate the facts",
            "  from the observations as natural prose or a short bullet list, so the",
            "  user reads an answer rather than a data dump.",
            "- Be concise and concrete. Show real numbers, paths and results from observations.",
            "- Transcribe numbers exactly as the observation labels them: never swap",
            "  labels (free is not used), never scale them (KB vs GB), never invent",
            "  figures. Quote the observation's own sentence if a figure is ambiguous.",
            "- If a tool was denied, declined, or failed, say so honestly; never "
            "pretend it worked.",
            "- Cite documents by copying the exact path of a passage that actually "
            "appeared in the observations into a marker: [source: followed by the "
            "path, then ]. Never invent document paths and never write a placeholder "
            "path.",
            "- If asked to perform a destructive operation, restate exactly what would change "
            "and note that confirmation is required.",
        ]
    )
