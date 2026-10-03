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


def render_tools(tools_schema: Sequence[dict[str, Any]]) -> str:
    """Render tool schemas as compact JSON for the prompt."""
    return json.dumps(list(tools_schema), indent=2, ensure_ascii=False)


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
        "You are Arthur, a local-first AI engineering assistant running on the user's Linux machine.",
        f"Date: {today}. Working directory: {workdir}. Everything happens locally; do not suggest cloud APIs.",
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
            "Shell access is disabled. Use the file/system/git/document tools instead of shell commands."
        )

    sections.append(
        """
## Decision protocol
At each step, respond with EXACTLY ONE JSON object and nothing else - no prose, no markdown, no code fences:
{
  "thought": "<your brief reasoning>",
  "plan": ["<optional: steps you intend to follow>"],
  "tool": "<one tool name from the list below, or null>",
  "args": {<arguments matching that tool's schema, or null>},
  "answer": null
}
Rules:
- Arguments must match the tool's parameters schema exactly; all required fields are mandatory.
- If you need information, pick exactly one tool and set "args".
- If you already know enough to answer, set "tool": null, "args": null, "answer": null.
  You will then be asked to write the final answer in plain text.
- Observations from tools arrive in later messages; read them carefully before deciding.
- Prefer absolute paths under the user's home directory; relative paths resolve against the working directory.
- Some operations require user confirmation; the system handles that - never try to bypass it.
- If a tool reports PERMISSION DENIED or INVALID ARGUMENTS, adapt instead of repeating the same call.
"""
    )

    if memories:
        lines = "\n".join(
            f"- [{m.category}] {m.content}" for m in memories[:MAX_MEMORIES]
        )
        sections.append(f"## Known memories (from previous conversations)\n{lines}")

    if indexed_sources:
        shown = sorted(indexed_sources)[:MAX_INDEXED_SOURCES]
        listing = "\n".join(f"- {source}" for source in shown)
        more = (
            "" if len(indexed_sources) <= MAX_INDEXED_SOURCES
            else f"\n- ...and {len(indexed_sources) - MAX_INDEXED_SOURCES} more"
        )
        sections.append(
            "## Indexed documents\nWhen you answer from these, cite them verbatim "
            "using the marker [source: path]:\n" + listing + more
        )

    sections.append("## Registered tools\n" + render_tools(tools_schema))
    return "\n".join(sections)


def build_answer_system_prompt(config: Config) -> str:
    """Build the system prompt for the final (streamed) answer phase."""
    return "\n".join(
        [
            "You are Arthur, a local-first AI engineering assistant running on the user's Linux machine.",
            "Write the final answer to the user's request based strictly on the conversation "
            "and tool observations provided.",
            "Rules:",
            "- Plain text only: no JSON, no tool calls, no code fences around the whole answer.",
            "  Markdown formatting inside the answer is fine.",
            "- Be concise and concrete. Show real numbers, paths and results from observations.",
            "- If a tool was denied, declined, or failed, say so honestly; never pretend it worked.",
            "- Cite documents ONLY with markers of the form [source: /absolute/path] for paths "
            "that actually appeared in observations. Never invent document paths.",
            "- If asked to perform a destructive operation, restate exactly what would change "
            "and note that confirmation is required.",
        ]
    )
