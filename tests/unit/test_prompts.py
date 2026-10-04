"""Prompt rendering: tool signature menu, section assembly."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from arthur.config.schema import Config
from arthur.llm.prompts import build_answer_system_prompt, build_system_prompt, render_tools

#: A schema shaped like Pydantic's ``model_json_schema()`` output.
SAMPLE = [
    {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {
            "properties": {
                "path": {"type": "string", "title": "Path", "description": "File to read"},
                "limit": {
                    "type": "integer",
                    "title": "Limit",
                    "default": 400,
                    "description": "Max lines",
                },
            },
            "required": ["path"],
            "title": "ReadFileArgs",
            "type": "object",
        },
    },
    {
        "name": "system_info",
        "description": "Host and kernel details.",
        "parameters": {"properties": {}, "type": "object"},
    },
]


def test_signatures_mark_required_and_optional() -> None:
    rendered = render_tools(SAMPLE)

    assert "- read_file(path: string, limit?: integer=400): Read a text file." in rendered
    assert "- system_info(): Host and kernel details." in rendered


def test_render_tools_is_compact() -> None:
    rendered = render_tools(SAMPLE)

    # No JSON blob: the menu must stay small enough for a local context window.
    assert "{" not in rendered
    assert "title" not in rendered
    assert len(rendered) < 400


def test_signature_renders_enum_and_array_types() -> None:
    schema = {
        "name": "tool",
        "description": "d",
        "parameters": {
            "properties": {
                "mode": {"enum": ["a", "b"], "title": "Mode"},
                "args": {"type": "array", "items": {"type": "string"}, "title": "Args"},
            },
            "required": ["mode"],
            "type": "object",
        },
    }

    rendered = render_tools([schema])

    assert "mode: a|b" in rendered
    assert "args?: string[]" in rendered


def test_system_prompt_contains_protocol_and_tools() -> None:
    prompt = build_system_prompt(Config(), SAMPLE, cwd=Path("/work"))

    assert "Decision protocol" in prompt
    assert "Registered tools" in prompt
    assert "read_file" in prompt
    assert str(Path("/work")) in prompt
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    assert today in prompt


def test_system_prompt_includes_memories_and_sources() -> None:
    from arthur.memory.store import MemoryItem

    memory = MemoryItem(
        id=1,
        content="deploy target is staging-eu1",
        category="project",
        source="test",
        importance=0.5,
        created_at=datetime.now(UTC),
    )

    prompt = build_system_prompt(
        Config(),
        SAMPLE,
        memories=[memory],
        indexed_sources=["/docs/runbook.md"],
    )

    assert "Known memories" in prompt
    assert "staging-eu1" in prompt
    assert "Indexed documents" in prompt
    assert "/docs/runbook.md" in prompt


def test_shell_section_reflects_config() -> None:
    disabled = build_system_prompt(Config(), SAMPLE)
    config = Config()
    config.security.default_shell_access = True
    enabled = build_system_prompt(config, SAMPLE)

    assert "Shell access is disabled" in disabled
    assert "allow-list" in enabled


def test_answer_prompt_is_plain_text_oriented() -> None:
    prompt = build_answer_system_prompt(Config())

    assert "no JSON" in prompt
    assert "[source:" in prompt
