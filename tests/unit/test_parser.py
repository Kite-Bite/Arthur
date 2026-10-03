"""Decision parsing: tolerant extraction, strict validation."""

from __future__ import annotations

import pytest

from arthur.agent.parser import Decision, DecisionParseError, parse_decision


def test_clean_json() -> None:
    decision = parse_decision('{"thought": "hi", "tool": "list_files", "args": {"path": "."}}')
    assert decision.tool == "list_files"
    assert decision.args == {"path": "."}


def test_fenced_json() -> None:
    raw = '```json\n{"thought": "x", "tool": null, "args": null}\n```'
    decision = parse_decision(raw)
    assert decision.tool is None


def test_json_with_surrounding_prose() -> None:
    raw = 'Sure! Here is my decision:\n{"thought": "t", "tool": "uptime", "args": {}}\nDone.'
    decision = parse_decision(raw)
    assert decision.tool == "uptime"


def test_null_tool_means_answer() -> None:
    decision = parse_decision('{"thought": "enough", "tool": null, "args": null}')
    assert decision.tool is None


def test_empty_tool_string_normalised_to_none() -> None:
    decision = parse_decision('{"tool": "", "args": null}')
    assert decision.tool is None


def test_plan_and_answer_fields() -> None:
    decision = parse_decision(
        '{"plan": ["a", "b"], "tool": null, "answer": "text"}'
    )
    assert decision.plan == ["a", "b"]
    assert decision.answer == "text"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "no json here at all",
        '{"tool": ',  # unbalanced
        "[1, 2, 3]",  # not an object
        '{"tool": 42}',  # wrong type (int not str/null)
        '{"args": "not-a-dict"}',
    ],
)
def test_invalid_outputs_raise(raw: str) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision(raw)


def test_decision_defaults() -> None:
    decision = Decision.model_validate({})
    assert decision.tool is None and decision.args is None and decision.plan == []
