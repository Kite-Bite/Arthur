"""Agent loop integration: tools, repairs, limits, confirmation, persistence.

These tests drive :class:`arthur.agent.orchestrator.Agent` through the full
``Services`` graph (real executor, real security policy, real SQLite audit
trail) with only the LLM replaced by the scripted double.
"""

from __future__ import annotations

import json
from pathlib import Path

from arthur.llm.base import LLMError
from arthur.llm.testing import ScriptedLLM


def test_multi_step_tool_run_reaches_answer(services, scripted, decision, tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    scripted(
        [
            decision("list_files", {"directory": str(tmp_path)}),
            decision(None),
            "There is one file: alpha.txt",
        ]
    )

    result = services.agent.run("What files are in my scratch dir?")

    assert result.stopped_reason == "answered"
    assert result.used_tools == ["list_files"]
    assert result.steps[0].status == "success"
    assert result.steps[0].permission == "SAFE"
    assert "alpha.txt" in result.answer
    # The audit trail recorded the tool call made during the run.
    assert services.executions.recent(10)[0].tool_name == "list_files"


def test_denied_tool_is_reported_and_loop_adapts(services, scripted, decision) -> None:
    scripted(
        [
            decision("read_file", {"path": "/etc/shadow"}),
            decision(None),
            "I could not read that file; access was denied.",
        ]
    )

    result = services.agent.run("Read /etc/shadow for me")

    assert result.stopped_reason == "answered"
    assert result.steps[0].status == "denied"
    assert "PERMISSION DENIED" in result.steps[0].observation


def test_invalid_model_output_is_repaired(services, scripted, decision) -> None:
    llm = scripted(["this is not JSON", decision(None), "All good."])

    result = services.agent.run("Say hello")

    assert result.stopped_reason == "answered"
    assert result.answer == "All good."
    assert llm.call_count == 3
    # The repair round-trip fed a corrective instruction back to the model.
    repair_messages = llm.calls[1]
    assert any("INVALID ACTION" in m.content for m in repair_messages)


def test_persistent_parse_failure_degrades_gracefully(services, scripted) -> None:
    scripted(["nonsense", "more nonsense", "still broken", "Best effort answer."])

    result = services.agent.run("Do something impossible")

    assert result.stopped_reason == "parse_failure"
    assert result.answer == "Best effort answer."


def test_step_budget_stops_loop_then_answers(services, scripted, decision) -> None:
    services.config.agent.max_steps = 2
    scripted(
        [
            decision("system_info", {}),
            decision("uptime", {}),
            "Summary from the observations.",
        ]
    )

    result = services.agent.run("Describe this machine")

    assert result.stopped_reason == "max_steps"
    assert result.used_tools == ["system_info", "uptime"]
    assert len(result.steps) == 2
    assert result.answer == "Summary from the observations."


def test_confirmation_pending_stops_before_execution(
    services, scripted, decision, tmp_path: Path
) -> None:
    target = tmp_path / "doomed.txt"
    target.write_text("still here", encoding="utf-8")
    scripted([decision("delete_file", {"path": str(target)})])

    result = services.agent.run(f"Delete {target}")

    assert result.stopped_reason == "confirmation_required"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "delete_file"
    assert target.exists()


def test_confirmation_approved_executes_operation(
    services, scripted, decision, tmp_path: Path
) -> None:
    target = tmp_path / "doomed.txt"
    target.write_text("bye", encoding="utf-8")
    scripted([decision("delete_file", {"path": str(target)}), decision(None), "Deleted."])

    result = services.agent.run(f"Delete {target}", confirm=True)

    assert result.stopped_reason == "answered"
    assert result.steps[0].status == "success"
    assert not target.exists()
    rows = [r for r in services.executions.recent(10) if r.tool_name == "delete_file"]
    assert rows and rows[0].confirmed is True


def test_confirmation_declined_leaves_state_intact(
    services, scripted, decision, tmp_path: Path
) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("safe", encoding="utf-8")
    scripted([decision("delete_file", {"path": str(target)}), decision(None), "Not deleted."])

    result = services.agent.run(f"Delete {target}", confirm=False)

    assert result.stopped_reason == "answered"
    assert result.steps[0].status == "declined"
    assert target.exists()


def test_llm_failure_produces_honest_error_answer(services, scripted) -> None:
    scripted(error=LLMError("connection refused by ollama"))

    result = services.agent.run("Anything")

    assert result.stopped_reason == "llm_error"
    assert "connection refused by ollama" in result.answer
    assert result.error is not None


def test_conversation_history_is_replayed(services, scripted, decision) -> None:
    scripted([decision(None), "Paris is the capital of France."])
    first = services.agent.run("What is the capital of France?")
    assert first.conversation_id

    llm = scripted([decision(None), "Paris."])
    services.agent.run("And its population?", conversation_id=first.conversation_id)

    contents = [m.content for m in llm.calls[0]]
    assert any("capital of France" in c for c in contents)
    assert any("Paris is the capital" in c for c in contents)
    history = services.conversations.history(first.conversation_id)
    roles = [role for role, _ in history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_relevant_memories_are_injected_into_prompt(services, scripted, decision) -> None:
    services.memory.add("Our deploy target is the staging-eu1 cluster", category="project")
    llm = scripted([decision(None), "staging-eu1"])

    services.agent.run("Which staging cluster do we deploy to?")

    system_prompt = llm.calls[0][0].content
    assert "Known memories" in system_prompt
    assert "staging-eu1" in system_prompt


def test_memory_injection_can_be_disabled(services, scripted, decision) -> None:
    services.config.agent.memory_injection = False
    services.memory.add("Our deploy target is the staging-eu1 cluster", category="project")
    llm = scripted([decision(None), "unknown"])

    services.agent.run("Which staging cluster do we deploy to?")

    assert "Known memories" not in llm.calls[0][0].content


def test_step_callback_streams_progress(services, scripted, decision) -> None:
    scripted([decision("uptime", {}), decision(None), "Up for a while."])
    seen: list[str] = []

    services.agent.run(
        "How long has this box been up?", on_step=lambda s: seen.append(s.tool or "")
    )

    assert seen == ["uptime"]


def test_decision_args_are_validated_against_schema(services, scripted, decision) -> None:
    # "limit" must be 1..20 for search_documents; 999 fails validation.
    bad_args = json.dumps(
        {
            "thought": "search",
            "plan": [],
            "tool": "search_documents",
            "args": {"query": "x", "limit": 999},
        }
    )
    scripted([bad_args, decision(None), "No results."])

    result = services.agent.run("Search the docs")

    assert result.steps[0].status == "invalid_args"
    assert result.stopped_reason == "answered"


def test_scripted_llm_is_deterministic_double() -> None:
    llm = ScriptedLLM(["one", "two"])
    assert llm.complete([]) == "one"
    assert llm.call_count == 1
    assert "".join(llm.stream([])) == "two "
