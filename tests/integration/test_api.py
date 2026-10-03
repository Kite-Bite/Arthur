"""FastAPI integration: auth, chat confirm flow, memory, documents, logs."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arthur.api.app import create_app
from arthur.services import Services


@pytest.fixture()
def client(services: Services):
    app = create_app(services)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def authed(services: Services):
    """Same graph, but with bearer auth enabled."""
    services.config.api.token = "test-token"
    app = create_app(services)
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_public_and_reports_counts(client: TestClient, services: Services) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["tools"] == len(services.registry)
    assert "version" in body


def test_bearer_auth_required_when_token_configured(authed: TestClient) -> None:
    assert authed.get("/tools").status_code == 401
    assert authed.get("/tools", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok = authed.get("/tools", headers={"Authorization": "Bearer test-token"})
    assert ok.status_code == 200
    names = [tool["name"] for tool in ok.json()]
    assert "read_file" in names and "run_command" in names

    # Health stays unauthenticated for monitoring.
    assert authed.get("/health").status_code == 200


def test_tool_listing_includes_schemas(client: TestClient) -> None:
    tools = client.get("/tools").json()
    read_file = next(t for t in tools if t["name"] == "read_file")
    assert read_file["permission"] in {"SAFE", "CONFIRM", "RESTRICTED", "DENIED"}
    assert "path" in read_file["parameters"]["properties"]
    run_command = next(t for t in tools if t["name"] == "run_command")
    assert run_command["permission"] == "CONFIRM"


def test_tool_execution_requires_confirmation_then_approves(client: TestClient) -> None:
    pending = client.post(
        "/tools/write_file/execute",
        json={"arguments": {"path": "/nonexistent-allowed-root/x.txt", "content": "hi"}},
    )
    # Path outside the sandbox is refused regardless of confirmation.
    assert pending.status_code == 200
    assert pending.json()["status"] == "denied"


def test_memory_crud_roundtrip(client: TestClient) -> None:
    created = client.post(
        "/memory",
        json={"content": "The staging cluster is staging-eu1", "category": "infra"},
    )
    assert created.status_code == 201
    memory_id = created.json()["id"]

    listed = client.get("/memory", params={"query": "staging cluster"})
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["memories"][0]["id"] == memory_id

    search_miss = client.get("/memory", params={"query": "unrelated quantum physics"})
    assert search_miss.json()["count"] == 0

    deleted = client.delete(f"/memory/{memory_id}")
    assert deleted.status_code == 204
    assert client.delete(f"/memory/{memory_id}").status_code == 404


def test_memory_validation_rejects_bad_input(client: TestClient) -> None:
    assert client.post("/memory", json={"content": ""}).status_code == 422
    assert client.post("/memory", json={"content": "x", "importance": 3.0}).status_code == 422


def test_documents_index_and_search(client: TestClient, tmp_path: Path) -> None:
    docs = tmp_path / "notes"
    docs.mkdir()
    (docs / "runbook.md").write_text(
        "# Runbook\n\nRestart the ingest worker before draining the queue.\n",
        encoding="utf-8",
    )

    indexed = client.post("/documents/index", json={"path": str(docs)})
    assert indexed.status_code == 200
    assert indexed.json()["summary"].startswith("indexed 1 files")

    found = client.get("/documents/search", params={"q": "restart the ingest worker"})
    assert found.status_code == 200
    assert found.json()["count"] >= 1
    assert "runbook.md" in found.json()["results"][0]["source"]


def test_documents_index_rejects_outside_sandbox(client: TestClient) -> None:
    response = client.post("/documents/index", json={"path": "/etc"})
    assert response.status_code == 400


def test_logs_reflect_executed_tools(client: TestClient, services: Services) -> None:
    services.executor.execute("uptime", {})

    logs = client.get("/logs", params={"limit": 5})

    assert logs.status_code == 200
    assert logs.json()[0]["tool_name"] == "uptime"
    assert logs.json()[0]["status"] == "success"


def test_chat_runs_agent_and_persists_conversation(
    client: TestClient, services: Services, scripted, decision
) -> None:
    scripted([decision(None), "Your uptime is 3 days."])

    response = client.post("/chat", json={"message": "How long has this machine been up?"})

    assert response.status_code == 200
    body = response.json()
    assert body["stopped_reason"] == "answered"
    assert body["answer"] == "Your uptime is 3 days."
    assert body["conversation_id"]

    # Follow-up in the same conversation reuses history.
    scripted([decision(None), "Still 3 days."])
    follow_up = client.post(
        "/chat",
        json={"message": "And now?", "conversation_id": body["conversation_id"]},
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["conversation_id"] == body["conversation_id"]


def test_chat_confirmation_flow(
    client: TestClient, services: Services, scripted, decision, tmp_path: Path
) -> None:
    target = tmp_path / "api-delete.txt"
    target.write_text("delete me", encoding="utf-8")
    scripted([decision("delete_file", {"path": str(target)})])

    pending = client.post("/chat", json={"message": f"Delete {target}"})

    assert pending.status_code == 200
    body = pending.json()
    assert body["stopped_reason"] == "confirmation_required"
    assert body["pending_confirmation"]["tool_name"] == "delete_file"
    assert target.exists()

    scripted([decision("delete_file", {"path": str(target)}), decision(None), "Removed."])
    approved = client.post("/chat", json={"message": f"Delete {target}", "confirm": True})

    assert approved.json()["stopped_reason"] == "answered"
    assert not target.exists()


def test_chat_rejects_empty_message(client: TestClient) -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_returns_503_when_llm_unavailable(client: TestClient, scripted) -> None:
    from arthur.llm.base import LLMError

    scripted(error=LLMError("ollama is not running"))

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert "ollama is not running" in response.json()["detail"]
