"""RAG integration: index -> search -> grounded answer with clean citations."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.config.schema import Config


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A small document tree with clearly distinct topics."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text(
        "# Deployment\n\n"
        "Deploys run through the `shipd` service. Rollback is automatic when "
        "the health check fails within 90 seconds of a release.\n",
        encoding="utf-8",
    )
    (docs / "network.md").write_text(
        "# Networking\n\n"
        "The load balancer listens on port 8443 and forwards to the app pool "
        "on port 8080 using HTTP/2.\n",
        encoding="utf-8",
    )
    return docs


def test_index_then_search_returns_matching_passages(services, corpus: Path) -> None:
    outcome = services.executor.execute("index_documents", {"path": str(corpus)})

    assert outcome.status == "success"
    assert outcome.result.data["files_indexed"] == 2

    hits = services.retrieval.search("How does rollback work after a deploy?", k=2)
    assert hits
    assert any("rollback" in hit.text.lower() for hit in hits)
    assert all(hit.source.startswith(str(corpus)) for hit in hits)


def test_reindex_is_idempotent(services, corpus: Path) -> None:
    services.executor.execute("index_documents", {"path": str(corpus)})
    before = services.vector_store.count()

    services.executor.execute("index_documents", {"path": str(corpus)})

    assert services.vector_store.count() == before
    assert len(services.documents.list_all()) == 2


def test_indexing_denied_path_is_refused(services, base_config: Config) -> None:
    outcome = services.executor.execute("index_documents", {"path": "/etc/shadow"})

    assert outcome.status == "denied"
    assert services.vector_store.count() == 0


def test_agent_cites_only_retrieved_sources(services, scripted, decision, corpus: Path) -> None:
    services.executor.execute("index_documents", {"path": str(corpus)})
    deploy_doc = str(corpus / "deploy.md")
    scripted(
        [
            decision("search_documents", {"query": "rollback after deployment"}),
            decision(None),
            (
                "Rollback is automatic when the health check fails. "
                f"[source: {deploy_doc}]\n"
                "Also see the invented notes [source: /etc/arthur/secret.md]."
            ),
        ]
    )

    result = services.agent.run("How do rollbacks work?")

    assert result.stopped_reason == "answered"
    assert deploy_doc in result.sources
    assert deploy_doc in result.citations
    # Fabricated citation was stripped from the answer and reported.
    assert "/etc/arthur/secret.md" in result.removed_citations
    assert "/etc/arthur/secret.md" not in result.answer


def test_empty_index_yields_no_passages(services) -> None:
    assert services.retrieval.search("anything at all") == []


def test_indexed_sources_appear_in_system_prompt(
    services, scripted, decision, corpus: Path
) -> None:
    services.executor.execute("index_documents", {"path": str(corpus)})
    llm = scripted([decision(None), "grounded answer"])

    services.agent.run("What is indexed?")

    prompt = llm.calls[0][0].content
    assert "Indexed documents" in prompt
    assert str(corpus / "deploy.md") in prompt
