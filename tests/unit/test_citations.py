"""Citation sanitisation: fabricated sources must never survive."""

from __future__ import annotations

from arthur.retrieval.citations import extract_citations, sanitize_citations


def test_extract_citations() -> None:
    text = "Btrfs uses snapshots [source: /home/u/notes/linux.md]. More [source: /a/b.txt]."
    assert extract_citations(text) == ["/home/u/notes/linux.md", "/a/b.txt"]


def test_keeps_valid_citation() -> None:
    text = "Answer here. [source: /home/u/docs/a.md]"
    cleaned, kept, removed = sanitize_citations(text, ["/home/u/docs/a.md"])
    assert kept == ["/home/u/docs/a.md"]
    assert removed == []
    assert "[source: /home/u/docs/a.md]" in cleaned


def test_removes_fabricated_citation() -> None:
    text = "Claim. [source: /home/u/does-not-exist.md]"
    cleaned, kept, removed = sanitize_citations(text, ["/home/u/docs/a.md"])
    assert kept == []
    assert removed == ["/home/u/does-not-exist.md"]
    assert "[source:" not in cleaned


def test_mixed_citations_partial_removal() -> None:
    text = "Good [source: /real/doc.md] and bad [source: /fake/doc.md]."
    cleaned, kept, removed = sanitize_citations(text, ["/real/doc.md"])
    assert kept == ["/real/doc.md"]
    assert removed == ["/fake/doc.md"]
    assert "/real/doc.md" in cleaned and "/fake/doc.md" not in cleaned


def test_tilde_form_matches_absolute_allowed_source(tmp_path) -> None:
    # Model cites the expanded path while retrieval reported a ~ path.
    text = "See [source: /home/someone/docs/x.md]"
    _, kept, removed = sanitize_citations(text, ["/home/someone/docs/x.md"])
    assert kept and not removed


def test_relative_citation_is_removed() -> None:
    text = "See [source: docs/x.md]"
    _, kept, removed = sanitize_citations(text, ["/abs/docs/x.md"])
    assert removed == ["docs/x.md"]
    assert kept == []
