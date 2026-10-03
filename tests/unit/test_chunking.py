"""Chunking behaviour: sizes, overlap, paragraph awareness."""

from __future__ import annotations

import pytest

from arthur.retrieval.chunking import chunk_text


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_single_chunk() -> None:
    text = "Hello world."
    chunks = chunk_text(text, size=100, overlap=10)
    assert chunks == ["Hello world."]


def test_paragraphs_stay_intact_when_possible() -> None:
    paragraphs = ["First paragraph about Linux.", "Second paragraph about Btrfs."]
    chunks = chunk_text("\n\n".join(paragraphs), size=80, overlap=10)
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0] and "Second paragraph" in chunks[0]


def test_chunk_size_respected_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=120, overlap=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    # Overlap: the tail of one chunk should appear at the head of the next.
    assert any(chunks[i][-10:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_oversized_paragraph_is_hard_split() -> None:
    text = "x" * 1000 + " " + "y" * 100
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size=0)
    with pytest.raises(ValueError):
        chunk_text("text", size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("text", size=100, overlap=-1)
