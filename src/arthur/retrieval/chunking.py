"""Paragraph-aware text chunking with overlap.

Fixed-size chunking naively cuts words in half; this splitter prefers
paragraph boundaries, hard-splits oversized paragraphs at word boundaries and
keeps ``overlap`` characters between hard-split pieces so statements spanning
a boundary remain retrievable.
"""

from __future__ import annotations

import re

_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` characters.

    Args:
        text: Raw document text.
        size: Maximum chunk length in characters.
        overlap: Characters of context repeated between hard-split chunks.

    Returns:
        Ordered list of non-empty chunks (empty list for blank input).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in _PARAGRAPH_RE.split(cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(paragraph, size, overlap))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            cut = window.rfind(" ")
            if cut > int(size * 0.6):
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces
