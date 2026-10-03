"""Text extraction and file discovery for document ingestion.

The ingestion layer is deliberately pluggable: add an extension -> extractor
mapping to support new formats. PDF support is optional (``arthur[pdf]``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


class UnsupportedFormatError(Exception):
    """The file extension is not in the configured format list."""


class ExtractionError(Exception):
    """The file exists but its text could not be extracted."""


def extract_text(path: Path, *, formats: set[str], max_bytes: int = 2_000_000) -> tuple[str, str]:
    """Extract text from ``path``.

    Args:
        path: File to read.
        formats: Allowed extensions (without dots), lowercased.
        max_bytes: Cap for text files (PDF page text is not byte-capped).

    Returns:
        ``(text, extension)``

    Raises:
        UnsupportedFormatError: extension not configured.
        ExtractionError: unreadable, binary, or PDF without pypdf.
    """
    ext = path.suffix.lstrip(".").lower()
    if ext not in formats:
        raise UnsupportedFormatError(
            f".{ext or '?'} files are not configured for indexing (see retrieval.formats)"
        )
    if not path.is_file():
        raise ExtractionError(f"not a regular file: {path}")

    if ext == "pdf":
        return _extract_pdf(path), ext

    try:
        raw = path.read_bytes()[: max_bytes + 1]
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    if b"\x00" in raw[:8192]:
        raise ExtractionError(f"{path} appears to be binary")
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n[file truncated during ingestion]"
    return text, ext


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError(
            "PDF support requires pypdf; install with: uv sync --extra pdf"
        ) from exc
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages[:200]]
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise ExtractionError(f"cannot parse PDF {path}: {exc}") from exc
    return "\n\n".join(pages)


def discover_files(
    root: Path,
    *,
    formats: set[str],
    recursive: bool = True,
    denied_dir_names: set[str] | None = None,
) -> Iterator[Path]:
    """Yield indexable files under ``root``, pruning denied directories."""
    denied = denied_dir_names or set()
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in denied]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lstrip(".").lower() in formats:
                    yield path
    else:
        for path in sorted(root.iterdir()):
            if (
                path.is_file()
                and path.suffix.lstrip(".").lower() in formats
                and path.name not in denied
            ):
                yield path
