"""Grounded-answer citation control.

The agent prompt requires the model to cite retrieved passages as
``[source: /absolute/path]``. After generation the answer is post-processed:
any citation that does not match an actually retrieved source is removed, so a
hallucinated document reference can never reach the user.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

SOURCE_RE = re.compile(r"\[source:\s*([^\]]+)\]")


def extract_citations(text: str) -> list[str]:
    """Return every cited source string in ``text`` (in order)."""
    return [match.strip() for match in SOURCE_RE.findall(text)]


def sanitize_citations(
    text: str, allowed_sources: Iterable[str]
) -> tuple[str, list[str], list[str]]:
    """Remove citations that are not in ``allowed_sources``.

    Args:
        text: Model-generated answer.
        allowed_sources: Sources actually returned by retrieval this request.

    Returns:
        ``(cleaned_text, kept_sources, removed_sources)``
    """
    allowed_exact: set[str] = set()
    allowed_resolved: set[str] = set()
    for source in allowed_sources:
        allowed_exact.add(source.strip())
        allowed_exact.add(str(Path(source).expanduser()))
        try:
            allowed_resolved.add(str(Path(source).expanduser().resolve()))
        except OSError:  # pragma: no cover - exotic paths
            pass

    kept: list[str] = []
    removed: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        cited = match.group(1).strip().strip("'\"").rstrip(".,;")
        candidates = {cited, str(Path(cited).expanduser())}
        try:
            candidates.add(str(Path(cited).expanduser().resolve()))
        except OSError:  # pragma: no cover
            pass
        if candidates & allowed_exact or candidates & allowed_resolved:
            kept.append(cited)
            return match.group(0)
        removed.append(cited)
        return ""

    cleaned = SOURCE_RE.sub(_replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), kept, removed
