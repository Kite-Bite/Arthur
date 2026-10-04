"""PDF ingestion (optional: ``uv sync --extra pdf``).

Both directions are covered: the actionable error when the extra is missing,
and real text extraction when it is installed. One of the two always runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.retrieval.extractors import ExtractionError, extract_text

FORMATS = {"md", "pdf"}

try:  # pragma: no cover - availability differs per machine
    import pypdf  # noqa: F401

    HAS_PYPDF = True
except ImportError:  # pragma: no cover
    HAS_PYPDF = False


def _tiny_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with a correct xref table.

    Written by hand so the test needs no PDF-generation dependency beyond the
    one under test.
    """
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.mark.skipif(HAS_PYPDF, reason="pypdf is installed; the missing-extra path is unreachable")
def test_missing_extra_raises_an_actionable_error(tmp_path: Path) -> None:
    """Without the extra the message must say how to install it."""
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"%PDF-1.4")

    with pytest.raises(ExtractionError, match="--extra pdf"):
        extract_text(target, formats=FORMATS)


@pytest.mark.skipif(not HAS_PYPDF, reason="pypdf not installed: uv sync --extra pdf")
def test_pdf_text_is_extracted(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(_tiny_pdf("Hello from Arthur"))

    text, ext = extract_text(target, formats=FORMATS)

    assert ext == "pdf"
    assert "Hello from Arthur" in text


@pytest.mark.skipif(not HAS_PYPDF, reason="pypdf not installed: uv sync --extra pdf")
def test_corrupt_pdf_raises_instead_of_propagating(tmp_path: Path) -> None:
    """pypdf raises many exception types; they must all surface as one error."""
    target = tmp_path / "broken.pdf"
    target.write_bytes(b"%PDF-1.4\nnot really a pdf at all")

    with pytest.raises(ExtractionError, match="cannot parse PDF"):
        extract_text(target, formats=FORMATS)
