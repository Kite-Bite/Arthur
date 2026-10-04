"""Text extraction: format gating, binary rejection, truncation, discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from arthur.retrieval.extractors import (
    ExtractionError,
    UnsupportedFormatError,
    discover_files,
    extract_text,
)

FORMATS = {"md", "txt", "py"}


def test_extracts_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("# Notes\ncafé résumé\n", encoding="utf-8")

    text, ext = extract_text(target, formats=FORMATS)

    assert ext == "md"
    assert "café résumé" in text


def test_undecodable_bytes_are_replaced_not_fatal(tmp_path: Path) -> None:
    target = tmp_path / "raw.txt"
    target.write_bytes(b"ok\xff\xfegood")

    text, _ = extract_text(target, formats=FORMATS)

    assert "ok" in text and "good" in text


def test_unconfigured_extension_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "archive.zip"
    target.write_bytes(b"PK\x03\x04")

    with pytest.raises(UnsupportedFormatError, match="not configured for indexing"):
        extract_text(target, formats=FORMATS)


def test_extension_matching_is_case_insensitive(tmp_path: Path) -> None:
    target = tmp_path / "README.MD"
    target.write_text("hello", encoding="utf-8")

    text, ext = extract_text(target, formats=FORMATS)

    assert ext == "md"
    assert text == "hello"


def test_binary_content_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "blob.txt"
    target.write_bytes(b"\x00\x01\x02 payload")

    with pytest.raises(ExtractionError, match="appears to be binary"):
        extract_text(target, formats=FORMATS)


def test_non_regular_file_is_rejected(tmp_path: Path) -> None:
    # Must carry a valid extension, otherwise the format gate fires first.
    directory = tmp_path / "notes.md"
    directory.mkdir()

    with pytest.raises(ExtractionError, match="not a regular file"):
        extract_text(directory, formats=FORMATS)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="not a regular file"):
        extract_text(tmp_path / "nope.md", formats=FORMATS)


def test_oversized_file_is_truncated_with_a_marker(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("A" * 50 + "TAIL", encoding="utf-8")

    text, _ = extract_text(target, formats=FORMATS, max_bytes=10)

    assert text.startswith("A" * 10)
    assert "[file truncated during ingestion]" in text
    assert "TAIL" not in text


def test_file_at_the_limit_is_not_truncated(tmp_path: Path) -> None:
    target = tmp_path / "edge.txt"
    target.write_text("exact", encoding="utf-8")

    text, _ = extract_text(target, formats=FORMATS, max_bytes=5)

    assert text == "exact"
    assert "truncated" not in text


def test_discover_files_is_recursive_and_filtered(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.zip").write_text("c", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.md").write_text("d", encoding="utf-8")

    found = {p.name for p in discover_files(tmp_path, formats=FORMATS)}

    assert found == {"a.md", "b.txt", "d.md"}


def test_discover_files_prunes_denied_directories(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("k", encoding="utf-8")
    hidden = tmp_path / ".ssh"
    hidden.mkdir()
    (hidden / "id.md").write_text("nope", encoding="utf-8")

    found = {p.name for p in discover_files(tmp_path, formats=FORMATS, denied_dir_names={".ssh"})}

    assert found == {"keep.md"}


def test_discover_files_non_recursive_only_reads_the_top_level(tmp_path: Path) -> None:
    (tmp_path / "top.md").write_text("t", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("d", encoding="utf-8")

    found = {p.name for p in discover_files(tmp_path, formats=FORMATS, recursive=False)}

    assert found == {"top.md"}


def test_discover_files_yields_a_direct_file_and_skips_missing_paths(
    tmp_path: Path,
) -> None:
    single = tmp_path / "one.md"
    single.write_text("1", encoding="utf-8")

    assert list(discover_files(single, formats=FORMATS)) == [single]
    assert list(discover_files(tmp_path / "gone", formats=FORMATS)) == []
