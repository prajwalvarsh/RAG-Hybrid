"""Tests for ingest/loader.py."""

import pytest

from ingest.loader import load_document
from ingest.schemas import FileType


def test_load_txt(tmp_path):
    """load_document returns a Document with correct text and metadata for .txt files."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello from a text file.", encoding="utf-8")

    doc = load_document(str(txt_file))

    assert doc.text == "Hello from a text file."
    assert doc.metadata.file_name == "sample.txt"
    assert doc.metadata.file_type == FileType.TXT
    assert doc.doc_id  # non-empty UUID


def test_load_md(tmp_path):
    """load_document returns a Document with correct text and metadata for .md files."""
    md_file = tmp_path / "readme.md"
    md_file.write_text("# Title\n\nSome markdown content.", encoding="utf-8")

    doc = load_document(str(md_file))

    assert doc.text == "# Title\n\nSome markdown content."
    assert doc.metadata.file_name == "readme.md"
    assert doc.metadata.file_type == FileType.MD
    assert doc.doc_id


def test_unsupported_file_type_raises(tmp_path):
    """load_document raises ValueError for file types that are not supported."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(csv_file))
