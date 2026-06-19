"""Document loader contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes import FakeMineruClient

from sciread.document.ingestion.loaders.pdf_loader import PdfLoader
from sciread.document.ingestion.loaders.txt_loader import TxtLoader

pytestmark = pytest.mark.contracts


def test_txt_loader_contract_for_missing_empty_and_binary_like_files(tmp_path: Path) -> None:
    """Text loaders should expose stable can_load, warning, and metadata behavior."""
    loader = TxtLoader()
    missing_file = tmp_path / "missing.txt"
    empty_file = tmp_path / "empty.txt"
    binary_like_file = tmp_path / "binary.txt"
    empty_file.write_text("", encoding="utf-8")
    binary_like_file.write_bytes(b"\x00\x01\x02\x03")

    empty_result = loader.load(empty_file)
    binary_result = loader.load(binary_like_file)

    assert loader.can_load(missing_file) is False
    assert loader.can_load(empty_file) is True
    assert empty_result.success is True
    assert empty_result.text == ""
    assert "File is empty or contains only whitespace" in empty_result.warnings
    assert binary_result.metadata.source_path == binary_like_file
    assert "character_count" in binary_result.extraction_info


def test_pdf_loader_contract_uses_injected_mineru_client_without_token(tmp_path: Path) -> None:
    """PDF markdown mode should accept an injected Mineru client and avoid config token lookup."""
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake payload for injected Mineru")
    mineru_client = FakeMineruClient(markdown="# Paper\n\n## Abstract\n\nConverted by fake Mineru.")
    loader = PdfLoader(to_markdown=True, mineru_client=mineru_client)

    result = loader.load(pdf_file)

    assert result.success is True
    assert result.text.startswith("# Paper")
    assert result.extraction_info["extraction_method"] == "mineru_markdown"
    assert mineru_client.calls == [pdf_file]


def test_pdf_loader_contract_records_failure_when_no_text_can_be_extracted(tmp_path: Path) -> None:
    """Fallback PDF extraction failures should be reflected in LoadResult errors."""
    pdf_file = tmp_path / "broken.pdf"
    pdf_file.write_bytes(b"not a valid pdf")
    loader = PdfLoader(to_markdown=False)

    result = loader.load(pdf_file)

    assert result.success is False
    assert result.text == ""
    assert "No text could be extracted from PDF" in result.errors
