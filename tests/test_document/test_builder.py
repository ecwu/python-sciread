"""Tests for document builder orchestration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from sciread.document import DocumentBuilder
from sciread.document.ingestion.loaders.base import LoadResult
from sciread.document.models import DocumentMetadata


class TestDocumentBuilder:
    """Test cases for DocumentBuilder."""

    def test_builder_resolves_default_loader_per_call(self, sample_txt_file, temp_dir: Path):
        """Test builder picks a fresh default loader for each file type."""
        builder = DocumentBuilder()

        txt_doc = builder.from_file(sample_txt_file, auto_split=False)
        assert txt_doc.source_path == sample_txt_file

        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%%EOF")

        with patch("sciread.document.document_builder.PdfLoader.load") as mock_load:
            mock_load.return_value = LoadResult(
                text="PDF body",
                metadata=DocumentMetadata(source_path=pdf_file, file_type="pdf"),
            )

            pdf_doc = builder.from_file(pdf_file, auto_split=False)

        assert pdf_doc.source_path == pdf_file
        assert pdf_doc.text == "PDF body"

    def test_builder_rejects_unsupported_file_format(self, temp_dir: Path):
        """Test unsupported file extensions fail with a clear error."""
        builder = DocumentBuilder()
        unsupported_file = temp_dir / "sample.docx"
        unsupported_file.write_text("not supported", encoding="utf-8")

        with pytest.raises(ValueError, match=r"Unsupported file format: \.docx"):
            builder.from_file(unsupported_file, auto_split=False)
