"""Tests for document builder orchestration."""

from pathlib import Path
from types import SimpleNamespace
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

    def test_builder_uses_markdown_splitter_config_for_default_overlap(self):
        """Test default markdown splitting reads overlap from configuration."""
        builder = DocumentBuilder()
        config = SimpleNamespace(
            document_splitters=SimpleNamespace(
                markdown=SimpleNamespace(
                    model_dump=lambda: {
                        "min_chunk_size": 200,
                        "max_chunk_size": 2000,
                        "chunk_overlap": 8,
                        "preserve_code_blocks": True,
                        "split_on_headers": True,
                        "confidence_threshold": 0.7,
                    }
                ),
                semantic=SimpleNamespace(
                    model_dump=lambda: {
                        "min_chunk_size": 200,
                        "max_chunk_size": 2000,
                        "chunk_overlap": 0,
                        "preserve_code_blocks": True,
                        "split_on_headers": True,
                        "confidence_threshold": 0.7,
                        "enable_academic_patterns": True,
                        "enable_markdown_patterns": False,
                    }
                ),
            )
        )
        text = "# Intro\n\nAlpha beta.\n\n# Results\n\nGamma delta."

        with patch("sciread.document.document_builder.get_config", return_value=config):
            document = builder.from_text(text, is_markdown=True)

        assert len(document.chunks) == 2
        assert document.chunks[0].overlap_next_chars > 0
        assert document.chunks[1].overlap_prev_chars > 0
