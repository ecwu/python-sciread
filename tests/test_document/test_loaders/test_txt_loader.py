"""Tests for TXT loader."""

from pathlib import Path
from unittest.mock import Mock

import chardet
import pytest

from sciread.document.ingestion.loaders.txt_loader import TxtLoader


class TestTxtLoader:
    """Test cases for TxtLoader."""

    @pytest.fixture
    def loader(self):
        """Create a TxtLoader instance."""
        return TxtLoader()

    def test_supported_extensions(self, loader):
        """Test supported file extensions."""
        extensions = loader.supported_extensions
        assert ".txt" in extensions
        assert ".text" in extensions
        assert ".md" in extensions
        assert ".rst" in extensions
        assert ".pdf" not in extensions

    def test_loader_name(self, loader):
        """Test loader name."""
        assert loader.loader_name == "TxtLoader"

    def test_can_load_supported_files(self, loader, sample_txt_file):
        """Test loader can load supported file types."""
        assert loader.can_load(sample_txt_file)

    def test_can_load_unsupported_files(self, loader, temp_dir):
        """Test loader cannot load unsupported file types."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")
        assert not loader.can_load(pdf_file)

    def test_can_load_nonexistent_file(self, loader):
        """Test loader cannot load nonexistent files."""
        nonexistent = Path("/nonexistent/file.txt")
        assert not loader.can_load(nonexistent)

    def test_load_txt_file(self, loader, sample_txt_file):
        """Test loading a text file."""
        result = loader.load(sample_txt_file)

        assert result.success
        assert len(result.text) > 0
        assert "Abstract" in result.text
        assert "Introduction" in result.text
        assert "Methods" in result.text
        assert result.metadata.file_type == "txt"
        assert result.metadata.source_path == sample_txt_file

    def test_load_empty_file(self, loader, empty_text_file):
        """Test loading an empty file."""
        result = loader.load(empty_text_file)

        assert result.success
        assert result.text == ""
        assert len(result.warnings) > 0
        assert "empty" in result.warnings[0].lower()

    def test_load_short_file(self, loader, short_text_file):
        """Test loading a very short file."""
        result = loader.load(short_text_file)

        assert result.success
        assert result.text == "This is a very short document."
        assert result.extraction_info["word_count"] == 6

    def test_load_with_encoding_detection(self, loader, temp_dir):
        """Test loading file with encoding detection."""
        # Create a file with UTF-8 content
        utf8_content = "Test content with unicode: café, naïve, résumé"
        utf8_file = temp_dir / "utf8_test.txt"
        utf8_file.write_text(utf8_content, encoding="utf-8")

        result = loader.load(utf8_file)

        assert result.success
        assert result.text == utf8_content
        assert "detected_encoding" in result.extraction_info

    def test_extraction_info(self, loader, sample_txt_file):
        """Test extraction information."""
        result = loader.load(sample_txt_file)

        assert "character_count" in result.extraction_info
        assert "word_count" in result.extraction_info
        assert "line_count" in result.extraction_info
        assert "detected_encoding" in result.extraction_info

        assert result.extraction_info["character_count"] > 0
        assert result.extraction_info["word_count"] > 0
        assert result.extraction_info["line_count"] > 0

    def test_result_has_issues(self, loader, temp_dir):
        """Test result issue detection."""
        # Test successful result with no issues
        normal_file = temp_dir / "normal.txt"
        normal_file.write_text("Normal content")
        result = loader.load(normal_file)
        assert not result.has_issues

        # Test result with warnings
        warning_file = temp_dir / "warning.txt"
        warning_file.write_text("")
        result = loader.load(warning_file)
        assert result.has_issues
        assert len(result.warnings) > 0

    def test_metadata_creation(self, loader, sample_txt_file):
        """Test metadata creation from file."""
        result = loader.load(sample_txt_file)

        assert result.metadata.source_path == sample_txt_file
        assert result.metadata.file_type == "txt"
        assert result.metadata.file_size > 0
        assert result.metadata.modified_at is not None

    def test_detect_encoding_falls_back_when_detected_encoding_is_invalid(self, loader, temp_dir, monkeypatch):
        """Invalid detected encodings should fall back to a decodable default."""
        text_file = temp_dir / "latin1.txt"
        text_file.write_bytes("café".encode("latin-1"))

        monkeypatch.setattr(chardet, "detect", lambda _raw_data: {"encoding": "definitely-not-real"})

        encoding = loader._detect_encoding(text_file)

        assert encoding == "latin-1"

    def test_detect_encoding_returns_utf8_when_file_is_empty(self, loader, temp_dir):
        """Empty files should default to UTF-8."""
        text_file = temp_dir / "empty.txt"
        text_file.write_bytes(b"")

        assert loader._detect_encoding(text_file) == "utf-8"

    def test_detect_encoding_returns_utf8_when_open_fails(self, loader, temp_dir, monkeypatch):
        """Unexpected detection failures should safely default to UTF-8."""
        file_path = temp_dir / "unreadable.txt"
        open_mock = Mock(side_effect=OSError("boom"))
        monkeypatch.setattr(Path, "open", open_mock)

        assert loader._detect_encoding(file_path) == "utf-8"

    def test_load_reports_unicode_decode_errors(self, loader, temp_dir, monkeypatch):
        """Unicode decode failures during file read should become loader errors."""
        text_file = temp_dir / "broken.txt"
        text_file.write_text("placeholder")
        monkeypatch.setattr(loader, "_detect_encoding", lambda _file_path: "utf-8")

        def fake_open(self, *args, **kwargs):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")

        monkeypatch.setattr(Path, "open", fake_open)

        result = loader.load(text_file)

        assert not result.success
        assert result.errors == ["Failed to decode file: 'utf-8' codec can't decode byte 0xff in position 0: bad byte"]

    def test_load_reports_unexpected_errors(self, loader, temp_dir, monkeypatch):
        """Non-IO unexpected exceptions should still be captured in the result."""
        text_file = temp_dir / "broken.txt"
        text_file.write_text("placeholder")
        monkeypatch.setattr(loader, "_detect_encoding", lambda _file_path: "utf-8")

        def fake_open(self, *args, **kwargs):
            raise RuntimeError("unexpected boom")

        monkeypatch.setattr(Path, "open", fake_open)

        result = loader.load(text_file)

        assert not result.success
        assert result.errors == ["Unexpected error loading file: unexpected boom"]
