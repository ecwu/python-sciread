"""Tests for PDF loader."""

from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from sciread.document.ingestion.loaders.pdf_loader import PdfLoader


class TestPdfLoader:
    """Test cases for PdfLoader."""

    @pytest.fixture
    def loader(self):
        """Create a PdfLoader instance."""
        return PdfLoader()

    def test_supported_extensions(self, loader):
        """Test supported file extensions."""
        extensions = loader.supported_extensions
        assert ".pdf" in extensions
        assert ".txt" not in extensions
        assert ".md" not in extensions

    def test_loader_name(self, loader):
        """Test loader name."""
        assert loader.loader_name == "PdfLoader"

    def test_can_load_supported_files(self, loader, sample_pdf_file):
        """Test loader can load supported file types."""
        assert loader.can_load(sample_pdf_file)

    def test_can_load_unsupported_files(self, loader, temp_dir):
        """Test loader cannot load unsupported file types."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("fake text content")
        assert not loader.can_load(txt_file)

    def test_can_load_nonexistent_file(self, loader):
        """Test loader cannot load nonexistent files."""
        nonexistent = Path("/nonexistent/file.pdf")
        assert not loader.can_load(nonexistent)

    def test_initialization_default(self):
        """Test default initialization."""
        loader = PdfLoader()
        assert loader.to_markdown is False

    def test_initialization_with_markdown(self):
        """Test initialization with markdown enabled."""
        loader = PdfLoader(to_markdown=True)
        assert loader.to_markdown is True

    def test_load_pdf_with_mineru_markdown_success(self, sample_pdf_file):
        """Test markdown extraction through Mineru."""
        mineru_client = Mock()
        mineru_client.extract_markdown.return_value = "# Markdown Title\n\nConverted content."
        loader = PdfLoader(to_markdown=True, mineru_client=mineru_client)

        result = loader.load(sample_pdf_file)

        assert result.success
        assert result.text == "# Markdown Title\n\nConverted content."
        assert result.extraction_info["extraction_method"] == "mineru_markdown"
        mineru_client.extract_markdown.assert_called_once_with(sample_pdf_file)

    def test_load_pdf_with_mineru_runtime_error_falls_back(self, sample_pdf_file):
        """Test Mineru failures fall back to local PDF extraction."""
        mineru_client = Mock()
        mineru_client.extract_markdown.side_effect = RuntimeError("Mineru unavailable")
        loader = PdfLoader(to_markdown=True, mineru_client=mineru_client)

        with patch.object(
            loader,
            "_fallback_extraction",
            return_value=("Fallback text content", {"title": "Fallback Title", "author": "Fallback Author", "page_count": 2}),
        ):
            result = loader.load(sample_pdf_file)

        assert result.success
        assert result.text == "Fallback text content"
        assert result.metadata.title == "Fallback Title"
        assert result.metadata.author == "Fallback Author"
        assert result.metadata.page_count == 2
        assert result.extraction_info["extraction_method"] == "fallback"
        assert "Mineru extraction failed" in result.warnings[0]

    @patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader")
    def test_load_pdf_with_pypdf_success(self, mock_pdf_reader, loader, sample_pdf_file):
        """Test successful PDF loading with pypdf."""
        # Mock pypdf reader
        mock_reader = Mock()
        mock_reader.metadata = Mock()
        mock_reader.metadata.get.return_value = "Test Title"
        mock_reader.pages = [Mock(), Mock()]

        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "This is page 1 content. " * 10  # Make it longer to avoid fallback
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "This is page 2 content. " * 10  # Make it longer to avoid fallback

        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader

        result = loader.load(sample_pdf_file)

        assert result.success
        assert "This is page 1 content." in result.text
        assert "This is page 2 content." in result.text
        assert result.metadata.title == "Test Title"
        assert result.extraction_info["extraction_method"] == "pypdf"
        assert result.extraction_info["character_count"] > 0

    @patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader")
    @patch("sciread.document.ingestion.loaders.pdf_loader.pdfplumber.open")
    def test_load_pdf_fallback_to_pdfplumber(self, mock_pdfplumber, mock_pdf_reader, loader, sample_pdf_file):
        """Test PDF loading fallback to pdfplumber when pypdf extraction is poor."""
        # Mock pypdf to return short text
        mock_reader = Mock()
        mock_reader.metadata = None
        mock_reader.pages = [Mock()]
        mock_page = Mock()
        mock_page.extract_text.return_value = "Short"
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        # Mock pdfplumber to return better text
        mock_pdf = Mock()
        mock_page_plumber = Mock()
        mock_page_plumber.extract_text.return_value = "Much longer content from pdfplumber extraction."
        mock_pdf.pages = [mock_page_plumber]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

        result = loader.load(sample_pdf_file)

        assert result.success
        assert "Much longer content from pdfplumber extraction." in result.text
        assert result.extraction_info["extraction_method"] == "pdfplumber"

    @patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader")
    def test_load_pdf_extraction_failure(self, mock_pdf_reader, loader, sample_pdf_file):
        """Test PDF loading when extraction fails completely."""
        # Mock pypdf to return empty text
        mock_reader = Mock()
        mock_reader.metadata = None
        mock_reader.pages = [Mock()]
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        with patch("sciread.document.ingestion.loaders.pdf_loader.pdfplumber.open") as mock_pdfplumber:
            # Mock pdfplumber to also return empty text
            mock_pdf = Mock()
            mock_page_plumber = Mock()
            mock_page_plumber.extract_text.return_value = ""
            mock_pdf.pages = [mock_page_plumber]
            mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

            result = loader.load(sample_pdf_file)

        assert not result.success
        assert len(result.errors) > 0
        assert "No text could be extracted from PDF" in result.errors[0]

    @patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader")
    def test_load_pdf_page_extraction_error(self, mock_pdf_reader, loader, sample_pdf_file):
        """Test PDF loading when a page fails to extract."""
        # Mock pypdf with one failing page
        mock_reader = Mock()
        mock_reader.metadata = Mock()
        mock_reader.metadata.get.return_value = "Test Title"
        mock_reader.pages = [Mock(), Mock()]

        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content."
        mock_page2 = Mock()
        mock_page2.extract_text.side_effect = Exception("Extraction failed")

        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader

        result = loader.load(sample_pdf_file)

        assert result.success
        assert "Page 1 content." in result.text
        # Should have warnings about the failed page
        assert len(result.warnings) > 0

    def test_extraction_quality_warnings(self, loader, temp_dir):
        """Test extraction quality warning detection."""
        # Create a PDF-like file with text that has quality issues
        pdf_file = temp_dir / "quality_issues.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        with patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader") as mock_pdf_reader:
            # Mock text with quality issues
            mock_reader = Mock()
            mock_reader.metadata = None
            mock_page = Mock()
            mock_page.extract_text.return_value = (
                "Text with       excessive     whitespace and encoding issues � and ||| artifacts. " * 5
            )  # Make it longer
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader

            result = loader.load(pdf_file)

        assert result.success
        # Check for expected warnings
        warning_messages = [w.lower() for w in result.warnings]
        assert len(warning_messages) >= 1  # At least one warning should be present
        assert any("encoding" in w for w in warning_messages)  # Should detect encoding issues
        # Note: whitespace and artifacts detection depends on specific patterns

    def test_metadata_creation(self, loader, sample_pdf_file):
        """Test metadata creation from PDF file."""
        with patch("sciread.document.ingestion.loaders.pdf_loader.pypdf.PdfReader") as mock_pdf_reader:
            # Mock PDF with metadata
            mock_reader = Mock()
            mock_reader.metadata = Mock()
            mock_reader.metadata.get.side_effect = lambda key, default="": {"/Title": "Test PDF Title", "/Author": "Test Author"}.get(
                key, default
            )
            mock_reader.pages = [Mock()]
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test content"
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader

            result = loader.load(sample_pdf_file)

        assert result.metadata.source_path == sample_pdf_file
        assert result.metadata.file_type == "pdf"
        assert result.metadata.file_size > 0
        assert result.metadata.modified_at is not None
        assert result.metadata.title == "Test PDF Title"
        assert result.metadata.author == "Test Author"
        assert result.metadata.page_count == 1


@pytest.fixture
def sample_pdf_file(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    # Create a minimal PDF-like file for testing
    # Note: This is not a real PDF, but allows testing file handling
    pdf_file = temp_dir / "sample.pdf"
    pdf_file.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000054 00000 n \n0000000103 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
    )
    return pdf_file
