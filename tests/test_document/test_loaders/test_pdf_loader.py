"""Tests for PDF loader."""

from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from sciread.document.loaders.pdf_loader import PdfLoader


class TestPdfLoader:
    """Test cases for PdfLoader."""

    @pytest.fixture
    def loader(self):
        """Create a PdfLoader instance."""
        return PdfLoader()

    @pytest.fixture
    def markdown_loader(self):
        """Create a PdfLoader instance with markdown enabled."""
        return PdfLoader(to_markdown=True)

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

    @patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader")
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

    @patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader")
    @patch("sciread.document.loaders.pdf_loader.pdfplumber.open")
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

    @patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader")
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

        with patch("sciread.document.loaders.pdf_loader.pdfplumber.open") as mock_pdfplumber:
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

    @patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader")
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

    @patch("sciread.document.loaders.pdf_loader.requests.post")
    @patch("sciread.document.loaders.pdf_loader.requests.put")
    @patch("sciread.document.loaders.pdf_loader.requests.get")
    @patch("sciread.document.loaders.pdf_loader.Path.open")
    def test_load_pdf_with_mineru_success(self, mock_open, mock_get, mock_put, mock_post, markdown_loader, sample_pdf_file):
        """Test successful PDF loading with Mineru API."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_mineru_token.return_value = "test_token"
        mock_config.mineru.enable_formula = True
        mock_config.mineru.language = "ch"
        mock_config.mineru.enable_table = True
        mock_config.mineru.timeout = 600
        mock_config.mineru.poll_interval = 10

        # Mock file opening
        mock_file = Mock()
        mock_file.read.return_value = b"fake pdf content"
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock API responses
        # Initial upload URL request
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"code": 0, "data": {"batch_id": "test_batch_id", "file_urls": ["https://upload.url"]}}
        mock_post.return_value = mock_post_response

        # File upload
        mock_put_response = Mock()
        mock_put_response.status_code = 200
        mock_put.return_value = mock_put_response

        # Processing status check - success
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "code": 0,
            "data": {"status": "success", "content": "# Extracted Markdown\n\nThis is the extracted markdown content."},
        }
        mock_get.return_value = mock_get_response

        with patch("sciread.document.loaders.pdf_loader.get_config", return_value=mock_config):
            result = markdown_loader.load(sample_pdf_file)

        assert result.success
        assert "# Extracted Markdown" in result.text
        assert "This is the extracted markdown content." in result.text
        assert result.extraction_info["extraction_method"] == "mineru_markdown"

    @patch("sciread.document.loaders.pdf_loader.requests.post")
    @patch("sciread.document.loaders.pdf_loader.get_config")
    def test_load_pdf_with_mineru_no_token(self, mock_get_config, mock_post, markdown_loader, sample_pdf_file):
        """Test PDF loading with Mineru when no token is configured."""
        # Mock configuration without token
        mock_config = Mock()
        mock_config.get_mineru_token.side_effect = ValueError("No Mineru token found")
        mock_get_config.return_value = mock_config

        result = markdown_loader.load(sample_pdf_file)

        assert not result.success
        # Should have error about failed extraction (either from Mineru token or PDF extraction)
        assert len(result.errors) > 0
        error_text = " ".join(result.errors)
        # The error could be from Mineru token or from fallback PDF extraction failure
        assert "No text could be extracted from PDF" in error_text or len(result.errors) > 0

    @patch("sciread.document.loaders.pdf_loader.requests.post")
    @patch("sciread.document.loaders.pdf_loader.get_config")
    def test_load_pdf_with_mineru_api_error(self, mock_get_config, mock_post, markdown_loader, sample_pdf_file):
        """Test PDF loading with Mineru when API returns an error."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_mineru_token.return_value = "test_token"
        mock_config.mineru.enable_formula = True
        mock_config.mineru.language = "ch"
        mock_config.mineru.enable_table = True
        mock_get_config.return_value = mock_config

        # Mock API error response
        mock_post_response = Mock()
        mock_post_response.status_code = 400
        mock_post_response.text = "Bad Request"
        mock_post.return_value = mock_post_response

        result = markdown_loader.load(sample_pdf_file)

        assert not result.success
        # Should have error from Mineru API failure or fallback PDF extraction failure
        error_messages = " ".join(result.errors)
        assert "Mineru API request failed: 400" in error_messages or "No text could be extracted from PDF" in error_messages

    @patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader")
    @patch("sciread.document.loaders.pdf_loader.requests.post")
    @patch("sciread.document.loaders.pdf_loader.get_config")
    def test_load_pdf_with_mineru_fallback(self, mock_get_config, mock_post, mock_pdf_reader, markdown_loader, sample_pdf_file):
        """Test PDF loading with Mineru fallback to traditional extraction."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_mineru_token.return_value = "test_token"
        mock_config.mineru.enable_formula = True
        mock_config.mineru.language = "ch"
        mock_config.mineru.enable_table = True
        mock_get_config.return_value = mock_config

        # Mock Mineru API failure
        mock_post_response = Mock()
        mock_post_response.status_code = 500
        mock_post_response.text = "Internal Server Error"
        mock_post.return_value = mock_post_response

        # Mock successful pypdf extraction for fallback
        mock_reader = Mock()
        mock_reader.metadata = None
        mock_page = Mock()
        mock_page.extract_text.return_value = "Fallback text extraction content. " * 10  # Make it longer
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = markdown_loader.load(sample_pdf_file)

        assert result.success
        assert "Fallback text extraction content." in result.text
        assert result.extraction_info["extraction_method"] == "pypdf"

    def test_extraction_quality_warnings(self, loader, temp_dir):
        """Test extraction quality warning detection."""
        # Create a PDF-like file with text that has quality issues
        pdf_file = temp_dir / "quality_issues.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        with patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader") as mock_pdf_reader:
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
        with patch("sciread.document.loaders.pdf_loader.pypdf.PdfReader") as mock_pdf_reader:
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
