"""PDF file loader implementation."""

import re
from pathlib import Path
from typing import Optional

import pdfplumber
import pypdf

from ...config import get_config
from ...logging_config import get_logger
from ..external_clients import MineruClient
from .base import BaseLoader
from .base import LoadResult


class PdfLoader(BaseLoader):
    """Loader for PDF files using multiple extraction methods."""

    def __init__(
        self, to_markdown: bool = False, mineru_client: Optional[MineruClient] = None
    ):
        """Initialize the PDF loader.

        Args:
            to_markdown: If True, use Mineru API to convert PDF to markdown.
            mineru_client: Optional MineruClient instance for markdown conversion.
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.to_markdown = to_markdown
        self.mineru_client = mineru_client

    @property
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".pdf"]

    @property
    def loader_name(self) -> str:
        """Return the loader name."""
        return "PdfLoader"

    def load(self, file_path: Path) -> LoadResult:
        """Load text content from a PDF file."""
        self.logger.debug(
            f"Loading PDF file: {file_path} (to_markdown={self.to_markdown})"
        )
        result = LoadResult(text="", metadata=self._create_metadata(file_path))

        try:
            if self.to_markdown:
                # Use Mineru API for markdown conversion
                try:
                    if not self.mineru_client:
                        # Try to create MineruClient from config
                        config = get_config()
                        mineru_token = config.get_mineru_token()
                        if not mineru_token:
                            raise RuntimeError("Mineru token not configured")

                        mineru_config = config.mineru
                        self.mineru_client = MineruClient(
                            token=mineru_token,
                            enable_formula=mineru_config.enable_formula,
                            language=mineru_config.language,
                            enable_table=mineru_config.enable_table,
                            timeout=mineru_config.timeout,
                            poll_interval=mineru_config.poll_interval,
                            enable_cache=mineru_config.enable_cache,
                            cache_dir=mineru_config.cache_dir,
                        )

                    markdown_text = self.mineru_client.extract_markdown(file_path)
                    result.text = markdown_text
                    result.extraction_info["extraction_method"] = "mineru_markdown"
                except RuntimeError as e:
                    # Handle Mineru failure with fallback
                    self.logger.warning(f"Mineru extraction failed: {e}")
                    # Perform fallback extraction
                    text, pdf_metadata = self._fallback_extraction(file_path)
                    result.text = text

                    if pdf_metadata:
                        result.metadata.title = pdf_metadata.get("title")
                        result.metadata.author = pdf_metadata.get("author")
                        result.metadata.page_count = pdf_metadata.get("page_count", 0)

                    result.extraction_info["extraction_method"] = "fallback"
                    result.add_warning("Mineru extraction failed, used fallback method")
            else:
                # Use traditional text extraction methods
                text, pdf_metadata = self._fallback_extraction(file_path)
                result.text = text

                if pdf_metadata:
                    result.metadata.title = pdf_metadata.get("title")
                    result.metadata.author = pdf_metadata.get("author")
                    result.metadata.page_count = pdf_metadata.get("page_count", 0)

                result.extraction_info["extraction_method"] = (
                    "pypdf" if len(text.strip()) >= 100 else "pdfplumber"
                )

            # Validate extracted text
            if not result.text.strip():
                self.logger.error("No text could be extracted from PDF")
                result.add_error("No text could be extracted from PDF")
                return result

            # Add extraction statistics
            result.extraction_info.update(
                {
                    "character_count": len(result.text),
                    "word_count": len(result.text.split()),
                }
            )

            self.logger.info(
                f"Successfully extracted {len(result.text)} characters from PDF"
            )

            # Check for common extraction issues (only for non-markdown extraction)
            if not self.to_markdown:
                self._check_extraction_quality(result)

        except Exception as e:
            self.logger.error(f"Failed to load PDF {file_path}: {e}")
            result.add_error(f"Failed to load PDF: {e}")

        return result

    def _fallback_extraction(self, file_path: Path) -> tuple[str, dict]:
        """Extract text using fallback methods (pypdf -> pdfplumber)."""
        # Try pypdf first
        text, pdf_metadata = self._extract_with_pypdf(file_path)

        # If pypdf extraction is too short, try pdfplumber
        if len(text.strip()) < 100:
            self.logger.warning(
                "PyPDF2 extraction yielded little text, trying pdfplumber"
            )
            pdfplumber_text = self._extract_with_pdfplumber(file_path)
            if len(pdfplumber_text) > len(text):
                text = pdfplumber_text

        return text, pdf_metadata

    def _extract_with_pypdf(self, file_path: Path) -> tuple[str, dict]:
        """Extract text using pypdf."""
        try:
            reader = pypdf.PdfReader(str(file_path))

            # Extract metadata
            metadata = {}
            if reader.metadata:
                metadata["title"] = reader.metadata.get("/Title", "").strip()
                metadata["author"] = reader.metadata.get("/Author", "").strip()

            metadata["page_count"] = len(reader.pages)

            # Extract text from all pages
            text_parts = []
            for _page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    # Continue with other pages if one fails
                    self.logger.warning(
                        f"Failed to extract text from page {_page_num}: {e}"
                    )
                    continue

            return "\n\n".join(text_parts), metadata

        except Exception:
            return "", {}

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber."""
        try:
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        # Continue with other pages
                        self.logger.warning(
                            f"Failed to extract text from page {page_num} with pdfplumber: {e}"
                        )
                        continue

            return "\n\n".join(text_parts)

        except Exception:
            return ""

    def _check_extraction_quality(self, result: LoadResult) -> None:
        """Check for common PDF extraction issues."""
        text = result.text

        # Check for excessive whitespace
        if re.search(r"\s{10,}", text):
            self.logger.warning("Document contains excessive whitespace")
            result.add_warning("Document contains excessive whitespace")

        # Check for potential OCR artifacts (if this was a scanned PDF)
        if re.search(r"\|\s*\|\s*\|", text) or re.search(r"\.{5,}", text):
            self.logger.warning("Document may contain OCR artifacts or tables")
            result.add_warning("Document may contain OCR artifacts or tables")

        # Check for encoding issues
        if "�" in text:
            self.logger.warning("Document contains encoding issues")
            result.add_warning("Document contains encoding issues")

        # Check for very short average line length (might indicate column formatting issues)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            avg_line_length = sum(len(line) for line in lines) / len(lines)
            if avg_line_length < 20:
                self.logger.warning(
                    "Average line length is very short, may have formatting issues"
                )
                result.add_warning(
                    "Average line length is very short, may have formatting issues"
                )
