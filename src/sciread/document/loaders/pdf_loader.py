"""PDF file loader implementation."""

import re
import time
import uuid
from pathlib import Path

import pdfplumber
import pypdf
import requests

from ...config import get_config
from ...logging_config import get_logger
from .base import BaseLoader
from .base import LoadResult


class PdfLoader(BaseLoader):
    """Loader for PDF files using multiple extraction methods."""

    def __init__(self, to_markdown: bool = False):
        """Initialize the PDF loader.

        Args:
            to_markdown: If True, use Mineru API to convert PDF to markdown.
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.to_markdown = to_markdown

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
        self.logger.info(f"Loading PDF file: {file_path} (to_markdown={self.to_markdown})")
        result = LoadResult(text="", metadata=self._create_metadata(file_path))

        try:
            if self.to_markdown:
                # Use Mineru API for markdown conversion
                try:
                    markdown_text = self._extract_with_mineru(file_path)
                    result.text = markdown_text
                    result.extraction_info["extraction_method"] = "mineru_markdown"
                except RuntimeError as e:
                    # Handle Mineru failure with fallback
                    self.logger.warning(f"Mineru extraction failed: {e}")
                    # Extract fallback method from error message
                    fallback_method = "pypdf"  # default
                    if "fallback extraction method:" in str(e):
                        fallback_method = str(e).split("fallback extraction method:")[-1].strip()

                    # Perform fallback extraction
                    text, pdf_metadata = self._extract_with_pypdf(file_path)
                    result.text = text

                    if pdf_metadata:
                        result.metadata.title = pdf_metadata.get("title")
                        result.metadata.author = pdf_metadata.get("author")
                        result.metadata.page_count = pdf_metadata.get("page_count", 0)

                    # If pypdf extraction is too short, try pdfplumber
                    if len(text.strip()) < 100:
                        self.logger.warning("PyPDF2 extraction yielded little text, trying pdfplumber")
                        result.add_warning("PyPDF2 extraction yielded little text, trying pdfplumber")
                        pdfplumber_text = self._extract_with_pdfplumber(file_path)
                        if len(pdfplumber_text) > len(text):
                            result.text = pdfplumber_text
                            fallback_method = "pdfplumber"

                    result.extraction_info["extraction_method"] = fallback_method
                    result.add_warning(f"Mineru extraction failed, used {fallback_method} fallback")
            else:
                # Use traditional text extraction methods
                text, pdf_metadata = self._extract_with_pypdf(file_path)
                result.text = text

                if pdf_metadata:
                    result.metadata.title = pdf_metadata.get("title")
                    result.metadata.author = pdf_metadata.get("author")
                    result.metadata.page_count = pdf_metadata.get("page_count", 0)

                # If text extraction failed or is too short, try pdfplumber
                if len(text.strip()) < 100:
                    self.logger.warning("PyPDF2 extraction yielded little text, trying pdfplumber")
                    result.add_warning("PyPDF2 extraction yielded little text, trying pdfplumber")
                    pdfplumber_text = self._extract_with_pdfplumber(file_path)
                    if len(pdfplumber_text) > len(text):
                        result.text = pdfplumber_text

                result.extraction_info["extraction_method"] = "pypdf" if len(text.strip()) >= 100 else "pdfplumber"

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

            self.logger.info(f"Successfully extracted {len(result.text)} characters from PDF")

            # Check for common extraction issues (only for non-markdown extraction)
            if not self.to_markdown:
                self._check_extraction_quality(result)

        except Exception as e:
            self.logger.error(f"Failed to load PDF {file_path}: {e}")
            result.add_error(f"Failed to load PDF: {e}")

        return result

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
                    self.logger.warning(f"Failed to extract text from page {_page_num}: {e}")
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
                        self.logger.warning(f"Failed to extract text from page {page_num} with pdfplumber: {e}")
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
                self.logger.warning("Average line length is very short, may have formatting issues")
                result.add_warning("Average line length is very short, may have formatting issues")

    def _extract_with_mineru(self, file_path: Path) -> str:
        """Extract markdown content using Mineru API."""
        try:
            # Get Mineru token from configuration
            config = get_config()
            mineru_token = config.get_mineru_token()

            # Get Mineru configuration
            mineru_config = config.mineru

            # Prepare API request
            url = "https://mineru.net/api/v4/file-urls/batch"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {mineru_token}",
            }

            # Generate unique data_id for this file
            data_id = str(uuid.uuid4())

            data = {
                "enable_formula": mineru_config.enable_formula,
                "language": mineru_config.language,
                "enable_table": mineru_config.enable_table,
                "files": [{"name": file_path.name, "is_ocr": True, "data_id": data_id}],
            }

            self.logger.info(f"Submitting PDF to Mineru API: {file_path.name}")

            # Request upload URLs
            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                self.logger.error(f"Mineru API request failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Mineru API request failed: {response.status_code}")

            result = response.json()

            if result.get("code") != 0:
                error_msg = result.get("msg", "Unknown error")
                self.logger.error(f"Mineru API returned error: {error_msg}")
                raise RuntimeError(f"Mineru API error: {error_msg}")

            # Upload the file
            batch_id = result["data"]["batch_id"]
            upload_urls = result["data"]["file_urls"]

            if not upload_urls:
                self.logger.error("No upload URLs received from Mineru API")
                raise RuntimeError("No upload URLs received from Mineru API")

            upload_url = upload_urls[0]

            self.logger.info(f"Uploading PDF to Mineru (batch_id: {batch_id})")

            with open(file_path, "rb") as f:
                upload_response = requests.put(upload_url, data=f, timeout=300)

            if upload_response.status_code != 200:
                self.logger.error(f"Failed to upload PDF to Mineru: {upload_response.status_code}")
                raise RuntimeError(f"Failed to upload PDF to Mineru: {upload_response.status_code}")

            self.logger.info("PDF uploaded successfully, waiting for processing...")

            # Poll for processing results
            result_url = f"https://mineru.net/api/v4/result/{batch_id}"
            max_attempts = mineru_config.timeout // mineru_config.poll_interval
            attempt = 0

            while attempt < max_attempts:
                attempt += 1
                self.logger.debug(f"Checking processing status (attempt {attempt}/{max_attempts})")

                try:
                    status_response = requests.get(result_url, headers=headers, timeout=30)

                    if status_response.status_code == 200:
                        status_data = status_response.json()

                        if status_data.get("code") == 0:
                            result_info = status_data.get("data", {})
                            status = result_info.get("status")

                            if status == "success":
                                # Extract markdown content
                                markdown_content = result_info.get("content", "")
                                if markdown_content:
                                    self.logger.info(f"Successfully extracted {len(markdown_content)} characters from Mineru")
                                    return markdown_content
                                else:
                                    self.logger.error("Mineru processing completed but no content returned")
                                    raise RuntimeError("No content returned from Mineru")
                            elif status == "failed":
                                error_msg = result_info.get("error_msg", "Processing failed")
                                self.logger.error(f"Mineru processing failed: {error_msg}")
                                raise RuntimeError(f"Mineru processing failed: {error_msg}")
                            elif status in ["processing", "pending"]:
                                # Still processing, wait and retry
                                time.sleep(mineru_config.poll_interval)
                                continue
                            else:
                                self.logger.warning(f"Unknown Mineru status: {status}")
                                time.sleep(mineru_config.poll_interval)
                                continue
                        else:
                            error_msg = status_data.get("msg", "Unknown error")
                            self.logger.error(f"Mineru status check failed: {error_msg}")
                            time.sleep(mineru_config.poll_interval)
                            continue
                    else:
                        self.logger.warning(f"Failed to check status: {status_response.status_code}")
                        time.sleep(mineru_config.poll_interval)
                        continue

                except requests.RequestException as e:
                    self.logger.warning(f"Status check request failed: {e}")
                    time.sleep(mineru_config.poll_interval)
                    continue

            # Timeout reached
            self.logger.error(f"Mineru processing timed out after {mineru_config.timeout} seconds")
            raise RuntimeError(f"Mineru processing timed out after {mineru_config.timeout} seconds")

        except Exception as e:
            self.logger.error(f"Failed to extract markdown with Mineru: {e}")
            # Fallback to traditional extraction
            self.logger.warning("Falling back to traditional PDF extraction")
            text, pdf_metadata = self._extract_with_pypdf(file_path)
            extraction_method = "pypdf"
            if len(text.strip()) < 100:
                text = self._extract_with_pdfplumber(file_path)
                extraction_method = "pdfplumber"
            # Raise the exception to be handled by the caller
            raise RuntimeError(f"Mineru extraction failed, fallback extraction method: {extraction_method}") from e
