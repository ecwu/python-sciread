"""External API clients for embedding and document processing operations."""

import io
import time
import uuid
import zipfile
from pathlib import Path

import requests

from ..logging_config import get_logger
from .mineru_cache import MineruCacheManager


class MineruClient:
    """Client for interacting with Mineru API for PDF to markdown conversion."""

    def __init__(
        self,
        token: str,
        enable_formula: bool = False,
        language: str = "en",
        enable_table: bool = True,
        timeout: int = 300,
        poll_interval: int = 10,
        enable_cache: bool = True,
        cache_dir: str | None = None,
    ):
        """
        Initialize Mineru client.

        Args:
            token: Mineru API token
            enable_formula: Whether to enable formula extraction
            language: Language for extraction
            enable_table: Whether to enable table extraction
            timeout: Maximum processing time in seconds
            poll_interval: Polling interval in seconds
            enable_cache: Whether to enable caching of API responses
            cache_dir: Directory for cache storage (default: ~/.sciread/mineru_cache)
        """
        self.token = token
        self.enable_formula = enable_formula
        self.language = language
        self.enable_table = enable_table
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.enable_cache = enable_cache
        self.logger = get_logger(__name__)

        # Initialize cache manager if caching is enabled
        if self.enable_cache:
            cache_path = Path(cache_dir) if cache_dir else None
            self.cache_manager = MineruCacheManager(cache_dir=cache_path)
            self.logger.info("Mineru caching enabled")
        else:
            self.cache_manager = None
            self.logger.info("Mineru caching disabled")

    def extract_markdown(self, file_path) -> str:
        """
        Extract markdown content from a PDF file using Mineru API.

        Args:
            file_path: Path to the PDF file

        Returns:
            Markdown content as string

        Raises:
            RuntimeError: If extraction fails
        """
        file_path = Path(file_path)

        # Check cache first if enabled
        if self.enable_cache and self.cache_manager:
            cached_zip_path = self.cache_manager.get_cached_zip(file_path)
            if cached_zip_path:
                self.logger.info(f"Using cached result for {file_path.name}")
                try:
                    # Extract markdown from cached zip
                    markdown_content = self._extract_from_cached_zip(cached_zip_path)
                    if markdown_content:
                        return markdown_content
                except RuntimeError as e:
                    self.logger.warning(f"Failed to use cached zip: {e}, falling back to API")

        # If no cache or cache failed, call API
        markdown_content, zip_content = self._call_mineru_api(file_path)

        # Save to cache if enabled
        if self.enable_cache and self.cache_manager and zip_content:
            self.cache_manager.save_to_cache(file_path, zip_content)

        return markdown_content

    def _extract_from_cached_zip(self, zip_path: Path) -> str:
        """Extract markdown content from a cached zip file."""
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            # Look for full.md file in the zip
            full_md_path = None
            for file_info in zip_file.filelist:
                if file_info.filename.endswith("full.md"):
                    full_md_path = file_info.filename
                    break

            if not full_md_path:
                self.logger.error("No full.md file found in cached zip")
                raise RuntimeError("No full.md file found in cached zip")

            # Read the content of full.md
            with zip_file.open(full_md_path) as md_file:
                markdown_content = md_file.read().decode("utf-8")

            self.logger.info(f"Successfully extracted {len(markdown_content)} characters from cached zip")
            return markdown_content

    def _call_mineru_api(self, file_path: Path) -> tuple[str, bytes | None]:
        """
        Call Mineru API to extract markdown from PDF.

        Returns:
            Tuple of (markdown_content, zip_file_content)
        """
        # Prepare API request
        url = "https://mineru.net/api/v4/file-urls/batch"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        # Generate unique data_id for this file
        data_id = str(uuid.uuid4())

        data = {
            "enable_formula": self.enable_formula,
            "language": self.language,
            "enable_table": self.enable_table,
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

        self.logger.debug(f"Received upload URLs: {upload_urls}")

        if not upload_urls:
            self.logger.error("No upload URLs received from Mineru API")
            raise RuntimeError("No upload URLs received from Mineru API")

        # Handle the case where upload_urls might be a list or nested structure
        if isinstance(upload_urls, list) and len(upload_urls) > 0:
            upload_url = upload_urls[0]
        elif isinstance(upload_urls, str):
            upload_url = upload_urls
        else:
            self.logger.error(f"Unexpected upload_urls format: {type(upload_urls)} - {upload_urls}")
            raise RuntimeError(f"Unexpected upload_urls format: {type(upload_urls)}")

        self.logger.info(f"Uploading PDF to Mineru (batch_id: {batch_id})")

        with file_path.open("rb") as f:
            upload_response = requests.put(upload_url, data=f, timeout=300)

        if upload_response.status_code != 200:
            self.logger.error(f"Failed to upload PDF to Mineru: {upload_response.status_code}")
            raise RuntimeError(f"Failed to upload PDF to Mineru: {upload_response.status_code}")

        self.logger.info("PDF uploaded successfully, waiting for processing...")

        # Poll for processing results
        result_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        max_attempts = self.timeout // self.poll_interval
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
                        self.logger.debug(f"API response data: {result_info}")

                        # Handle different possible response structures
                        extract_result = result_info.get("extract_result", {})
                        if isinstance(extract_result, list) and len(extract_result) > 0:
                            extract_result = extract_result[0]
                        elif isinstance(extract_result, list):
                            self.logger.error("extract_result is an empty list")
                            raise RuntimeError("Empty extract_result received from API")

                        self.logger.debug(f"Extract result: {extract_result}")
                        state = extract_result.get("state")

                        if state == "done":
                            # Download and extract the zip file
                            zip_url = extract_result.get("full_zip_url")
                            if not zip_url:
                                self.logger.error("Mineru processing completed but no zip URL provided")
                                raise RuntimeError("No zip URL returned from Mineru")

                            markdown_content, zip_content = self._download_and_extract_zip(zip_url)
                            if markdown_content:
                                self.logger.info(f"Successfully extracted {len(markdown_content)} characters from Mineru")
                                return markdown_content, zip_content
                            else:
                                self.logger.error("Failed to extract markdown content from Mineru zip")
                                raise RuntimeError("Failed to extract markdown from Mineru zip")
                        elif state == "failed":
                            error_msg = extract_result.get("err_msg", "Processing failed")
                            self.logger.error(f"Mineru processing failed: {error_msg}")
                            raise RuntimeError(f"Mineru processing failed: {error_msg}")
                        elif state in [
                            "waiting-file",
                            "pending",
                            "running",
                            "converting",
                        ]:
                            # Still processing, wait and retry
                            self.logger.debug(f"Mineru processing status: {state}")
                            time.sleep(self.poll_interval)
                            continue
                        else:
                            self.logger.warning(f"Unknown Mineru state: {state}")
                            time.sleep(self.poll_interval)
                            continue
                    else:
                        error_msg = status_data.get("msg", "Unknown error")
                        self.logger.error(f"Mineru status check failed: {error_msg}")
                        time.sleep(self.poll_interval)
                        continue
                else:
                    self.logger.warning(f"Failed to check status: {status_response.status_code}")
                    time.sleep(self.poll_interval)
                    continue

            except requests.RequestException as e:
                self.logger.warning(f"Status check request failed: {e}")
                time.sleep(self.poll_interval)
                continue

        # Timeout reached
        self.logger.error(f"Mineru processing timed out after {self.timeout} seconds")
        raise RuntimeError(f"Mineru processing timed out after {self.timeout} seconds")

    def _download_and_extract_zip(self, zip_url: str) -> tuple[str, bytes]:
        """Download and extract the zip file to get the markdown content.

        Returns:
            Tuple of (markdown_content, zip_file_content)
        """
        try:
            self.logger.info(f"Downloading zip file from: {zip_url}")

            # Download the zip file
            response = requests.get(zip_url, timeout=300)
            if response.status_code != 200:
                self.logger.error(f"Failed to download zip file: {response.status_code}")
                raise RuntimeError(f"Failed to download zip file: {response.status_code}")

            # Store zip content for caching
            zip_content = response.content

            # Extract the zip content in memory
            zip_data = io.BytesIO(zip_content)

            with zipfile.ZipFile(zip_data) as zip_file:
                # Look for full.md file in the zip
                full_md_path = None
                for file_info in zip_file.filelist:
                    if file_info.filename.endswith("full.md"):
                        full_md_path = file_info.filename
                        break

                if not full_md_path:
                    self.logger.error("No full.md file found in the zip archive")
                    # List all files for debugging
                    file_list = [f.filename for f in zip_file.filelist]
                    self.logger.error(f"Files in zip: {file_list}")
                    raise RuntimeError("No full.md file found in the zip archive")

                # Read the content of full.md
                with zip_file.open(full_md_path) as md_file:
                    markdown_content = md_file.read().decode("utf-8")

                self.logger.info(f"Successfully extracted {len(markdown_content)} characters from full.md")
                return markdown_content, zip_content

        except requests.RequestException as e:
            self.logger.error(f"Failed to download zip file: {e}")
            raise RuntimeError(f"Failed to download zip file: {e}") from e
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file format: {e}")
            raise RuntimeError(f"Invalid zip file format: {e}") from e
        except UnicodeDecodeError as e:
            self.logger.error(f"Failed to decode markdown file: {e}")
            raise RuntimeError(f"Failed to decode markdown file: {e}") from e
