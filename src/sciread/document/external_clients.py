"""Ollama API client for embedding operations."""

import io
import math
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

import requests

from ..logging_config import get_logger


class OllamaClient:
    """Client for interacting with Ollama API for embeddings."""

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        cache_embeddings: bool = True,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        self.embedding_cache: dict[str, list[float]] = {}
        self.logger = get_logger(__name__)

    def get_embeddings(
        self, texts: list[str], batch_size: int = 10
    ) -> list[list[float]]:
        """
        Get embeddings for texts using Ollama API.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        batch_embeddings = []

        for text in texts:
            # Check cache first
            cache_key = f"{self.model}:{hash(text)}"
            if self.cache_embeddings and cache_key in self.embedding_cache:
                batch_embeddings.append(self.embedding_cache[cache_key])
                continue

            # Get embedding from Ollama
            try:
                embedding = self._get_single_embedding(text)
                if embedding:
                    batch_embeddings.append(embedding)
                    if self.cache_embeddings:
                        self.embedding_cache[cache_key] = embedding
                else:
                    batch_embeddings.append([0.0] * 768)  # Fallback
            except Exception:
                batch_embeddings.append([0.0] * 768)  # Fallback

        return batch_embeddings

    def _get_single_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for a single text from Ollama API."""
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.model, "prompt": text}

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]

            return None
        except Exception as e:
            self.logger.warning(f"Failed to get embedding from Ollama: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Failed to connect to Ollama: {e}")
            return False

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def calculate_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate centroid of embeddings."""
        if not embeddings:
            return []

        n = len(embeddings[0])
        centroid = [0.0] * n

        for embedding in embeddings:
            for i, value in enumerate(embedding):
                centroid[i] += value

        return [value / len(embeddings) for value in centroid]

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self.embedding_cache),
            "model": self.model,
            "cache_enabled": self.cache_embeddings,
        }


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
        cache_dir: Optional[str] = None,
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
            from .mineru_cache import MineruCacheManager

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
                    self.logger.warning(
                        f"Failed to use cached zip: {e}, falling back to API"
                    )

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

            self.logger.info(
                f"Successfully extracted {len(markdown_content)} characters from cached zip"
            )
            return markdown_content

    def _call_mineru_api(self, file_path: Path) -> tuple[str, Optional[bytes]]:
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
            self.logger.error(
                f"Mineru API request failed: {response.status_code} - {response.text}"
            )
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
            self.logger.error(
                f"Unexpected upload_urls format: {type(upload_urls)} - {upload_urls}"
            )
            raise RuntimeError(f"Unexpected upload_urls format: {type(upload_urls)}")

        self.logger.info(f"Uploading PDF to Mineru (batch_id: {batch_id})")

        with file_path.open("rb") as f:
            upload_response = requests.put(upload_url, data=f, timeout=300)

        if upload_response.status_code != 200:
            self.logger.error(
                f"Failed to upload PDF to Mineru: {upload_response.status_code}"
            )
            raise RuntimeError(
                f"Failed to upload PDF to Mineru: {upload_response.status_code}"
            )

        self.logger.info("PDF uploaded successfully, waiting for processing...")

        # Poll for processing results
        result_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        max_attempts = self.timeout // self.poll_interval
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            self.logger.debug(
                f"Checking processing status (attempt {attempt}/{max_attempts})"
            )

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
                                self.logger.error(
                                    "Mineru processing completed but no zip URL provided"
                                )
                                raise RuntimeError("No zip URL returned from Mineru")

                            markdown_content, zip_content = (
                                self._download_and_extract_zip(zip_url)
                            )
                            if markdown_content:
                                self.logger.info(
                                    f"Successfully extracted {len(markdown_content)} characters from Mineru"
                                )
                                return markdown_content, zip_content
                            else:
                                self.logger.error(
                                    "Failed to extract markdown content from Mineru zip"
                                )
                                raise RuntimeError(
                                    "Failed to extract markdown from Mineru zip"
                                )
                        elif state == "failed":
                            error_msg = extract_result.get(
                                "err_msg", "Processing failed"
                            )
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
                    self.logger.warning(
                        f"Failed to check status: {status_response.status_code}"
                    )
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
                self.logger.error(
                    f"Failed to download zip file: {response.status_code}"
                )
                raise RuntimeError(
                    f"Failed to download zip file: {response.status_code}"
                )

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

                self.logger.info(
                    f"Successfully extracted {len(markdown_content)} characters from full.md"
                )
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
