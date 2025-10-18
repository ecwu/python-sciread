"""Cache manager for Mineru API responses."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger


@dataclass
class CacheEntry:
    """Represents a cache entry for a PDF file."""

    pdf_md5: str
    zip_path: str
    zip_md5: str
    timestamp: float


class MineruCacheManager:
    """Manages caching of Mineru API responses."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached zip files. If None, uses ~/.sciread/mineru_cache
        """
        self.logger = get_logger(__name__)

        if cache_dir is None:
            cache_dir = Path.home() / ".sciread" / "mineru_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache index file stores mappings
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()

    def _load_index(self) -> dict[str, CacheEntry]:
        """Load the cache index from disk."""
        if not self.index_file.exists():
            return {}

        try:
            with self.index_file.open("r") as f:
                data = json.load(f)

            # Convert dict to CacheEntry objects
            return {
                pdf_md5: CacheEntry(**entry_data)
                for pdf_md5, entry_data in data.items()
            }
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load cache index: {e}")
            return {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            # Convert CacheEntry objects to dicts
            data = {
                pdf_md5: {
                    "pdf_md5": entry.pdf_md5,
                    "zip_path": entry.zip_path,
                    "zip_md5": entry.zip_md5,
                    "timestamp": entry.timestamp,
                }
                for pdf_md5, entry in self.cache_index.items()
            }

            with self.index_file.open("w") as f:
                json.dump(data, f, indent=2)
        except (OSError, TypeError) as e:
            self.logger.error(f"Failed to save cache index: {e}")

    def calculate_file_md5(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash as hex string
        """
        md5_hash = hashlib.md5()  # noqa: S324

        with file_path.open("rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest()

    def get_cached_zip(self, pdf_path: Path) -> Optional[Path]:
        """
        Check if a cached zip file exists for the given PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Path to cached zip file if valid cache exists, None otherwise
        """
        try:
            # Calculate PDF MD5
            pdf_md5 = self.calculate_file_md5(pdf_path)
            self.logger.debug(f"PDF MD5: {pdf_md5}")

            # Check if cache entry exists
            if pdf_md5 not in self.cache_index:
                self.logger.debug(f"No cache entry found for PDF: {pdf_path.name}")
                return None

            entry = self.cache_index[pdf_md5]
            zip_path = Path(entry.zip_path)

            # Verify zip file exists
            if not zip_path.exists():
                self.logger.warning(
                    f"Cached zip file not found: {zip_path}, removing cache entry"
                )
                del self.cache_index[pdf_md5]
                self._save_index()
                return None

            # Verify zip file integrity
            current_zip_md5 = self.calculate_file_md5(zip_path)
            if current_zip_md5 != entry.zip_md5:
                self.logger.warning(
                    f"Zip file MD5 mismatch for {zip_path}, removing cache entry"
                )
                # Remove corrupted zip file
                zip_path.unlink(missing_ok=True)
                del self.cache_index[pdf_md5]
                self._save_index()
                return None

            self.logger.info(f"Valid cache found for {pdf_path.name}")
            return zip_path

        except (OSError, KeyError) as e:
            self.logger.error(f"Error checking cache: {e}")
            return None

    def save_to_cache(self, pdf_path: Path, zip_content: bytes) -> Optional[Path]:
        """
        Save a zip file to cache.

        Args:
            pdf_path: Path to the source PDF file
            zip_content: Content of the zip file as bytes

        Returns:
            Path to the saved zip file, or None if save failed
        """
        try:
            # Calculate PDF MD5
            pdf_md5 = self.calculate_file_md5(pdf_path)

            # Create unique filename for zip
            zip_filename = f"{pdf_md5}.zip"
            zip_path = self.cache_dir / zip_filename

            # Save zip file
            with zip_path.open("wb") as f:
                f.write(zip_content)

            # Calculate zip MD5
            zip_md5 = self.calculate_file_md5(zip_path)

            # Create cache entry
            entry = CacheEntry(
                pdf_md5=pdf_md5,
                zip_path=str(zip_path.absolute()),
                zip_md5=zip_md5,
                timestamp=time.time(),
            )

            # Update index
            self.cache_index[pdf_md5] = entry
            self._save_index()

            self.logger.info(f"Saved zip to cache: {zip_path}")
            return zip_path

        except OSError as e:
            self.logger.error(f"Failed to save to cache: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached files and index."""
        try:
            # Remove all zip files
            for entry in self.cache_index.values():
                zip_path = Path(entry.zip_path)
                zip_path.unlink(missing_ok=True)

            # Clear index
            self.cache_index = {}
            self._save_index()

            self.logger.info("Cache cleared")

        except OSError as e:
            self.logger.error(f"Failed to clear cache: {e}")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        valid_entries = 0

        for entry in self.cache_index.values():
            zip_path = Path(entry.zip_path)
            if zip_path.exists():
                total_size += zip_path.stat().st_size
                valid_entries += 1

        return {
            "total_entries": len(self.cache_index),
            "valid_entries": valid_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
