"""Tests for Mineru ingestion helpers."""

import io
import zipfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from sciread.document.ingestion.external_clients import MineruClient
from sciread.document.ingestion.mineru_cache import MineruCacheManager


def _build_zip_bytes(markdown: str = "# Title\n\nBody", inner_path: str = "result/full.md") -> bytes:
    """Build an in-memory zip archive for Mineru responses."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(inner_path, markdown)
    return buffer.getvalue()


def _create_client(temp_dir: Path) -> MineruClient:
    """Create a Mineru client for tests."""
    fake_token = "dummy-token"  # noqa: S105
    return MineruClient(token=fake_token, cache_dir=str(temp_dir / "mineru-cache"))


@pytest.fixture
def sample_pdf_file(temp_dir: Path) -> Path:
    """Create a sample PDF file for Mineru-related tests."""
    pdf_file = temp_dir / "sample.pdf"
    pdf_file.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
    )
    return pdf_file


class TestMineruCacheManager:
    """Test cases for MineruCacheManager."""

    def test_save_to_cache_and_get_cached_zip_round_trip(self, temp_dir: Path, sample_pdf_file: Path):
        """Saved zip archives should be retrievable from cache."""
        cache_manager = MineruCacheManager(cache_dir=temp_dir / "mineru-cache")
        zip_content = _build_zip_bytes(markdown="# Cached")

        cached_path = cache_manager.save_to_cache(sample_pdf_file, zip_content)
        result = cache_manager.get_cached_zip(sample_pdf_file)

        assert cached_path is not None
        assert result == cached_path

        stats = cache_manager.get_cache_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["total_size_bytes"] > 0

    def test_get_cached_zip_removes_missing_archive(self, temp_dir: Path, sample_pdf_file: Path):
        """Missing cache files should evict the stale index entry."""
        cache_manager = MineruCacheManager(cache_dir=temp_dir / "mineru-cache")
        cached_path = cache_manager.save_to_cache(sample_pdf_file, _build_zip_bytes())

        assert cached_path is not None
        cached_path.unlink()

        result = cache_manager.get_cached_zip(sample_pdf_file)

        assert result is None
        assert cache_manager.cache_index == {}

    def test_get_cached_zip_removes_corrupted_archive(self, temp_dir: Path, sample_pdf_file: Path):
        """Modified cache files should be treated as invalid."""
        cache_manager = MineruCacheManager(cache_dir=temp_dir / "mineru-cache")
        cached_path = cache_manager.save_to_cache(sample_pdf_file, _build_zip_bytes())

        assert cached_path is not None
        cached_path.write_bytes(b"corrupted-zip")

        result = cache_manager.get_cached_zip(sample_pdf_file)

        assert result is None
        assert not cached_path.exists()
        assert cache_manager.cache_index == {}

    def test_clear_cache_removes_archives_and_index(self, temp_dir: Path, sample_pdf_file: Path):
        """clear_cache should remove all cached content."""
        cache_manager = MineruCacheManager(cache_dir=temp_dir / "mineru-cache")
        cached_path = cache_manager.save_to_cache(sample_pdf_file, _build_zip_bytes())

        assert cached_path is not None

        cache_manager.clear_cache()

        assert cache_manager.cache_index == {}
        assert not cached_path.exists()
        assert cache_manager.get_cache_stats()["valid_entries"] == 0


class TestMineruClient:
    """Test cases for MineruClient."""

    def test_extract_markdown_uses_cached_zip(self, temp_dir: Path, sample_pdf_file: Path):
        """Cached Mineru output should bypass the API call."""
        client = _create_client(temp_dir)
        client.cache_manager.save_to_cache(sample_pdf_file, _build_zip_bytes(markdown="# Cached Markdown"))

        with patch.object(client, "_call_mineru_api") as mock_call_api:
            markdown = client.extract_markdown(sample_pdf_file)

        assert markdown == "# Cached Markdown"
        mock_call_api.assert_not_called()

    def test_extract_markdown_falls_back_to_api_after_cache_runtime_error(self, temp_dir: Path, sample_pdf_file: Path):
        """Cache extraction failures should fall back to the Mineru API."""
        client = _create_client(temp_dir)
        cached_zip_path = temp_dir / "cached.zip"

        with patch.object(client.cache_manager, "get_cached_zip", return_value=cached_zip_path):
            with patch.object(client, "_extract_from_cached_zip", side_effect=RuntimeError("bad cache")):
                with patch.object(client, "_call_mineru_api", return_value=("# Fresh Markdown", b"zip-bytes")) as mock_call_api:
                    with patch.object(client.cache_manager, "save_to_cache") as mock_save:
                        markdown = client.extract_markdown(sample_pdf_file)

        assert markdown == "# Fresh Markdown"
        mock_call_api.assert_called_once_with(sample_pdf_file)
        mock_save.assert_called_once_with(sample_pdf_file, b"zip-bytes")

    @patch("sciread.document.ingestion.external_clients.requests.get")
    @patch("sciread.document.ingestion.external_clients.requests.put")
    @patch("sciread.document.ingestion.external_clients.requests.post")
    def test_call_mineru_api_success(self, mock_post, mock_put, mock_get, temp_dir: Path, sample_pdf_file: Path):
        """A successful Mineru run should upload, poll, and download markdown."""
        client = _create_client(temp_dir)

        post_response = Mock()
        post_response.status_code = 200
        post_response.json.return_value = {
            "code": 0,
            "data": {
                "batch_id": "batch-1",
                "file_urls": ["https://upload.example.com/file"],
            },
        }
        mock_post.return_value = post_response

        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response

        get_response = Mock()
        get_response.status_code = 200
        get_response.json.return_value = {
            "code": 0,
            "data": {
                "extract_result": {
                    "state": "done",
                    "full_zip_url": "https://download.example.com/result.zip",
                }
            },
        }
        mock_get.return_value = get_response

        with patch.object(client, "_download_and_extract_zip", return_value=("# Extracted", b"zip-content")) as mock_download:
            markdown, zip_content = client._call_mineru_api(sample_pdf_file)

        assert markdown == "# Extracted"
        assert zip_content == b"zip-content"
        mock_post.assert_called_once()
        mock_put.assert_called_once()
        mock_get.assert_called_once()
        mock_download.assert_called_once_with("https://download.example.com/result.zip")

    @patch("sciread.document.ingestion.external_clients.requests.get")
    @patch("sciread.document.ingestion.external_clients.requests.put")
    @patch("sciread.document.ingestion.external_clients.requests.post")
    def test_call_mineru_api_failed_state_raises_runtime_error(self, mock_post, mock_put, mock_get, temp_dir: Path, sample_pdf_file: Path):
        """A failed Mineru job should surface as RuntimeError."""
        client = _create_client(temp_dir)

        post_response = Mock()
        post_response.status_code = 200
        post_response.json.return_value = {
            "code": 0,
            "data": {
                "batch_id": "batch-1",
                "file_urls": ["https://upload.example.com/file"],
            },
        }
        mock_post.return_value = post_response

        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response

        get_response = Mock()
        get_response.status_code = 200
        get_response.json.return_value = {
            "code": 0,
            "data": {
                "extract_result": {
                    "state": "failed",
                    "err_msg": "processing failed",
                }
            },
        }
        mock_get.return_value = get_response

        with pytest.raises(RuntimeError, match="processing failed"):
            client._call_mineru_api(sample_pdf_file)

    @patch("sciread.document.ingestion.external_clients.requests.get")
    def test_download_and_extract_zip_success(self, mock_get, temp_dir: Path):
        """Downloaded Mineru zips should return markdown and raw bytes."""
        client = _create_client(temp_dir)
        zip_content = _build_zip_bytes(markdown="# Downloaded Markdown")

        response = Mock()
        response.status_code = 200
        response.content = zip_content
        mock_get.return_value = response

        markdown, downloaded_content = client._download_and_extract_zip("https://download.example.com/result.zip")

        assert markdown == "# Downloaded Markdown"
        assert downloaded_content == zip_content

    @patch("sciread.document.ingestion.external_clients.requests.get")
    def test_download_and_extract_zip_raises_when_full_md_missing(self, mock_get, temp_dir: Path):
        """Mineru zips without full.md should be rejected."""
        client = _create_client(temp_dir)
        zip_content = _build_zip_bytes(markdown="body", inner_path="result/summary.md")

        response = Mock()
        response.status_code = 200
        response.content = zip_content
        mock_get.return_value = response

        with pytest.raises(RuntimeError, match=r"No full\.md file found"):
            client._download_and_extract_zip("https://download.example.com/result.zip")
