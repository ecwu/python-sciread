"""Tests for Ollama embedding client integration."""

from unittest.mock import Mock
from unittest.mock import patch

from sciread.document import OllamaClient
from sciread.embedding_provider.ollama import OllamaEmbeddingProvider


class TestOllamaClient:
    """Test cases for the local Ollama embedding client."""

    def test_initialization_normalizes_base_url_and_defaults(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434/")

        assert client.model == "embeddinggemma:latest"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 30
        assert client.cache_embeddings is True
        assert client.embedding_dimension is None
        assert client.embedding_cache == {}

    @patch("requests.post")
    def test_get_single_embedding_success_posts_expected_payload(self, mock_post: Mock) -> None:
        mock_post.return_value = Mock(status_code=200, json=Mock(return_value={"embedding": [0.1, 0.2, 0.3]}))
        client = OllamaClient(model="nomic-embed-text", base_url="http://ollama.local", timeout=12)

        embedding = client._get_single_embedding("paper text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_post.assert_called_once_with(
            "http://ollama.local/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "paper text"},
            timeout=12,
            headers={"Content-Type": "application/json"},
        )

    @patch("requests.post")
    def test_get_single_embedding_returns_none_for_bad_response_shapes(self, mock_post: Mock) -> None:
        client = OllamaClient()

        mock_post.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
        assert client._get_single_embedding("missing embedding") is None

        mock_post.return_value = Mock(status_code=500, json=Mock(return_value={"embedding": [1.0]}))
        assert client._get_single_embedding("server failure") is None

    @patch("requests.post")
    def test_get_embeddings_deduplicates_and_uses_zero_fallback(self, mock_post: Mock) -> None:
        mock_post.side_effect = [
            Mock(status_code=200, json=Mock(return_value={"embedding": [0.4, 0.5]})),
            Mock(status_code=500, json=Mock(return_value={})),
        ]
        client = OllamaClient(cache_embeddings=True)

        embeddings = client.get_embeddings(["same", "same", "missing"], batch_size=2)

        assert embeddings == [[0.4, 0.5], [0.4, 0.5], [0.0, 0.0]]
        assert client.embedding_dimension == 2
        assert mock_post.call_count == 2

        mock_post.reset_mock()
        assert client.get_embedding("same") == [0.4, 0.5]
        assert mock_post.call_count == 0

    @patch("requests.get")
    def test_test_connection_checks_tags_endpoint(self, mock_get: Mock) -> None:
        client = OllamaClient(base_url="http://ollama.local/")

        mock_get.return_value = Mock(status_code=200)
        assert client.test_connection() is True
        mock_get.assert_called_once_with("http://ollama.local/api/tags", timeout=5)

        mock_get.return_value = Mock(status_code=503)
        assert client.test_connection() is False

        mock_get.side_effect = RuntimeError("offline")
        assert client.test_connection() is False

    def test_provider_supports_known_and_custom_ollama_model_names(self) -> None:
        supported = OllamaEmbeddingProvider.get_supported_models()

        assert "nomic-embed-text" in supported
        assert OllamaEmbeddingProvider.is_model_supported("nomic-embed-text") is True
        assert OllamaEmbeddingProvider.is_model_supported("custom-embed:latest") is True
        assert OllamaEmbeddingProvider.is_model_supported("plaincustom") is False
        assert OllamaEmbeddingProvider.supports_concurrent_requests() is False

        client = OllamaEmbeddingProvider.create_client("custom-embed:latest", base_url="http://ollama.local")
        assert isinstance(client, OllamaClient)
        assert client.model == "custom-embed:latest"
