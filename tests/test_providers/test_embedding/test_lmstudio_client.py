"""Test LM Studio embedding client integration."""

from unittest.mock import Mock
from unittest.mock import patch

import requests

from sciread.document import LMStudioClient
from sciread.providers.embedding.lmstudio import LMStudioEmbeddingProvider


class TestLMStudioClient:
    """Test cases for LMStudioClient."""

    def test_initialization_default_params(self):
        """Test client initialization with default parameters."""
        client = LMStudioClient()

        assert client.model == "embeddinggemma:latest"
        assert client.base_url == "http://localhost:1234/v1"
        assert client.api_key == "lm_studio"
        assert client.timeout == 30
        assert client.cache_embeddings is True
        assert client.embedding_dimension is None
        assert len(client.embedding_cache) == 0

    def test_initialization_custom_params(self):
        """Test client initialization with custom parameters."""
        client = LMStudioClient(
            model="custom-model",
            base_url="http://localhost:9999/v1",
            api_key="custom-key",
            timeout=60,
            cache_embeddings=False,
            embedding_dimension=1024,
        )

        assert client.model == "custom-model"
        assert client.base_url == "http://localhost:9999/v1"
        assert client.api_key == "custom-key"
        assert client.timeout == 60
        assert client.cache_embeddings is False
        assert client.embedding_dimension == 1024

    @patch("requests.post")
    def test_get_batch_embeddings_success_restores_response_order(self, mock_post):
        """Test successful batch embedding retrieval with OpenAI-compatible response ordering."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        client = LMStudioClient(api_key="lm_studio")
        embeddings = client._get_batch_embeddings(["text1", "text2"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_post.assert_called_once_with(
            "http://localhost:1234/v1/embeddings",
            json={
                "model": "embeddinggemma:latest",
                "input": ["text1", "text2"],
                "encoding_format": "float",
            },
            headers={
                "Authorization": "Bearer lm_studio",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

    @patch("requests.post")
    def test_get_single_embedding_success(self, mock_post):
        """Test successful single embedding retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        client = LMStudioClient(api_key="lm_studio")
        embedding = client._get_single_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_get_single_embedding_failure(self, mock_post):
        """Test failed single embedding retrieval."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response

        client = LMStudioClient(api_key="lm_studio")
        embedding = client._get_single_embedding("test text")

        assert embedding is None

    @patch("requests.post")
    def test_get_embeddings_with_cache(self, mock_post):
        """Test batch embeddings with caching."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                    ]
                }
            ),
        )

        client = LMStudioClient(api_key="lm_studio", cache_embeddings=True)

        embeddings = client.get_embeddings(["text1", "text2"], batch_size=10)

        assert len(embeddings) == 2
        assert client.embedding_dimension == 3
        assert mock_post.call_count == 1

        mock_post.reset_mock()
        cached_embeddings = client.get_embeddings(["text1"], batch_size=1)

        assert cached_embeddings == [[0.1, 0.2, 0.3]]
        assert mock_post.call_count == 0

    @patch("requests.post")
    def test_batch_embeddings_restores_response_order(self, mock_post):
        """Out-of-order batch responses should be mapped back to input positions."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "data": [
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"index": 2, "embedding": [0.7, 0.8, 0.9]},
                    ]
                }
            ),
        )

        client = LMStudioClient(api_key="lm_studio")
        embeddings = client._get_batch_embeddings(["a", "b", "c"])

        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    @patch("requests.post")
    def test_batch_embeddings_falls_back_to_sequential_index(self, mock_post):
        """Entries without an explicit index should be placed sequentially."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "data": [
                        {"index": 2, "embedding": [0.7, 0.8, 0.9]},
                        {"embedding": [0.1, 0.2, 0.3]},  # Missing index -> fallback 1
                        {"index": 0, "embedding": [0.4, 0.5, 0.6]},
                    ]
                }
            ),
        )

        client = LMStudioClient(api_key="lm_studio")
        embeddings = client._get_batch_embeddings(["a", "b", "c"])

        assert embeddings == [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]

    @patch("requests.post")
    def test_batch_embeddings_returns_none_when_any_position_unfilled(self, mock_post):
        """A batch response that leaves any position empty should be rejected."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"index": 5, "embedding": [0.4, 0.5, 0.6]},  # Out of range
                    ]
                }
            ),
        )

        client = LMStudioClient(api_key="lm_studio")
        embeddings = client._get_batch_embeddings(["a", "b", "c"])

        assert embeddings == [None, None, None]

    @patch("requests.post")
    def test_batch_embeddings_exception_returns_none_list(self, mock_post):
        """Request exceptions should degrade to a list of None embeddings."""
        mock_post.side_effect = requests.RequestException("connection refused")

        client = LMStudioClient(api_key="lm_studio")
        embeddings = client._get_batch_embeddings(["a", "b"])

        assert embeddings == [None, None]

    @patch("requests.post")
    def test_single_embedding_exception_returns_none(self, mock_post):
        """Single embedding failures should return None."""
        mock_post.side_effect = RuntimeError("boom")

        client = LMStudioClient(api_key="lm_studio")
        assert client._get_single_embedding("hello") is None

    @patch("requests.post")
    def test_test_connection_returns_false_on_failure(self, mock_post):
        """Connection test should return False when the embedding call fails."""
        mock_post.return_value = Mock(status_code=500, text="error")

        client = LMStudioClient(api_key="lm_studio")
        assert client.test_connection() is False

    @patch("requests.post")
    def test_test_connection_returns_true_on_success(self, mock_post):
        """Connection test should return True when the embedding call succeeds."""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}),
        )

        client = LMStudioClient(api_key="lm_studio")
        assert client.test_connection() is True

    def test_cache_stats_include_embedding_dimension(self):
        client = LMStudioClient(api_key="lm_studio", embedding_dimension=768)
        stats = client.get_cache_stats()
        assert stats["embedding_dimension"] == 768


class TestLMStudioEmbeddingProvider:
    """Tests for the LM Studio embedding provider static API."""

    def test_get_supported_models_returns_known_models(self):
        models = LMStudioEmbeddingProvider.get_supported_models()
        assert "embeddinggemma:latest" in models
        assert "nomic-embed-text" in models

    def test_is_model_supported(self):
        assert LMStudioEmbeddingProvider.is_model_supported("embeddinggemma:latest") is True
        assert LMStudioEmbeddingProvider.is_model_supported("custom-model:latest") is True
        assert LMStudioEmbeddingProvider.is_model_supported("plainname") is False
        assert LMStudioEmbeddingProvider.is_model_supported("") is False

    def test_create_client_builds_lmstudio_client(self):
        client = LMStudioEmbeddingProvider.create_client("custom-model", base_url="http://test:1234/v1", timeout=60)
        assert isinstance(client, LMStudioClient)
        assert client.model == "custom-model"
        assert client.base_url == "http://test:1234/v1"
        assert client.timeout == 60

    def test_supports_concurrent_requests(self):
        assert LMStudioEmbeddingProvider.supports_concurrent_requests() is True
