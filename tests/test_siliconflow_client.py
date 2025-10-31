"""Test SiliconFlow client integration."""

import pytest
from unittest.mock import Mock, patch
from sciread.document import SiliconFlowClient


class TestSiliconFlowClient:
    """Test cases for SiliconFlowClient."""

    def test_initialization_default_params(self):
        """Test client initialization with default parameters."""
        client = SiliconFlowClient()

        assert client.model == "Qwen/Qwen3-Embedding-8B"
        assert client.base_url == "https://api.siliconflow.cn/v1"
        assert client.timeout == 30
        assert client.cache_embeddings is True
        assert client.embedding_dimension == 4096
        assert len(client.embedding_cache) == 0

    def test_initialization_custom_params(self):
        """Test client initialization with custom parameters."""
        client = SiliconFlowClient(
            model="custom-model",
            base_url="https://custom.api.com/v1",
            api_key="test-key",
            timeout=60,
            cache_embeddings=False,
            embedding_dimension=1024,
        )

        assert client.model == "custom-model"
        assert client.base_url == "https://custom.api.com/v1"
        assert client.api_key == "test-key"
        assert client.timeout == 60
        assert client.cache_embeddings is False
        assert client.embedding_dimension == 1024

    @patch.dict("os.environ", {"SILICONFLOW_API_KEY": "env-test-key"})
    def test_api_key_from_environment(self):
        """Test API key loading from environment variable."""
        client = SiliconFlowClient()
        assert client.api_key == "env-test-key"

    def test_cache_functionality(self):
        """Test embedding cache functionality."""
        client = SiliconFlowClient(cache_embeddings=True)

        # Add to cache
        cache_key = f"{client.model}:{hash('test')}"
        test_embedding = [1.0, 2.0, 3.0]
        client.embedding_cache[cache_key] = test_embedding

        # Verify cache
        assert len(client.embedding_cache) == 1
        assert client.embedding_cache[cache_key] == test_embedding

        # Clear cache
        client.clear_cache()
        assert len(client.embedding_cache) == 0

    def test_cache_stats(self):
        """Test cache statistics retrieval."""
        client = SiliconFlowClient(
            model="test-model",
            cache_embeddings=True,
            embedding_dimension=2048,
        )

        stats = client.get_cache_stats()

        assert stats["cache_size"] == 0
        assert stats["model"] == "test-model"
        assert stats["cache_enabled"] is True
        assert stats["embedding_dimension"] == 2048

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        client = SiliconFlowClient()

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert abs(client.cosine_similarity(vec1, vec2) - 1.0) < 1e-6

        # Orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(client.cosine_similarity(vec1, vec2)) < 1e-6

        # Opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert abs(client.cosine_similarity(vec1, vec2) - (-1.0)) < 1e-6

        # Different lengths (should return 0)
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert client.cosine_similarity(vec1, vec2) == 0.0

    def test_calculate_centroid(self):
        """Test centroid calculation."""
        client = SiliconFlowClient()

        embeddings = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]

        centroid = client.calculate_centroid(embeddings)

        assert len(centroid) == 3
        assert abs(centroid[0] - 4.0) < 1e-6
        assert abs(centroid[1] - 5.0) < 1e-6
        assert abs(centroid[2] - 6.0) < 1e-6

        # Empty list
        assert client.calculate_centroid([]) == []

    @patch("requests.post")
    def test_get_single_embedding_success(self, mock_post):
        """Test successful single embedding retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        client = SiliconFlowClient(api_key="test-key")
        embedding = client._get_single_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_get_single_embedding_failure(self, mock_post):
        """Test failed single embedding retrieval."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        client = SiliconFlowClient(api_key="invalid-key")
        embedding = client._get_single_embedding("test text")

        assert embedding is None

    @patch("requests.post")
    def test_get_embeddings_with_cache(self, mock_post):
        """Test batch embeddings with caching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        client = SiliconFlowClient(api_key="test-key", cache_embeddings=True)

        # First call should hit API
        texts = ["text1", "text2"]
        embeddings = client.get_embeddings(texts, batch_size=1)

        assert len(embeddings) == 2
        assert mock_post.call_count == 2

        # Second call with same text should use cache
        mock_post.reset_mock()
        embeddings2 = client.get_embeddings(["text1"], batch_size=1)

        assert len(embeddings2) == 1
        assert mock_post.call_count == 0  # Should not call API

    def test_get_embeddings_fallback_on_error(self):
        """Test fallback embeddings on error."""
        client = SiliconFlowClient(api_key=None, embedding_dimension=100)

        # Should return fallback embeddings (zeros)
        texts = ["text1", "text2"]
        embeddings = client.get_embeddings(texts)

        assert len(embeddings) == 2
        assert all(len(emb) == 100 for emb in embeddings)
        assert all(all(v == 0.0 for v in emb) for emb in embeddings)

    @patch("requests.post")
    def test_test_connection_success(self, mock_post):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        client = SiliconFlowClient(api_key="test-key")
        assert client.test_connection() is True

    @patch("requests.post")
    def test_test_connection_failure(self, mock_post):
        """Test failed connection test."""
        mock_post.side_effect = Exception("Connection error")

        client = SiliconFlowClient(api_key="test-key")
        assert client.test_connection() is False

    def test_test_connection_no_api_key(self):
        """Test connection test without API key."""
        client = SiliconFlowClient(api_key=None)
        assert client.test_connection() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
