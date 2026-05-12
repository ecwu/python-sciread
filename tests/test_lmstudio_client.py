"""Test LM Studio embedding client integration."""

from unittest.mock import Mock
from unittest.mock import patch

from sciread.document import LMStudioClient


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
