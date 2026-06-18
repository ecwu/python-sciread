"""Tests for SiliconFlow rerank client."""

from unittest.mock import Mock
from unittest.mock import patch

from sciread.providers.rerank import SiliconFlowRerankClient


def test_initialization_default_params() -> None:
    """The rerank client should use SiliconFlow defaults."""
    client = SiliconFlowRerankClient(api_key="test-key")

    assert client.model == "BAAI/bge-reranker-v2-m3"
    assert client.base_url == "https://api.siliconflow.cn/v1"
    assert client.timeout == 30
    assert client.api_key == "test-key"


@patch.dict("os.environ", {"SILICONFLOW_API_KEY": "env-test-key"})
def test_api_key_from_environment() -> None:
    """The rerank client should read SILICONFLOW_API_KEY."""
    client = SiliconFlowRerankClient()

    assert client.api_key == "env-test-key"


@patch("requests.post")
def test_rerank_success(mock_post: Mock) -> None:
    """Successful SiliconFlow rerank responses should be normalized."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "rerank-1",
        "results": [
            {"index": 1, "document": {"text": "banana"}, "relevance_score": 0.9},
            {"index": 0, "document": {"text": "apple"}, "relevance_score": 0.2},
        ],
    }
    mock_post.return_value = mock_response

    client = SiliconFlowRerankClient(api_key="test-key", return_documents=True)
    results = client.rerank("fruit", ["apple", "banana"], top_n=2)

    assert [result.index for result in results] == [1, 0]
    assert [result.relevance_score for result in results] == [0.9, 0.2]
    assert results[0].document == "banana"
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "fruit",
        "documents": ["apple", "banana"],
        "return_documents": True,
        "top_n": 2,
    }


@patch("requests.post")
def test_rerank_failure_returns_empty(mock_post: Mock) -> None:
    """Failed rerank responses should degrade to an empty result set."""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response

    client = SiliconFlowRerankClient(api_key="bad-key")

    assert client.rerank("fruit", ["apple"]) == []


@patch.dict("os.environ", {}, clear=True)
def test_rerank_without_api_key_returns_empty() -> None:
    """Missing API keys should not attempt a remote request."""
    client = SiliconFlowRerankClient(api_key=None)

    assert client.rerank("fruit", ["apple"]) == []
