"""SiliconFlow embedding provider."""

import os
from typing import Any
from typing import ClassVar

import requests

from ..platform.logging import get_logger
from .base import BaseEmbeddingClient
from .base import BaseEmbeddingProvider


class SiliconFlowClient(BaseEmbeddingClient):
    """Client for interacting with SiliconFlow API (OpenAI-compatible) for embeddings."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key: str | None = None,
        timeout: int = 30,
        cache_embeddings: bool = True,
        embedding_dimension: int = 4096,
    ):
        """
        Initialize SiliconFlow client.

        Args:
            model: SiliconFlow model name for embeddings
            base_url: SiliconFlow API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings
            embedding_dimension: Dimension of the embedding vectors
        """
        super().__init__(
            model=model,
            timeout=timeout,
            cache_embeddings=cache_embeddings,
            embedding_dimension=embedding_dimension,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.logger = get_logger(__name__)

        if not self.api_key:
            self.api_key = os.getenv("SILICONFLOW_API_KEY")
            if not self.api_key:
                self.logger.warning("No SiliconFlow API key provided. Set SILICONFLOW_API_KEY environment variable.")

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Get embeddings for a batch of texts using the OpenAI-compatible batch API."""
        try:
            if not self.api_key:
                self.logger.warning("No API key available for SiliconFlow")
                return [None] * len(texts)

            url = f"{self.base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            }

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                raw_embeddings = data.get("data", [])
                batch_embeddings: list[list[float] | None] = [None] * len(texts)
                for fallback_index, item in enumerate(raw_embeddings):
                    index = item.get("index", fallback_index)
                    embedding = item.get("embedding")
                    if isinstance(index, int) and 0 <= index < len(texts):
                        batch_embeddings[index] = embedding
                if raw_embeddings and all(embedding is not None for embedding in batch_embeddings):
                    return batch_embeddings

            self.logger.warning(f"SiliconFlow API returned status {response.status_code}: {response.text}")
            return [None] * len(texts)
        except Exception as e:
            self.logger.warning(f"Failed to get batch embeddings from SiliconFlow: {e}")
            return [None] * len(texts)

    def _get_single_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text from SiliconFlow API using OpenAI format."""
        try:
            return self._get_batch_embeddings([text])[0]
        except Exception as e:
            self.logger.warning(f"Failed to get embedding from SiliconFlow: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to SiliconFlow server."""
        try:
            if not self.api_key:
                self.logger.warning("No API key available for testing connection")
                return False

            # Test with a simple embedding request
            test_text = "test"
            embedding = self._get_single_embedding(test_text)
            return embedding is not None
        except Exception as e:
            self.logger.warning(f"Failed to connect to SiliconFlow: {e}")
            return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        stats = super().get_cache_stats()
        stats["embedding_dimension"] = self.embedding_dimension
        return stats


class SiliconFlowEmbeddingProvider(BaseEmbeddingProvider):
    """SiliconFlow embedding provider for cloud-based embedding models."""

    # Supported models - SiliconFlow embedding models
    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "Qwen/Qwen3-Embedding-8B": "Qwen3 Embedding 8B - High quality embeddings (4096 dim)",
        "BAAI/bge-large-zh-v1.5": "BGE Large Chinese - Chinese text embeddings (1024 dim)",
        "BAAI/bge-large-en-v1.5": "BGE Large English - English text embeddings (1024 dim)",
        "BAAI/bge-m3": "BGE M3 - Multilingual embeddings (1024 dim)",
    }

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported SiliconFlow embedding models."""
        return SiliconFlowEmbeddingProvider.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by SiliconFlow provider."""
        return model_name in SiliconFlowEmbeddingProvider.SUPPORTED_MODELS

    @staticmethod
    def create_client(model_name: str, **kwargs: Any) -> SiliconFlowClient:
        """Create a SiliconFlow embedding client.

        Args:
            model_name: Name of the SiliconFlow model
            **kwargs: Additional arguments (api_key, base_url, timeout, cache_embeddings, embedding_dimension)

        Returns:
            SiliconFlowClient instance
        """
        # Set default embedding dimensions based on model
        if "embedding_dimension" not in kwargs:
            if "Qwen3" in model_name:
                kwargs["embedding_dimension"] = 4096
            elif "bge" in model_name:
                kwargs["embedding_dimension"] = 1024

        return SiliconFlowClient(model=model_name, **kwargs)

    @staticmethod
    def supports_concurrent_requests() -> bool:
        """SiliconFlow is a cloud API that supports concurrent requests.

        Returns:
            True - SiliconFlow supports parallel requests
        """
        return True
