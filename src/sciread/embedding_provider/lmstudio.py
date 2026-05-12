"""LM Studio embedding provider."""

from typing import Any
from typing import ClassVar

import requests

from ..platform.logging import get_logger
from .base import BaseEmbeddingClient
from .base import BaseEmbeddingProvider


class LMStudioClient(BaseEmbeddingClient):
    """Client for interacting with LM Studio's OpenAI-compatible embeddings API."""

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm_studio",
        timeout: int = 30,
        cache_embeddings: bool = True,
        embedding_dimension: int | None = None,
    ):
        """
        Initialize LM Studio client.

        Args:
            model: LM Studio model name for embeddings.
            base_url: LM Studio OpenAI-compatible API base URL.
            api_key: Placeholder API key for OpenAI-compatible local requests.
            timeout: Request timeout in seconds.
            cache_embeddings: Whether to cache embeddings.
            embedding_dimension: Dimension of the embedding vectors.
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

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Get embeddings for a batch of texts using the OpenAI-compatible API."""
        try:
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

            self.logger.warning(f"LM Studio API returned status {response.status_code}: {response.text}")
            return [None] * len(texts)
        except Exception as e:
            self.logger.warning(f"Failed to get batch embeddings from LM Studio: {e}")
            return [None] * len(texts)

    def _get_single_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text from LM Studio."""
        try:
            return self._get_batch_embeddings([text])[0]
        except Exception as e:
            self.logger.warning(f"Failed to get embedding from LM Studio: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to LM Studio with a small embedding request."""
        try:
            return self._get_single_embedding("test") is not None
        except Exception as e:
            self.logger.warning(f"Failed to connect to LM Studio: {e}")
            return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        stats = super().get_cache_stats()
        stats["embedding_dimension"] = self.embedding_dimension
        return stats


class LMStudioEmbeddingProvider(BaseEmbeddingProvider):
    """LM Studio embedding provider for local embedding models."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "embeddinggemma:latest": "Embedding Gemma - Google's embedding model",
        "nomic-embed-text": "Nomic Embed Text - High quality text embeddings",
        "mxbai-embed-large": "MixedBread AI Large - High performance embeddings",
        "all-minilm": "All MiniLM - Fast and efficient embeddings",
    }

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported LM Studio embedding model examples."""
        return LMStudioEmbeddingProvider.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model name looks like a local embedding model."""
        if not model_name or not model_name.strip():
            return False
        if model_name in LMStudioEmbeddingProvider.SUPPORTED_MODELS:
            return True
        return ":" in model_name or "-" in model_name

    @staticmethod
    def create_client(model_name: str, **kwargs: Any) -> LMStudioClient:
        """Create an LM Studio embedding client."""
        return LMStudioClient(model=model_name, **kwargs)

    @staticmethod
    def supports_concurrent_requests() -> bool:
        """LM Studio's OpenAI-compatible API can accept batched requests."""
        return True
