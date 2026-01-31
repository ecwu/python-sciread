"""SiliconFlow embedding provider."""

import math
import os
from typing import Any
from typing import ClassVar

import requests

from ..logging_config import get_logger
from .base import BaseEmbeddingProvider


class SiliconFlowClient:
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
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        self.embedding_dimension = embedding_dimension
        self.embedding_cache: dict[str, list[float]] = {}
        self.logger = get_logger(__name__)

        if not self.api_key:
            self.api_key = os.getenv("SILICONFLOW_API_KEY")
            if not self.api_key:
                self.logger.warning("No SiliconFlow API key provided. Set SILICONFLOW_API_KEY environment variable.")

    def get_embeddings(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """
        Get embeddings for texts using SiliconFlow API.

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

            # Get embedding from SiliconFlow
            try:
                embedding = self._get_single_embedding(text)
                if embedding:
                    batch_embeddings.append(embedding)
                    if self.cache_embeddings:
                        self.embedding_cache[cache_key] = embedding
                else:
                    batch_embeddings.append([0.0] * self.embedding_dimension)  # Fallback
            except Exception:
                batch_embeddings.append([0.0] * self.embedding_dimension)  # Fallback

        return batch_embeddings

    def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text from SiliconFlow API."""
        # Check cache first
        cache_key = f"{self.model}:{hash(text)}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Get embedding from SiliconFlow
        embedding = self._get_single_embedding(text)
        if embedding and self.cache_embeddings:
            self.embedding_cache[cache_key] = embedding

        return embedding

    def _get_single_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text from SiliconFlow API using OpenAI format."""
        try:
            if not self.api_key:
                self.logger.warning("No API key available for SiliconFlow")
                return None

            url = f"{self.base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": text,
                "encoding_format": "float",
            }

            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0].get("embedding")
                    if embedding:
                        return embedding

            self.logger.warning(f"SiliconFlow API returned status {response.status_code}: {response.text}")
            return None
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
            "embedding_dimension": self.embedding_dimension,
        }


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
