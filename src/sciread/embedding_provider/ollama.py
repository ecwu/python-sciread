"""Ollama embedding provider."""

import math
from typing import Any
from typing import Optional

import requests

from ..logging_config import get_logger
from .base import BaseEmbeddingProvider


class OllamaClient:
    """Client for interacting with Ollama API for embeddings."""

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        cache_embeddings: bool = True,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        self.embedding_cache: dict[str, list[float]] = {}
        self.logger = get_logger(__name__)

    def get_embeddings(
        self, texts: list[str], batch_size: int = 10
    ) -> list[list[float]]:
        """
        Get embeddings for texts using Ollama API.

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

            # Get embedding from Ollama
            try:
                embedding = self._get_single_embedding(text)
                if embedding:
                    batch_embeddings.append(embedding)
                    if self.cache_embeddings:
                        self.embedding_cache[cache_key] = embedding
                else:
                    batch_embeddings.append([0.0] * 768)  # Fallback
            except Exception:
                batch_embeddings.append([0.0] * 768)  # Fallback

        return batch_embeddings

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for a single text from Ollama API."""
        # Check cache first
        cache_key = f"{self.model}:{hash(text)}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Get embedding from Ollama
        embedding = self._get_single_embedding(text)
        if embedding and self.cache_embeddings:
            self.embedding_cache[cache_key] = embedding

        return embedding

    def _get_single_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for a single text from Ollama API."""
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.model, "prompt": text}

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]

            return None
        except Exception as e:
            self.logger.warning(f"Failed to get embedding from Ollama: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Failed to connect to Ollama: {e}")
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
        }


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local embedding models."""

    # Supported models - common Ollama embedding models
    SUPPORTED_MODELS = {
        "nomic-embed-text": "Nomic Embed Text - High quality text embeddings (768 dim)",
        "mxbai-embed-large": "MixedBread AI Large - High performance embeddings (1024 dim)",
        "all-minilm": "All MiniLM - Fast and efficient embeddings (384 dim)",
        "embeddinggemma:latest": "Embedding Gemma - Google's embedding model (768 dim)",
        "snowflake-arctic-embed": "Snowflake Arctic Embed - Enterprise embeddings (1024 dim)",
    }

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported Ollama embedding models."""
        return OllamaEmbeddingProvider.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by Ollama provider."""
        # Ollama supports custom models, so we're permissive here
        # Return True for known models, or if it looks like an Ollama model name
        if model_name in OllamaEmbeddingProvider.SUPPORTED_MODELS:
            return True
        # Allow any model name that might be a valid Ollama model
        # (e.g., "nomic-embed-text:latest", custom models)
        return ":" in model_name or "-" in model_name

    @staticmethod
    def create_client(model_name: str, **kwargs: Any) -> OllamaClient:
        """Create an Ollama embedding client.

        Args:
            model_name: Name of the Ollama model
            **kwargs: Additional arguments (base_url, timeout, cache_embeddings)

        Returns:
            OllamaClient instance
        """
        return OllamaClient(model=model_name, **kwargs)

    @staticmethod
    def supports_concurrent_requests() -> bool:
        """Ollama typically runs locally and handles requests sequentially.

        Returns:
            False - Ollama should use sequential requests
        """
        return False
