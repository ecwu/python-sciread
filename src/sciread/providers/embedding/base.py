"""Base provider and client helpers for embedding models."""

from __future__ import annotations

import hashlib
import math
from abc import ABC
from abc import abstractmethod
from typing import Any


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity while safely handling invalid input."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
    except Exception:
        return 0.0

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


class BaseEmbeddingClient(ABC):
    """Shared embedding client behavior for caching and batched fetches."""

    default_fallback_dimension = 768

    def __init__(
        self,
        *,
        model: str,
        timeout: int = 30,
        cache_embeddings: bool = True,
        embedding_dimension: int | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        self.embedding_dimension = embedding_dimension
        self.embedding_cache: dict[str, list[float]] = {}

    def _build_cache_key(self, text: str) -> str:
        """Return a stable cache key for one text input."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}:{digest}"

    def _cache_get(self, text: str) -> list[float] | None:
        """Get a cached embedding when caching is enabled."""
        if not self.cache_embeddings:
            return None
        return self.embedding_cache.get(self._build_cache_key(text))

    def _cache_set(self, text: str, embedding: list[float] | None) -> None:
        """Store an embedding in cache when enabled."""
        if not self.cache_embeddings or not embedding:
            return
        self.embedding_cache[self._build_cache_key(text)] = embedding

    def _normalize_embedding(self, embedding: list[float] | None) -> list[float] | None:
        """Capture embedding dimension from successful responses."""
        if not embedding:
            return None
        if self.embedding_dimension is None:
            self.embedding_dimension = len(embedding)
        return embedding

    def _fallback_embedding(self) -> list[float]:
        """Return a zero-vector fallback matching the configured dimension."""
        dimension = self.embedding_dimension or self.default_fallback_dimension
        return [0.0] * dimension

    def get_embeddings(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Get embeddings for texts while deduplicating repeated inputs."""
        if not texts:
            return []

        resolved_embeddings: list[list[float] | None] = [None] * len(texts)
        pending_positions: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            cached_embedding = self._cache_get(text)
            if cached_embedding is not None:
                resolved_embeddings[index] = cached_embedding
                continue
            pending_positions.setdefault(text, []).append(index)

        unique_pending = list(pending_positions.keys())
        if unique_pending:
            effective_batch_size = max(1, batch_size)
            for offset in range(0, len(unique_pending), effective_batch_size):
                batch = unique_pending[offset : offset + effective_batch_size]
                batch_embeddings = self._get_batch_embeddings(batch)

                if len(batch_embeddings) != len(batch):
                    batch_embeddings = [self._get_single_embedding(text) for text in batch]

                for text, embedding in zip(batch, batch_embeddings, strict=True):
                    normalized_embedding = self._normalize_embedding(embedding)
                    if normalized_embedding is not None:
                        self._cache_set(text, normalized_embedding)
                    for position in pending_positions[text]:
                        resolved_embeddings[position] = normalized_embedding

        return [embedding if embedding is not None else self._fallback_embedding() for embedding in resolved_embeddings]

    def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text input."""
        cached_embedding = self._cache_get(text)
        if cached_embedding is not None:
            return cached_embedding

        embedding = self._normalize_embedding(self._get_single_embedding(text))
        if embedding is not None:
            self._cache_set(text, embedding)
        return embedding

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Fetch one batch of embeddings; subclasses can override with native batching."""
        return [self._get_single_embedding(text) for text in texts]

    @abstractmethod
    def _get_single_embedding(self, text: str) -> list[float] | None:
        """Fetch one embedding from the backing provider."""

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Proxy shared cosine similarity for backwards compatibility."""
        return cosine_similarity(vec1, vec2)

    def calculate_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate centroid of embeddings."""
        if not embeddings:
            return []

        centroid = [0.0] * len(embeddings[0])
        for embedding in embeddings:
            for index, value in enumerate(embedding):
                centroid[index] += value

        return [value / len(embeddings) for value in centroid]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self.embedding_cache),
            "model": self.model,
            "cache_enabled": self.cache_embeddings,
        }


class BaseEmbeddingProvider(ABC):
    """Base class for embedding model providers."""

    @staticmethod
    @abstractmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported models for this provider."""

    @staticmethod
    @abstractmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by this provider."""

    @staticmethod
    @abstractmethod
    def create_client(model_name: str, **kwargs: Any):
        """Create an embedding client instance."""

    @staticmethod
    @abstractmethod
    def supports_concurrent_requests() -> bool:
        """Check if this provider supports concurrent requests."""
