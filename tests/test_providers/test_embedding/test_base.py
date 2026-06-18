"""Tests for the base embedding client behavior."""

import pytest

from sciread.providers.embedding.base import BaseEmbeddingClient
from sciread.providers.embedding.base import cosine_similarity


class DummyEmbeddingClient(BaseEmbeddingClient):
    """Deterministic embedding client for testing base behavior."""

    def __init__(self, *, fail_batch: bool = False, bad_batch_length: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.fail_batch = fail_batch
        self.bad_batch_length = bad_batch_length
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        self.batch_calls.append(texts)
        if self.fail_batch:
            return [None] * len(texts)
        if self.bad_batch_length:
            return [[1.0, 0.0]]  # Wrong length on purpose
        return [[1.0, 0.0] for _ in texts]

    def _get_single_embedding(self, text: str) -> list[float] | None:
        self.single_calls.append(text)
        if text == "fail":
            return None
        return [0.0, 1.0]


class TestCosineSimilarity:
    """Tests for the shared cosine_similarity helper."""

    def test_identical_vectors_return_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_invalid_input_returns_zero(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_zero_magnitude_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_non_numeric_values_return_zero(self):
        assert cosine_similarity([1.0, "x"], [1.0, 0.0]) == 0.0  # type: ignore[list-item]


class TestBaseEmbeddingClient:
    """Tests for shared caching and batching behavior."""

    def test_empty_texts_return_empty_list(self):
        client = DummyEmbeddingClient(model="dummy")
        assert client.get_embeddings([]) == []

    def test_cache_disabled_skips_cache(self):
        client = DummyEmbeddingClient(model="dummy", cache_embeddings=False)
        result = client.get_embeddings(["hello"])
        assert result == [[1.0, 0.0]]
        assert len(client.embedding_cache) == 0

    def test_cache_hit_avoids_remote_call(self):
        client = DummyEmbeddingClient(model="dummy")
        first = client.get_embeddings(["hello"])
        second = client.get_embeddings(["hello"])
        assert first == second
        assert len(client.batch_calls) == 1

    def test_deduplicates_repeated_texts_in_one_request(self):
        client = DummyEmbeddingClient(model="dummy")
        result = client.get_embeddings(["a", "b", "a", "b"])
        assert result == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert len(client.batch_calls) == 1
        assert client.batch_calls[0] == ["a", "b"]

    def test_batch_size_zero_treated_as_one(self):
        client = DummyEmbeddingClient(model="dummy")
        result = client.get_embeddings(["a", "b", "c"], batch_size=0)
        assert len(result) == 3
        assert all(len(batch) == 1 for batch in client.batch_calls)

    def test_batch_length_mismatch_falls_back_to_single_calls(self):
        client = DummyEmbeddingClient(model="dummy", bad_batch_length=True)
        result = client.get_embeddings(["a", "b"])
        assert len(result) == 2
        assert client.single_calls == ["a", "b"]

    def test_batch_failure_falls_back_to_zero_vectors(self):
        client = DummyEmbeddingClient(model="dummy", fail_batch=True, embedding_dimension=2)
        result = client.get_embeddings(["a"])
        assert result == [[0.0, 0.0]]

    def test_get_embedding_uses_cache(self):
        client = DummyEmbeddingClient(model="dummy")
        first = client.get_embedding("hello")
        second = client.get_embedding("hello")
        assert first == second
        assert len(client.single_calls) == 1

    def test_get_embedding_returns_none_for_failed_single(self):
        client = DummyEmbeddingClient(model="dummy")
        assert client.get_embedding("fail") is None

    def test_embedding_dimension_captured_from_first_result(self):
        client = DummyEmbeddingClient(model="dummy")
        client.get_embeddings(["hello"])
        assert client.embedding_dimension == 2

    def test_fallback_embedding_uses_configured_dimension(self):
        client = DummyEmbeddingClient(model="dummy", embedding_dimension=4)
        client.fail_batch = True
        result = client.get_embeddings(["hello"])
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_cache_stats_reflect_state(self):
        client = DummyEmbeddingClient(model="dummy")
        client.get_embeddings(["a", "b"])
        stats = client.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["model"] == "dummy"
        assert stats["cache_enabled"] is True

    def test_clear_cache_empties_store(self):
        client = DummyEmbeddingClient(model="dummy")
        client.get_embeddings(["a"])
        client.clear_cache()
        assert len(client.embedding_cache) == 0

    def test_calculate_centroid_averages_vectors(self):
        client = DummyEmbeddingClient(model="dummy")
        centroid = client.calculate_centroid([[1.0, 0.0], [0.0, 2.0]])
        assert centroid == [0.5, 1.0]

    def test_calculate_centroid_empty_returns_empty(self):
        client = DummyEmbeddingClient(model="dummy")
        assert client.calculate_centroid([]) == []
