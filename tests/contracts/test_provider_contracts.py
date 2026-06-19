"""Provider interface contract tests."""

from __future__ import annotations

import os

import pytest
from tests.fakes import FakeEmbeddingClient
from tests.fakes import FakeRerankClient

from sciread.providers.embedding.base import BaseEmbeddingClient
from sciread.providers.embedding.factory import EmbeddingFactory
from sciread.providers.rerank.base import BaseRerankClient
from sciread.providers.rerank.factory import RerankFactory
from sciread.providers.rerank.siliconflow import SiliconFlowRerankClient

pytestmark = pytest.mark.contracts


def test_embedding_client_contract_batches_deduplicates_caches_and_falls_back() -> None:
    """Embedding clients should support batch calls, duplicate inputs, cache stats, and fallback vectors."""
    client = FakeEmbeddingClient(dimension=4, fail_for={"missing"})

    embeddings = client.get_embeddings(["alpha", "alpha", "missing"], batch_size=2)
    cached_embedding = client.get_embedding("alpha")
    stats = client.get_cache_stats()

    assert len(embeddings) == 3
    assert embeddings[0] == embeddings[1] == cached_embedding
    assert embeddings[2] == [0.0, 0.0, 0.0, 0.0]
    assert client.batch_calls == [["alpha", "missing"]]
    assert stats["model"] == "fake-embedding"
    assert stats["cache_enabled"] is True


def test_real_embedding_providers_create_base_clients_without_network_calls() -> None:
    """Provider factories should return BaseEmbeddingClient instances without contacting services."""
    supported = EmbeddingFactory.get_supported_providers()

    for provider_name, models in supported.items():
        model_name = next(iter(models))
        kwargs = {
            "base_url": "http://127.0.0.1:9",
            "timeout": 1,
            "cache_embeddings": False,
        }
        if provider_name != "ollama":
            kwargs["api_key"] = "test-key"

        client = EmbeddingFactory.create_client(
            f"{provider_name}/{model_name}",
            **kwargs,
        )
        assert isinstance(client, BaseEmbeddingClient)
        assert client.model == model_name


def test_rerank_client_contract_orders_limits_and_handles_empty_results() -> None:
    """Rerank clients should return ranked index/score pairs honoring top_n."""
    client = FakeRerankClient(scores=[0.2, 0.8, 0.5])

    results = client.rerank("query", ["doc-a", "doc-b", "doc-c"], top_n=2)
    empty_results = FakeRerankClient(return_empty=True).rerank("query", ["doc-a"], top_n=1)

    assert [result.index for result in results] == [1, 2]
    assert [result.relevance_score for result in results] == [0.8, 0.5]
    assert [result.document for result in results] == ["doc-b", "doc-c"]
    assert empty_results == []


def test_real_rerank_provider_creates_base_client_without_network_calls() -> None:
    """Rerank factories should create clients that expose the base rerank interface."""
    model_identifier = RerankFactory.list_all_supported_models()[0]

    client = RerankFactory.create_client(
        model_identifier,
        api_key="",
        base_url="http://127.0.0.1:9",
        timeout=1,
    )

    assert isinstance(client, BaseRerankClient)
    assert client.rerank("query", ["document"], top_n=1) == []


@pytest.mark.live
def test_live_siliconflow_rerank_contract() -> None:
    """Optional live contract test for SiliconFlow rerank response shape."""
    if os.getenv("SCIREAD_RUN_LIVE") != "1" or not os.getenv("SILICONFLOW_API_KEY"):
        pytest.skip("Set SCIREAD_RUN_LIVE=1 and SILICONFLOW_API_KEY to run live rerank contract tests.")

    client = SiliconFlowRerankClient(timeout=30)
    results = client.rerank("retrieval", ["retrieval system", "unrelated cooking note"], top_n=1)

    assert len(results) == 1
    assert results[0].index in {0, 1}
    assert isinstance(results[0].relevance_score, float)
