"""Tests for rerank provider factory helpers."""

from unittest.mock import patch

import pytest

from sciread.rerank_provider.factory import InvalidRerankIdentifierError
from sciread.rerank_provider.factory import RerankFactory
from sciread.rerank_provider.factory import UnsupportedRerankModelError
from sciread.rerank_provider.factory import get_rerank_client


class DummySiliconFlowRerankProvider:
    """Stub rerank provider with slash-containing model names."""

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        return model_name in {"nested/model", "plain-model"}

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        return {
            "nested/model": "Nested rerank model",
            "plain-model": "Plain rerank model",
        }

    @staticmethod
    def create_client(model_name: str, **kwargs: object) -> dict[str, object]:
        return {"provider": "siliconflow", "model": model_name, "kwargs": kwargs}


@pytest.fixture
def stub_providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Replace the rerank provider registry with deterministic test doubles."""
    providers = {"siliconflow": DummySiliconFlowRerankProvider}
    monkeypatch.setattr(RerankFactory, "PROVIDERS", providers)
    return providers


def test_parse_rerank_identifier_rejects_empty_values(stub_providers: dict[str, type]) -> None:
    """Blank rerank identifiers should be rejected."""
    del stub_providers

    with pytest.raises(InvalidRerankIdentifierError, match="cannot be empty"):
        RerankFactory.parse_rerank_identifier("  ")


def test_parse_rerank_identifier_preserves_explicit_provider(stub_providers: dict[str, type]) -> None:
    """Explicit provider prefixes should be parsed directly."""
    del stub_providers

    provider, model = RerankFactory.parse_rerank_identifier("siliconflow/plain-model")

    assert provider == "siliconflow"
    assert model == "plain-model"


def test_parse_rerank_identifier_inferrs_provider_for_nested_model(stub_providers: dict[str, type]) -> None:
    """Model names containing slashes should still map to SiliconFlow."""
    del stub_providers

    provider, model = RerankFactory.parse_rerank_identifier("nested/model")

    assert provider == "siliconflow"
    assert model == "nested/model"


def test_create_client_rejects_unsupported_model(stub_providers: dict[str, type]) -> None:
    """Unsupported rerank models should raise before client construction."""
    del stub_providers

    with pytest.raises(UnsupportedRerankModelError, match="not supported by provider 'siliconflow'"):
        RerankFactory.create_client("siliconflow/unknown-model")


def test_get_rerank_client_delegates_to_factory(stub_providers: dict[str, type]) -> None:
    """Convenience helper should forward arguments to the factory."""
    del stub_providers

    with patch.object(RerankFactory, "create_client", return_value="client") as mock_create_client:
        result = get_rerank_client("siliconflow/plain-model", timeout=3)

    assert result == "client"
    mock_create_client.assert_called_once_with("siliconflow/plain-model", timeout=3)


def test_real_siliconflow_reranker_identifier_resolves_to_provider() -> None:
    """Bare SiliconFlow reranker model names should resolve to the SiliconFlow provider."""
    provider, model = RerankFactory.parse_rerank_identifier("BAAI/bge-reranker-v2-m3")

    assert provider == "siliconflow"
    assert model == "BAAI/bge-reranker-v2-m3"
