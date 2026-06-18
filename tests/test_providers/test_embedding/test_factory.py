"""Tests for embedding provider factory helpers."""

from unittest.mock import patch

import pytest

from sciread.providers.embedding.factory import EmbeddingFactory
from sciread.providers.embedding.factory import InvalidEmbeddingIdentifierError
from sciread.providers.embedding.factory import UnsupportedEmbeddingModelError
from sciread.providers.embedding.factory import get_embedding_client
from sciread.providers.embedding.siliconflow import SiliconFlowClient


class DummyLMStudioProvider:
    """Stub LM Studio provider for deterministic embedding factory tests."""

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        return model_name in {"dummy-embed", "fallback-model"}

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        return {
            "dummy-embed": "Dummy LM Studio embedding model",
            "fallback-model": "Fallback LM Studio embedding model",
        }

    @staticmethod
    def create_client(model_name: str, **kwargs: object) -> dict[str, object]:
        return {"provider": "lmstudio", "model": model_name, "kwargs": kwargs}

    @staticmethod
    def supports_concurrent_requests() -> bool:
        return True


class DummyOllamaProvider:
    """Stub provider for deterministic embedding factory tests."""

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        return model_name in {"ollama-embed", "ollama-fallback"}

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        return {
            "ollama-embed": "Dummy Ollama embedding model",
            "ollama-fallback": "Fallback Ollama embedding model",
        }

    @staticmethod
    def create_client(model_name: str, **kwargs: object) -> dict[str, object]:
        return {"provider": "ollama", "model": model_name, "kwargs": kwargs}

    @staticmethod
    def supports_concurrent_requests() -> bool:
        return False


class DummySiliconFlowProvider:
    """Stub provider with slash-containing model names."""

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        return model_name in {"nested/model", "provider-model"}

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        return {
            "nested/model": "Model with slash in name",
            "provider-model": "Explicit provider model",
        }

    @staticmethod
    def create_client(model_name: str, **kwargs: object) -> dict[str, object]:
        return {"provider": "siliconflow", "model": model_name, "kwargs": kwargs}

    @staticmethod
    def supports_concurrent_requests() -> bool:
        return True


@pytest.fixture
def stub_providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Replace provider registry with deterministic test doubles."""
    providers = {
        "lmstudio": DummyLMStudioProvider,
        "ollama": DummyOllamaProvider,
        "siliconflow": DummySiliconFlowProvider,
    }
    monkeypatch.setattr(EmbeddingFactory, "PROVIDERS", providers)
    return providers


def test_parse_embedding_identifier_rejects_empty_values(stub_providers: dict[str, type]) -> None:
    """Blank embedding identifiers should be rejected."""
    del stub_providers

    with pytest.raises(InvalidEmbeddingIdentifierError, match="cannot be empty"):
        EmbeddingFactory.parse_embedding_identifier("  ")


def test_parse_embedding_identifier_preserves_explicit_provider(stub_providers: dict[str, type]) -> None:
    """Explicit provider prefixes should be parsed directly."""
    del stub_providers

    provider, model = EmbeddingFactory.parse_embedding_identifier("siliconflow/provider-model")

    assert provider == "siliconflow"
    assert model == "provider-model"

    provider, model = EmbeddingFactory.parse_embedding_identifier("ollama/ollama-embed")

    assert provider == "ollama"
    assert model == "ollama-embed"


def test_parse_embedding_identifier_inferrs_provider_for_nested_model(stub_providers: dict[str, type]) -> None:
    """Model names that contain slashes should still map to the right provider."""
    del stub_providers

    provider, model = EmbeddingFactory.parse_embedding_identifier("nested/model")

    assert provider == "siliconflow"
    assert model == "nested/model"


def test_parse_embedding_identifier_rejects_invalid_provider_format(stub_providers: dict[str, type]) -> None:
    """Missing provider or model segments should be rejected."""
    del stub_providers

    with pytest.raises(InvalidEmbeddingIdentifierError, match="Invalid embedding identifier format"):
        EmbeddingFactory.parse_embedding_identifier("provider/")

    with pytest.raises(InvalidEmbeddingIdentifierError, match="Invalid embedding identifier format"):
        EmbeddingFactory.parse_embedding_identifier("/model")


def test_parse_embedding_identifier_defaults_to_lmstudio_for_unknown_plain_model(stub_providers: dict[str, type]) -> None:
    """Unknown bare model names should fall back to the default local provider."""
    del stub_providers

    provider, model = EmbeddingFactory.parse_embedding_identifier("custom-model")

    assert provider == "lmstudio"
    assert model == "custom-model"


def test_get_provider_class_rejects_unknown_provider(stub_providers: dict[str, type]) -> None:
    """Provider lookup should show supported values on failure."""
    del stub_providers

    with pytest.raises(UnsupportedEmbeddingModelError, match="Unsupported embedding provider: unknown"):
        EmbeddingFactory.get_provider_class("unknown")


def test_create_client_builds_client_for_supported_model(stub_providers: dict[str, type]) -> None:
    """Factory should validate models before constructing clients."""
    del stub_providers

    result = EmbeddingFactory.create_client("siliconflow/provider-model", timeout=5)

    assert result == {
        "provider": "siliconflow",
        "model": "provider-model",
        "kwargs": {"timeout": 5},
    }


def test_create_client_rejects_unsupported_model(stub_providers: dict[str, type]) -> None:
    """Unsupported models should raise before provider construction."""
    del stub_providers

    with pytest.raises(UnsupportedEmbeddingModelError, match="not supported by provider 'siliconflow'"):
        EmbeddingFactory.create_client("siliconflow/unknown-model")


def test_supports_concurrent_requests_returns_provider_capability(stub_providers: dict[str, type]) -> None:
    """Concurrent support should come from the resolved provider."""
    del stub_providers

    assert EmbeddingFactory.supports_concurrent_requests("nested/model") is True
    assert EmbeddingFactory.supports_concurrent_requests("fallback-model") is True
    assert EmbeddingFactory.supports_concurrent_requests("ollama/ollama-fallback") is False


def test_supports_concurrent_requests_returns_false_when_identifier_is_invalid(stub_providers: dict[str, type]) -> None:
    """Invalid identifiers should safely fall back to sequential execution."""
    del stub_providers

    assert EmbeddingFactory.supports_concurrent_requests("") is False


def test_get_supported_providers_returns_all_provider_models(stub_providers: dict[str, type]) -> None:
    """Supported provider listing should expose the underlying provider metadata."""
    del stub_providers

    providers = EmbeddingFactory.get_supported_providers()

    assert providers == {
        "lmstudio": {
            "dummy-embed": "Dummy LM Studio embedding model",
            "fallback-model": "Fallback LM Studio embedding model",
        },
        "ollama": {
            "ollama-embed": "Dummy Ollama embedding model",
            "ollama-fallback": "Fallback Ollama embedding model",
        },
        "siliconflow": {
            "nested/model": "Model with slash in name",
            "provider-model": "Explicit provider model",
        },
    }


def test_list_all_supported_models_includes_provider_prefixes(stub_providers: dict[str, type]) -> None:
    """Model listing should expose fully-qualified provider/model identifiers."""
    del stub_providers

    models = EmbeddingFactory.list_all_supported_models()

    assert models == [
        "lmstudio/dummy-embed",
        "lmstudio/fallback-model",
        "ollama/ollama-embed",
        "ollama/ollama-fallback",
        "siliconflow/nested/model",
        "siliconflow/provider-model",
    ]


def test_get_embedding_client_delegates_to_factory(stub_providers: dict[str, type]) -> None:
    """Convenience helper should forward arguments to the factory."""
    del stub_providers

    with patch.object(EmbeddingFactory, "create_client", return_value="client") as mock_create_client:
        result = get_embedding_client("lmstudio/dummy-embed", timeout=3)

    assert result == "client"
    mock_create_client.assert_called_once_with("lmstudio/dummy-embed", timeout=3)


def test_real_siliconflow_bge_m3_identifier_resolves_to_provider() -> None:
    """Bare SiliconFlow model names with slashes should resolve to the SiliconFlow provider."""
    provider, model = EmbeddingFactory.parse_embedding_identifier("BAAI/bge-m3")

    assert provider == "siliconflow"
    assert model == "BAAI/bge-m3"


@patch.dict("os.environ", {"SILICONFLOW_API_KEY": "test-key"})
def test_real_siliconflow_bge_m3_client_creation() -> None:
    """Explicit SiliconFlow identifiers should create a configured SiliconFlow client."""
    client = get_embedding_client("siliconflow/BAAI/bge-m3")

    assert isinstance(client, SiliconFlowClient)
    assert client.model == "BAAI/bge-m3"
    assert client.embedding_dimension == 1024
