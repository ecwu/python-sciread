"""Factory for rerank client instances."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from sciread.rerank_provider.siliconflow import SiliconFlowRerankProvider

if TYPE_CHECKING:
    from sciread.rerank_provider.siliconflow import SiliconFlowRerankClient


class UnsupportedRerankModelError(Exception):
    """Raised when an unsupported rerank model is requested."""


class InvalidRerankIdentifierError(Exception):
    """Raised when a rerank model identifier is invalid."""


class RerankFactory:
    """Factory for creating rerank clients."""

    PROVIDERS: ClassVar[dict[str, type]] = {
        "siliconflow": SiliconFlowRerankProvider,
    }

    @classmethod
    def parse_rerank_identifier(cls, rerank_identifier: str) -> tuple[str, str]:
        """Parse a rerank identifier into provider and model name."""
        if not rerank_identifier or not rerank_identifier.strip():
            raise InvalidRerankIdentifierError("Rerank identifier cannot be empty")

        rerank_identifier = rerank_identifier.strip()
        if "/" not in rerank_identifier:
            return "siliconflow", rerank_identifier

        provider_name, model_name = rerank_identifier.split("/", 1)
        if provider_name in cls.PROVIDERS:
            if not model_name.strip():
                raise InvalidRerankIdentifierError(f"Invalid rerank identifier format: {rerank_identifier}")
            return provider_name.strip(), model_name.strip()

        for candidate_provider, provider_class in cls.PROVIDERS.items():
            if rerank_identifier in provider_class.get_supported_models() or provider_class.is_model_supported(rerank_identifier):
                return candidate_provider, rerank_identifier

        if not provider_name or not model_name:
            raise InvalidRerankIdentifierError(f"Invalid rerank identifier format: {rerank_identifier}")
        return provider_name.strip(), model_name.strip()

    @classmethod
    def get_provider_class(cls, provider_name: str):
        """Get a rerank provider class by name."""
        if provider_name not in cls.PROVIDERS:
            supported = ", ".join(cls.PROVIDERS)
            raise UnsupportedRerankModelError(f"Unsupported rerank provider: {provider_name}. Supported providers: {supported}")
        return cls.PROVIDERS[provider_name]

    @classmethod
    def create_client(cls, rerank_identifier: str, **kwargs: Any) -> SiliconFlowRerankClient:
        """Create a rerank client from a provider/model identifier."""
        provider_name, model_name = cls.parse_rerank_identifier(rerank_identifier)
        provider_class = cls.get_provider_class(provider_name)
        if not provider_class.is_model_supported(model_name):
            supported_models = list(provider_class.get_supported_models().keys())
            raise UnsupportedRerankModelError(
                f"Model '{model_name}' is not supported by provider '{provider_name}'. Supported models: {', '.join(supported_models)}"
            )
        return provider_class.create_client(model_name, **kwargs)

    @classmethod
    def get_supported_providers(cls) -> dict[str, dict[str, str]]:
        """Get supported rerank providers and models."""
        return {provider_name: provider_class.get_supported_models() for provider_name, provider_class in cls.PROVIDERS.items()}

    @classmethod
    def list_all_supported_models(cls) -> list[str]:
        """Get provider-qualified rerank model identifiers."""
        models = []
        for provider_name, provider_class in cls.PROVIDERS.items():
            for model_name in provider_class.get_supported_models():
                models.append(f"{provider_name}/{model_name}")
        return models


def get_rerank_client(rerank_identifier: str, **kwargs: Any) -> SiliconFlowRerankClient:
    """Get a rerank client instance."""
    return RerankFactory.create_client(rerank_identifier, **kwargs)
