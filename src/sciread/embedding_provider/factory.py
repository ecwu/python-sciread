"""Factory for creating embedding client instances from string identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from .ollama import OllamaEmbeddingProvider
from .siliconflow import SiliconFlowEmbeddingProvider

if TYPE_CHECKING:
    from .ollama import OllamaClient
    from .siliconflow import SiliconFlowClient


class UnsupportedEmbeddingModelError(Exception):
    """Raised when an unsupported embedding model is requested."""


class InvalidEmbeddingIdentifierError(Exception):
    """Raised when an embedding model identifier is in an invalid format."""


class EmbeddingFactory:
    """Factory for creating embedding client instances."""

    PROVIDERS: ClassVar[dict[str, type]] = {
        "ollama": OllamaEmbeddingProvider,
        "siliconflow": SiliconFlowEmbeddingProvider,
    }

    @classmethod
    def parse_embedding_identifier(cls, embedding_identifier: str) -> tuple[str, str]:
        """Parse an embedding identifier into provider and model name.

        Args:
            embedding_identifier: Embedding identifier in format "provider/model" or just "model"

        Returns:
            Tuple of (provider_name, model_name)

        Raises:
            InvalidEmbeddingIdentifierError: If the identifier format is invalid
        """
        if not embedding_identifier or not embedding_identifier.strip():
            raise InvalidEmbeddingIdentifierError(
                "Embedding identifier cannot be empty"
            )

        embedding_identifier = embedding_identifier.strip()

        if "/" in embedding_identifier:
            # Check if it's a provider/model format
            parts = embedding_identifier.split("/", 1)

            # Special case: SiliconFlow models contain "/" in their names
            # e.g., "siliconflow/Qwen/Qwen3-Embedding-8B"
            if parts[0] in cls.PROVIDERS:
                # It's an explicit provider specification
                provider_name = parts[0].strip()
                model_name = parts[1].strip()
                return provider_name, model_name
            else:
                # It might be a model name itself (e.g., "Qwen/Qwen3-Embedding-8B")
                # Check if any provider supports this full name
                for provider_name, provider_class in cls.PROVIDERS.items():
                    if provider_class.is_model_supported(embedding_identifier):
                        return provider_name, embedding_identifier

                # If not found, treat first part as provider name
                if not parts[0] or not parts[1]:
                    raise InvalidEmbeddingIdentifierError(
                        f"Invalid embedding identifier format: {embedding_identifier}"
                    )
                return parts[0].strip(), parts[1].strip()
        else:
            # No explicit provider, try to infer from model name
            for provider_name, provider_class in cls.PROVIDERS.items():
                if provider_class.is_model_supported(embedding_identifier):
                    return provider_name, embedding_identifier

            # Use default Ollama provider if no match found
            return "ollama", embedding_identifier

    @classmethod
    def get_provider_class(cls, provider_name: str):
        """Get the provider class for a given provider name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider class

        Raises:
            UnsupportedEmbeddingModelError: If the provider is not supported
        """
        if provider_name not in cls.PROVIDERS:
            supported = ", ".join(cls.PROVIDERS.keys())
            raise UnsupportedEmbeddingModelError(
                f"Unsupported embedding provider: {provider_name}. "
                f"Supported providers: {supported}"
            )
        return cls.PROVIDERS[provider_name]

    @classmethod
    def create_client(
        cls, embedding_identifier: str, **kwargs: Any
    ) -> OllamaClient | SiliconFlowClient:
        """Create an embedding client instance from an embedding identifier.

        Args:
            embedding_identifier: Embedding identifier in format "provider/model" or just "model"
            **kwargs: Additional arguments to pass to the client

        Returns:
            Embedding client instance (OllamaClient or SiliconFlowClient)

        Raises:
            UnsupportedEmbeddingModelError: If the provider or model is not supported
            InvalidEmbeddingIdentifierError: If the identifier format is invalid

        Examples:
            >>> client = EmbeddingFactory.create_client("siliconflow/Qwen/Qwen3-Embedding-8B")
            >>> client = EmbeddingFactory.create_client("ollama/nomic-embed-text")
            >>> client = EmbeddingFactory.create_client("embeddinggemma:latest")  # Uses Ollama
        """
        provider_name, model_name = cls.parse_embedding_identifier(embedding_identifier)
        provider_class = cls.get_provider_class(provider_name)

        # Validate that the model is supported by the provider
        if not provider_class.is_model_supported(model_name):
            supported_models = list(provider_class.get_supported_models().keys())
            raise UnsupportedEmbeddingModelError(
                f"Model '{model_name}' is not supported by provider '{provider_name}'. "
                f"Supported models: {', '.join(supported_models)}"
            )

        # Create the client instance
        return provider_class.create_client(model_name, **kwargs)

    @classmethod
    def supports_concurrent_requests(cls, embedding_identifier: str) -> bool:
        """Check if an embedding provider supports concurrent requests.

        Args:
            embedding_identifier: Embedding identifier in format "provider/model" or just "model"

        Returns:
            True if concurrent requests are supported, False otherwise
        """
        try:
            provider_name, _ = cls.parse_embedding_identifier(embedding_identifier)
            provider_class = cls.get_provider_class(provider_name)
            return provider_class.supports_concurrent_requests()
        except (InvalidEmbeddingIdentifierError, UnsupportedEmbeddingModelError):
            # Default to False (sequential) if we can't determine
            return False

    @classmethod
    def get_supported_providers(cls) -> dict[str, dict[str, str]]:
        """Get all supported providers and their models.

        Returns:
            Dictionary mapping provider names to their supported models
        """
        result = {}
        for provider_name, provider_class in cls.PROVIDERS.items():
            result[provider_name] = provider_class.get_supported_models()
        return result

    @classmethod
    def list_all_supported_models(cls) -> list[str]:
        """Get a list of all supported embedding model identifiers.

        Returns:
            List of embedding identifiers
        """
        models = []
        for provider_name, provider_class in cls.PROVIDERS.items():
            for model_name in provider_class.get_supported_models():
                models.append(f"{provider_name}/{model_name}")
        return models


def get_embedding_client(
    embedding_identifier: str, **kwargs: Any
) -> OllamaClient | SiliconFlowClient:
    """Get an embedding client instance.

    This is the main public interface for creating embedding client instances.
    The function uses a factory pattern to create clients from different providers.

    Args:
        embedding_identifier: Embedding identifier in format "provider/model" or just "model"
        **kwargs: Additional arguments to pass to the client

    Returns:
        Embedding client instance (OllamaClient or SiliconFlowClient)

    Examples:
        >>> # Explicit provider specification
        >>> client = get_embedding_client("siliconflow/Qwen/Qwen3-Embedding-8B")
        >>> client = get_embedding_client("ollama/nomic-embed-text")
        >>>
        >>> # Inferred provider from model name
        >>> client = get_embedding_client("embeddinggemma:latest")  # Uses Ollama
        >>> client = get_embedding_client("Qwen/Qwen3-Embedding-8B")  # Uses SiliconFlow
        >>>
        >>> # Check if concurrent requests are supported
        >>> if EmbeddingFactory.supports_concurrent_requests("siliconflow/Qwen/Qwen3-Embedding-8B"):
        ...     print("Can use parallel requests!")
    """
    return EmbeddingFactory.create_client(embedding_identifier, **kwargs)
