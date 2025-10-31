"""Base provider for embedding models."""

from abc import ABC
from abc import abstractmethod
from typing import Any


class BaseEmbeddingProvider(ABC):
    """Base class for embedding model providers."""

    @staticmethod
    @abstractmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported models for this provider.

        Returns:
            Dictionary mapping model names to descriptions
        """

    @staticmethod
    @abstractmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by this provider.

        Args:
            model_name: Name of the model to check

        Returns:
            True if the model is supported, False otherwise
        """

    @staticmethod
    @abstractmethod
    def create_client(model_name: str, **kwargs: Any):
        """Create an embedding client instance.

        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the client

        Returns:
            Embedding client instance
        """

    @staticmethod
    @abstractmethod
    def supports_concurrent_requests() -> bool:
        """Check if this provider supports concurrent requests.

        Returns:
            True if concurrent requests are supported, False otherwise
        """
