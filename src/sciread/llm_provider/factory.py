"""Factory for creating LLM model instances from string identifiers."""

from typing import Any
from typing import ClassVar

from pydantic_ai.models.openai import OpenAIChatModel

from ..platform.config import get_config
from .deepseek import DeepSeekProvider
from .ollama import OllamaProvider
from .volcengine import VolcengineProvider


class UnsupportedModelError(Exception):
    """Raised when an unsupported model is requested."""


class InvalidModelIdentifierError(Exception):
    """Raised when a model identifier is in an invalid format."""


class ModelFactory:
    """Factory for creating LLM model instances."""

    PROVIDERS: ClassVar[dict[str, type]] = {
        "deepseek": DeepSeekProvider,
        "volcengine": VolcengineProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def parse_model_identifier(cls, model_identifier: str) -> tuple[str, str]:
        """Parse a model identifier into provider and model name.

        Args:
            model_identifier: Model identifier in format "provider/model" or just "model"

        Returns:
            Tuple of (provider_name, model_name)

        Raises:
            InvalidModelIdentifierError: If the identifier format is invalid
        """
        if not model_identifier or not model_identifier.strip():
            raise InvalidModelIdentifierError("Model identifier cannot be empty")

        model_identifier = model_identifier.strip()

        if "/" in model_identifier:
            parts = model_identifier.split("/", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise InvalidModelIdentifierError(
                    f"Invalid model identifier format: {model_identifier}. Expected format: 'provider/model' or 'model'"
                )
            return parts[0].strip(), parts[1].strip()
        else:
            # No provider specified, use default
            config = get_config()
            default_provider = config.default.provider

            # If the identifier is a known model name for a provider, use that provider
            for provider_name, provider_class in cls.PROVIDERS.items():
                if provider_class.is_model_supported(model_identifier):
                    return provider_name, model_identifier

            # Otherwise, use default provider with specified model name
            return default_provider, model_identifier

    @classmethod
    def get_provider_class(cls, provider_name: str):
        """Get the provider class for a given provider name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider class

        Raises:
            UnsupportedModelError: If the provider is not supported
        """
        if provider_name not in cls.PROVIDERS:
            supported = ", ".join(cls.PROVIDERS.keys())
            raise UnsupportedModelError(f"Unsupported provider: {provider_name}. Supported providers: {supported}")
        return cls.PROVIDERS[provider_name]

    @classmethod
    def create_model(cls, model_identifier: str, **kwargs: Any) -> OpenAIChatModel:
        """Create a model instance from a model identifier.

        Args:
            model_identifier: Model identifier in format "provider/model" or just "model"
            **kwargs: Additional arguments to pass to the model

        Returns:
            Model instance

        Raises:
            UnsupportedModelError: If the provider or model is not supported
            InvalidModelIdentifierError: If the identifier format is invalid
        """
        provider_name, model_name = cls.parse_model_identifier(model_identifier)
        provider_class = cls.get_provider_class(provider_name)

        # Validate that the model is supported by the provider
        if not provider_class.is_model_supported(model_name):
            supported_models = list(provider_class.get_supported_models().keys())
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported by provider '{provider_name}'. Supported models: {', '.join(supported_models)}"
            )

        # Create the model instance
        return provider_class.create_model(model_name, **kwargs)

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
        """Get a list of all supported model identifiers.

        Returns:
            List of model identifiers in format "provider/model"
        """
        models = []
        for provider_name, provider_class in cls.PROVIDERS.items():
            for model_name in provider_class.get_supported_models():
                models.append(f"{provider_name}/{model_name}")
        return models


def get_model(model_identifier: str, **kwargs: Any) -> OpenAIChatModel:
    """Get a model instance.

    This is the main public interface for creating LLM model instances.
    The function uses a factory pattern to create models from different providers.

    Args:
        model_identifier: Model identifier in format "provider/model" or just "model"
        **kwargs: Additional arguments to pass to the model

    Returns:
        Model instance

    Examples:
        >>> model = get_model("deepseek/deepseek-chat")
        >>> model = get_model("volcengine/doubao-seed-2.0-code")
        >>> model = get_model("ollama/qwen3:4b")
        >>> # Use default provider for known models
        >>> model = get_model("deepseek-chat")
    """
    return ModelFactory.create_model(model_identifier, **kwargs)
