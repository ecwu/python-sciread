"""Ollama provider implementation for pydantic-ai."""

from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider as PydanticOllamaProvider

from ..platform.config import get_config


class OllamaProvider:
    """Ollama LLM provider using pydantic-ai."""

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> OpenAIChatModel:
        """Create an Ollama model instance.

        Args:
            model_name: Name of the Ollama model (e.g., 'qwen3:4b', 'llama2:7b')
            **kwargs: Additional arguments to pass to the model

        Returns:
            OpenAIChatModel configured for Ollama

        Raises:
            ValueError: If base_url is not configured
        """
        config = get_config()
        provider_config = config.get_provider_config("ollama")

        base_url = provider_config.base_url or "http://localhost:11434/v1"

        # Ollama doesn't require an API key since it's local
        # The base_url is handled by the OllamaProvider, not OpenAIChatModel
        return OpenAIChatModel(model_name=model_name, provider=PydanticOllamaProvider(base_url=base_url), **kwargs)

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get list of supported models.

        Note: Ollama supports any model that is available locally.
        This method returns some common examples.
        """
        return {
            "qwen3:4b": "Qwen3 4B parameter model",
            "qwen3:8b": "Qwen3 8B parameter model",
            "qwen3:14b": "Qwen3 14B parameter model",
            "llama2:7b": "Llama2 7B parameter model",
            "llama2:13b": "Llama2 13B parameter model",
            "llama3:8b": "Llama3 8B parameter model",
            "llama3:70b": "Llama3 70B parameter model",
            "mistral:7b": "Mistral 7B parameter model",
            "codellama:7b": "Code Llama 7B parameter model",
        }

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name could be supported by this provider.

        Ollama can theoretically support any model, but we should only
        return True for models that are clearly Ollama models (contain
        typical Ollama naming patterns) to avoid conflicts with specific
        provider models.
        """
        if not model_name or not model_name.strip():
            return False

        # Return True for common Ollama patterns
        ollama_patterns = [
            ":",  # Tag patterns like "qwen3:4b"
            "llama",
            "mistral",
            "codellama",
            "qwen",
            "gemma",
            "phi",
        ]

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in ollama_patterns)
