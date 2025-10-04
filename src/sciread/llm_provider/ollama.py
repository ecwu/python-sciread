"""Ollama provider implementation for pydantic-ai."""

from typing import Any, Dict

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from ..config import get_config


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
        provider_config = config.get_provider_config('ollama')

        base_url = provider_config.base_url or 'http://localhost:11434/v1'

        # Ollama doesn't require an API key since it's local
        provider = OllamaProvider()
        if base_url != "http://localhost:11434/v1":
            # For custom base URL, we need to handle it differently
            provider = OllamaProvider()

        return OpenAIChatModel(
            model_name=model_name,
            provider=provider,
            base_url=base_url,
            **kwargs
        )

    @classmethod
    def get_supported_models(cls) -> Dict[str, str]:
        """Get list of supported models.

        Note: Ollama supports any model that is available locally.
        This method returns some common examples.
        """
        return {
            'qwen3:4b': 'Qwen3 4B parameter model',
            'qwen3:8b': 'Qwen3 8B parameter model',
            'qwen3:14b': 'Qwen3 14B parameter model',
            'llama2:7b': 'Llama2 7B parameter model',
            'llama2:13b': 'Llama2 13B parameter model',
            'llama3:8b': 'Llama3 8B parameter model',
            'llama3:70b': 'Llama3 70B parameter model',
            'mistral:7b': 'Mistral 7B parameter model',
            'codellama:7b': 'Code Llama 7B parameter model',
        }

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name could be supported by this provider.

        Ollama can theoretically support any model, so we return True
        for any non-empty model name. The actual availability depends
        on what models are installed in the local Ollama instance.
        """
        return bool(model_name and model_name.strip())