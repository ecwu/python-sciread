"""LM Studio provider implementation for pydantic-ai."""

from typing import Any
from typing import ClassVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from ..platform.config import get_config


class LMStudioProvider:
    """LM Studio LLM provider using an OpenAI-compatible local API."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "qwen3:4b": "Qwen3 4B parameter model",
        "qwen3:8b": "Qwen3 8B parameter model",
        "qwen3:14b": "Qwen3 14B parameter model",
        "llama-3.2-3b-it": "Llama 3.2 3B Instruct model",
        "llama3:8b": "Llama3 8B parameter model",
        "mistral:7b": "Mistral 7B parameter model",
        "gemma3:4b": "Gemma 3 4B parameter model",
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> OpenAIChatModel:
        """Create an LM Studio model instance.

        Args:
            model_name: Name of the model served by LM Studio.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            OpenAIChatModel configured for LM Studio.
        """
        config = get_config()
        provider_config = config.get_provider_config("lmstudio")

        base_url = provider_config.base_url or "http://localhost:1234/v1"
        api_key = provider_config.api_key or "lm_studio"

        provider = PydanticOpenAIProvider(api_key=api_key, base_url=base_url)
        return OpenAIChatModel(model_name=model_name, provider=provider, **kwargs)

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get list of supported model examples."""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name looks like a locally served LM Studio model."""
        if not model_name or not model_name.strip():
            return False

        if model_name in cls.SUPPORTED_MODELS:
            return True

        lmstudio_patterns = [
            ":",
            "llama",
            "mistral",
            "qwen",
            "gemma",
            "phi",
        ]
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in lmstudio_patterns)
