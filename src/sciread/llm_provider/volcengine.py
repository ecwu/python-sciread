"""Volcengine provider implementation for pydantic-ai."""

import os
from typing import Any
from typing import ClassVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from ..platform.config import get_config


class VolcengineProvider:
    """Volcengine LLM provider using OpenAI-compatible API."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "doubao-seed-2.0-code": "Doubao Seed 2.0 Code Model",
        "doubao-seed-2.0-pro": "Doubao Seed 2.0 Pro Model",
        "doubao-seed-2.0-lite": "Doubao Seed 2.0 Lite Model",
        "doubao-seed-code": "Doubao Seed Code Model",
        "minimax-m2.5": "MiniMax M2.5 Model",
        "glm-4.7": "GLM-4.7 Model",
        "deepseek-v3.2": "DeepSeek V3.2 Model",
        "kimi-k2.5": "Kimi K2.5 Model",
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> OpenAIChatModel:
        """Create a Volcengine model instance.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional arguments to pass to the model

        Returns:
            OpenAIChatModel configured for Volcengine

        Raises:
            ValueError: If model_name is not supported or API key is missing
        """
        if model_name not in cls.SUPPORTED_MODELS:
            supported = ", ".join(cls.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unsupported Volcengine model: {model_name}. Supported models: {supported}")

        config = get_config()
        provider_config = config.get_provider_config("volcengine")

        api_key = provider_config.api_key or os.getenv("VOLCES_API")
        if not api_key:
            raise ValueError("No API key found for provider 'volcengine'. Set VOLCES_API environment variable or configure in config file.")

        base_url = provider_config.base_url or "https://ark.cn-beijing.volces.com/api/coding/v3"

        provider = PydanticOpenAIProvider(api_key=api_key, base_url=base_url)
        return OpenAIChatModel(model_name=model_name, provider=provider, **kwargs)

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get list of supported models and their descriptions."""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name is supported by this provider."""
        return model_name in cls.SUPPORTED_MODELS
