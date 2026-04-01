"""DeepSeek provider implementation for pydantic-ai."""

from typing import Any
from typing import ClassVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider as PydanticDeepSeekProvider

from ..platform.config import get_config


class DeepSeekProvider:
    """DeepSeek LLM provider using pydantic-ai."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "deepseek-chat": "DeepSeek Chat Model",
        "deepseek-reasoner": "DeepSeek Reasoner Model",
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> OpenAIChatModel:
        """Create a DeepSeek model instance.

        Args:
            model_name: Name of the DeepSeek model (e.g., 'deepseek-chat', 'deepseek-reasoner')
            **kwargs: Additional arguments to pass to the model

        Returns:
            OpenAIChatModel configured for DeepSeek

        Raises:
            ValueError: If model_name is not supported or API key is missing
        """
        if model_name not in cls.SUPPORTED_MODELS:
            supported = ", ".join(cls.SUPPORTED_MODELS.keys())
            raise ValueError(f"Unsupported DeepSeek model: {model_name}. Supported models: {supported}")

        config = get_config()

        api_key = config.get_api_key("deepseek")

        return OpenAIChatModel(
            model_name=model_name,
            provider=PydanticDeepSeekProvider(api_key=api_key),
            **kwargs,
        )

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get list of supported models and their descriptions."""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name is supported by this provider."""
        return model_name in cls.SUPPORTED_MODELS
