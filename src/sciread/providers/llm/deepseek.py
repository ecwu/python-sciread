"""DeepSeek provider implementation for pydantic-ai."""

from typing import Any
from typing import ClassVar

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider as PydanticDeepSeekProvider

from ...platform.config import get_config


class DeepSeekProvider:
    """DeepSeek LLM provider using pydantic-ai."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "deepseek-v4-flash": "DeepSeek Chat Model",
        "deepseek-v4-pro": "DeepSeek Reasoner Model",
    }

    @staticmethod
    def _with_default_settings(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Default DeepSeek v4 to non-thinking mode for tool-compatible agent runs."""
        settings = dict(kwargs.get("settings") or {})
        settings.setdefault("thinking", False)
        return {**kwargs, "settings": settings}

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> OpenAIChatModel:
        """Create a DeepSeek model instance.

        Args:
            model_name: Name of the DeepSeek model (e.g., 'deepseek-v4-flash', 'deepseek-v4-pro')
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
        model_kwargs = cls._with_default_settings(kwargs)

        return OpenAIChatModel(
            model_name=model_name,
            provider=PydanticDeepSeekProvider(api_key=api_key),
            **model_kwargs,
        )

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get list of supported models and their descriptions."""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name is supported by this provider."""
        return model_name in cls.SUPPORTED_MODELS
