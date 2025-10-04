"""Zhipu GLM provider implementation for pydantic-ai."""

from typing import Any, Dict

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider

from ..config import get_config


class ZhipuProvider:
    """Zhipu GLM LLM provider using pydantic-ai."""

    SUPPORTED_MODELS = {
        'glm-4.6': 'GLM-4.6 Model',
        'glm-4.5': 'GLM-4.5 Model'
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any) -> AnthropicModel:
        """Create a Zhipu GLM model instance.

        Args:
            model_name: Name of the GLM model (e.g., 'glm-4.6', 'glm-4.5')
            **kwargs: Additional arguments to pass to the model

        Returns:
            AnthropicModel configured for Zhipu GLM

        Raises:
            ValueError: If model_name is not supported or API key is missing
        """
        if model_name not in cls.SUPPORTED_MODELS:
            supported = ', '.join(cls.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported Zhipu model: {model_name}. "
                f"Supported models: {supported}"
            )

        config = get_config()
        provider_config = config.get_provider_config('zhipu')

        api_key = config.get_api_key('zhipu')
        base_url = provider_config.base_url or 'https://open.bigmodel.cn/api/anthropic'

        provider = PydanticAnthropicProvider(api_key=api_key, base_url=base_url)
        return AnthropicModel(
            model_name=model_name,
            provider=provider,
            **kwargs
        )

    @classmethod
    def get_supported_models(cls) -> Dict[str, str]:
        """Get list of supported models and their descriptions."""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model name is supported by this provider."""
        return model_name in cls.SUPPORTED_MODELS