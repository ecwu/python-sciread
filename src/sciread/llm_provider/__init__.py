"""LLM Provider module for sciread package.

This module provides a unified interface for working with different LLM providers
including DeepSeek, Zhipu GLM, and Ollama using pydantic-ai.

Main Interface:
    get_model(model_identifier: str) -> Model

Example Usage:
    from sciread.llm_provider import get_model

    # Explicit provider specification
    model = get_model("deepseek/deepseek-chat")
    model = get_model("zhipu/glm-4.6")
    model = get_model("ollama/qwen3:4b")

    # Use default provider for known models
    model = get_model("deepseek-chat")  # Uses deepseek provider
    model = get_model("glm-4.6")        # Uses zhipu provider
"""

from typing import Any, Union

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel

from .factory import (
    ModelFactory,
    get_model,
    UnsupportedModelError,
    InvalidModelIdentifierError,
)

__all__ = [
    # Main interface
    'get_model',
    'ModelFactory',

    # Exceptions
    'UnsupportedModelError',
    'InvalidModelIdentifierError',

    # Type hints
    'OpenAIChatModel',
    'AnthropicModel',
]

# Version of the llm_provider module
__version__ = "1.0.0"