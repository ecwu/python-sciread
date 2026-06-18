"""LLM Provider module for sciread package.

This module provides a unified interface for working with different LLM providers
including DeepSeek, Volcengine, LM Studio, and Ollama using pydantic-ai.

Main Interface:
    get_model(model_identifier: str) -> Model

Example Usage:
    from sciread.providers.llm import get_model

    # Explicit provider specification
    model = get_model("deepseek/deepseek-v4-flash")
    model = get_model("volcengine/doubao-seed-2.0-code")
    model = get_model("lmstudio/qwen3:4b")
    model = get_model("ollama/qwen3:4b")

    # Use default provider for known models
    model = get_model("deepseek-v4-flash")  # Uses deepseek provider
    model = get_model("glm-4.7")        # Uses volcengine provider
"""

from pydantic_ai.models.openai import OpenAIChatModel

from .factory import InvalidModelIdentifierError
from .factory import ModelFactory
from .factory import UnsupportedModelError
from .factory import get_model

__all__ = [
    "InvalidModelIdentifierError",
    "ModelFactory",
    "OpenAIChatModel",
    "UnsupportedModelError",
    "get_model",
]

# Version of the providers.llm module
__version__ = "1.0.0"
