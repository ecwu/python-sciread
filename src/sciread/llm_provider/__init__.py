"""LLM Provider module for sciread package.

This module provides a unified interface for working with different LLM providers
including DeepSeek, Volcengine, and Ollama using pydantic-ai.

Main Interface:
    get_model(model_identifier: str) -> Model

Example Usage:
    from sciread.llm_provider import get_model

    # Explicit provider specification
    model = get_model("deepseek/deepseek-chat")
    model = get_model("volcengine/doubao-seed-2.0-code")
    model = get_model("ollama/qwen3:4b")

    # Use default provider for known models
    model = get_model("deepseek-chat")  # Uses deepseek provider
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

# Version of the llm_provider module
__version__ = "1.0.0"
