"""Embedding provider module for managing different embedding model providers."""

from .factory import EmbeddingFactory
from .factory import get_embedding_client
from .lmstudio import LMStudioClient
from .ollama import OllamaClient
from .siliconflow import SiliconFlowClient

__all__ = [
    "EmbeddingFactory",
    "LMStudioClient",
    "OllamaClient",
    "SiliconFlowClient",
    "get_embedding_client",
]
