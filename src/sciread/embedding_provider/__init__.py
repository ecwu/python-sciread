"""Embedding provider module for managing different embedding model providers."""

from .factory import EmbeddingFactory
from .factory import get_embedding_client
from .ollama import OllamaClient
from .siliconflow import SiliconFlowClient

__all__ = [
    "get_embedding_client",
    "EmbeddingFactory",
    "OllamaClient",
    "SiliconFlowClient",
]
