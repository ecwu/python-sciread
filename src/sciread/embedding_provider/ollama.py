"""Ollama embedding provider."""

from typing import Any
from typing import ClassVar

import requests

from ..platform.logging import get_logger
from .base import BaseEmbeddingClient
from .base import BaseEmbeddingProvider


class OllamaClient(BaseEmbeddingClient):
    """Client for interacting with Ollama API for embeddings."""

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        cache_embeddings: bool = True,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings
        """
        super().__init__(
            model=model,
            timeout=timeout,
            cache_embeddings=cache_embeddings,
        )
        self.base_url = base_url.rstrip("/")
        self.logger = get_logger(__name__)

    def _get_single_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text from Ollama API."""
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.model, "prompt": text}

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]

            return None
        except Exception as e:
            self.logger.warning(f"Failed to get embedding from Ollama: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Failed to connect to Ollama: {e}")
            return False


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local embedding models."""

    # Supported models - common Ollama embedding models
    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "nomic-embed-text": "Nomic Embed Text - High quality text embeddings (768 dim)",
        "mxbai-embed-large": "MixedBread AI Large - High performance embeddings (1024 dim)",
        "all-minilm": "All MiniLM - Fast and efficient embeddings (384 dim)",
        "embeddinggemma:latest": "Embedding Gemma - Google's embedding model (768 dim)",
        "snowflake-arctic-embed": "Snowflake Arctic Embed - Enterprise embeddings (1024 dim)",
    }

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported Ollama embedding models."""
        return OllamaEmbeddingProvider.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by Ollama provider."""
        # Ollama supports custom models, so we're permissive here
        # Return True for known models, or if it looks like an Ollama model name
        if model_name in OllamaEmbeddingProvider.SUPPORTED_MODELS:
            return True
        # Allow any model name that might be a valid Ollama model
        # (e.g., "nomic-embed-text:latest", custom models)
        return ":" in model_name or "-" in model_name

    @staticmethod
    def create_client(model_name: str, **kwargs: Any) -> OllamaClient:
        """Create an Ollama embedding client.

        Args:
            model_name: Name of the Ollama model
            **kwargs: Additional arguments (base_url, timeout, cache_embeddings)

        Returns:
            OllamaClient instance
        """
        return OllamaClient(model=model_name, **kwargs)

    @staticmethod
    def supports_concurrent_requests() -> bool:
        """Ollama typically runs locally and handles requests sequentially.

        Returns:
            False - Ollama should use sequential requests
        """
        return False
