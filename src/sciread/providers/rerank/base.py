"""Base models for rerank providers."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RerankResult:
    """One reranked document result."""

    index: int
    relevance_score: float
    document: str | None = None


class BaseRerankClient(ABC):
    """Minimal shared interface for rerank clients."""

    def __init__(
        self,
        *,
        model: str,
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        """Rerank documents by relevance to the query."""


class BaseRerankProvider(ABC):
    """Base class for rerank model providers."""

    @staticmethod
    @abstractmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported models for this provider."""

    @staticmethod
    @abstractmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by this provider."""

    @staticmethod
    @abstractmethod
    def create_client(model_name: str, **kwargs: Any):
        """Create a rerank client instance."""
