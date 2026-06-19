"""Reusable fake providers and agent result objects for layered tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sciread.providers.embedding.base import BaseEmbeddingClient
from sciread.providers.rerank.base import BaseRerankClient
from sciread.providers.rerank.base import RerankResult


@dataclass
class FakeAgentRunResult:
    """Minimal pydantic-ai compatible run result."""

    output: Any


class FakeLLMModel:
    """Small model placeholder for tests that should not initialize real providers."""

    model_name = "fake-model"


class FakeEmbeddingClient(BaseEmbeddingClient):
    """Deterministic embedding client with optional failure injection."""

    embedding_batch_size = 2

    def __init__(
        self,
        *,
        model: str = "fake-embedding",
        dimension: int = 6,
        fail_for: set[str] | None = None,
        cache_embeddings: bool = True,
    ) -> None:
        super().__init__(model=model, cache_embeddings=cache_embeddings, embedding_dimension=dimension)
        self.dimension = dimension
        self.fail_for = fail_for or set()
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def _get_single_embedding(self, text: str) -> list[float] | None:
        self.single_calls.append(text)
        if text in self.fail_for:
            return None

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [round(digest[index] / 255, 6) for index in range(self.dimension)]

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        self.batch_calls.append(texts.copy())
        return [self._get_single_embedding(text) for text in texts]


class FakeRerankClient(BaseRerankClient):
    """Deterministic reranker that can exercise fallback and duplicate handling."""

    def __init__(
        self,
        *,
        model: str = "fake-rerank",
        scores: list[float] | None = None,
        duplicate_first: bool = False,
        return_empty: bool = False,
    ) -> None:
        super().__init__(model=model)
        self.scores = scores
        self.duplicate_first = duplicate_first
        self.return_empty = return_empty
        self.calls: list[tuple[str, list[str], int | None]] = []

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        self.calls.append((query, documents.copy(), top_n))
        if self.return_empty:
            return []

        results = []
        for index, document in enumerate(documents):
            score = self.scores[index] if self.scores and index < len(self.scores) else float(len(documents) - index)
            results.append(RerankResult(index=index, relevance_score=score, document=document))

        results.sort(key=lambda item: item.relevance_score, reverse=True)
        selected = results[:top_n] if top_n is not None else results
        if self.duplicate_first and selected:
            selected = [selected[0], selected[0], *selected[1:]]
        return selected


class FakeMineruClient:
    """Fake Mineru client for PDF-to-markdown loader tests."""

    def __init__(self, *, markdown: str = "# Fake Paper\n\nConverted markdown.", error: Exception | None = None) -> None:
        self.markdown = markdown
        self.error = error
        self.calls: list[Path] = []

    def extract_markdown(self, file_path: Path) -> str:
        self.calls.append(file_path)
        if self.error is not None:
            raise self.error
        return self.markdown
