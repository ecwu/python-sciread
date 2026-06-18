"""SiliconFlow rerank provider."""

from __future__ import annotations

import os
from typing import Any
from typing import ClassVar

import requests

from sciread.platform.logging import get_logger
from sciread.providers.rerank.base import BaseRerankClient
from sciread.providers.rerank.base import BaseRerankProvider
from sciread.providers.rerank.base import RerankResult


class SiliconFlowRerankClient(BaseRerankClient):
    """Client for SiliconFlow's rerank API."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key: str | None = None,
        timeout: int = 30,
        return_documents: bool = False,
        instruction: str | None = None,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.return_documents = return_documents
        self.instruction = instruction
        self.max_chunks_per_doc = max_chunks_per_doc
        self.overlap_tokens = overlap_tokens
        self.logger = get_logger(__name__)

        if not self.api_key:
            self.logger.warning("No SiliconFlow API key provided. Set SILICONFLOW_API_KEY environment variable.")

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        """Rerank text documents through SiliconFlow."""
        normalized_query = query.strip()
        if not normalized_query or not documents or not any(document.strip() for document in documents):
            return []

        if not self.api_key:
            self.logger.warning("No API key available for SiliconFlow rerank")
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "query": normalized_query,
            "documents": documents,
            "return_documents": self.return_documents,
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(top_n, len(documents)))
        if self.instruction:
            payload["instruction"] = self.instruction
        if self.max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = self.max_chunks_per_doc
        if self.overlap_tokens is not None:
            payload["overlap_tokens"] = self.overlap_tokens

        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                self.logger.warning(f"SiliconFlow rerank API returned status {response.status_code}: {response.text}")
                return []

            data = response.json()
            results = []
            for item in data.get("results", []):
                parsed_result = self._parse_result(item)
                if parsed_result is not None and 0 <= parsed_result.index < len(documents):
                    results.append(parsed_result)
            return results
        except Exception as exc:
            self.logger.warning(f"Failed to rerank documents with SiliconFlow: {exc}")
            return []

    def _parse_result(self, item: dict[str, Any]) -> RerankResult | None:
        """Parse one SiliconFlow result object."""
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not isinstance(score, int | float):
            return None

        document_text = None
        document = item.get("document")
        if isinstance(document, dict):
            text = document.get("text")
            if isinstance(text, str):
                document_text = text

        return RerankResult(index=index, relevance_score=float(score), document=document_text)

    def test_connection(self) -> bool:
        """Test connection to SiliconFlow rerank API."""
        return bool(self.rerank("test", ["test"], top_n=1))


class SiliconFlowRerankProvider(BaseRerankProvider):
    """SiliconFlow rerank provider."""

    SUPPORTED_MODELS: ClassVar[dict[str, str]] = {
        "BAAI/bge-reranker-v2-m3": "BGE reranker v2 M3",
        "Pro/BAAI/bge-reranker-v2-m3": "BGE reranker v2 M3 Pro",
        "netease-youdao/bce-reranker-base_v1": "BCE reranker base v1",
        "Qwen/Qwen3-Reranker-8B": "Qwen3 reranker 8B",
        "Qwen/Qwen3-Reranker-4B": "Qwen3 reranker 4B",
        "Qwen/Qwen3-Reranker-0.6B": "Qwen3 reranker 0.6B",
    }

    @staticmethod
    def get_supported_models() -> dict[str, str]:
        """Get supported SiliconFlow rerank models."""
        return SiliconFlowRerankProvider.SUPPORTED_MODELS.copy()

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported by SiliconFlow rerank."""
        return model_name in SiliconFlowRerankProvider.SUPPORTED_MODELS

    @staticmethod
    def create_client(model_name: str, **kwargs: Any) -> SiliconFlowRerankClient:
        """Create a SiliconFlow rerank client."""
        return SiliconFlowRerankClient(model=model_name, **kwargs)
