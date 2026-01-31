"""Factory module for creating documents with common configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

from ..embedding_provider import OllamaClient
from .document_builder import DocumentBuilder
from .models import DocumentMetadata
from .splitters.consecutive_flow import ConsecutiveFlowSplitter
from .splitters.cumulative_flow import CumulativeFlowSplitter


class DocumentFactory:
    """Factory class for creating documents with common configurations."""

    @staticmethod
    def create_from_file(file_path: str | Path, to_markdown: bool = False) -> Document:
        """
        Create document from file with default configuration.

        Args:
            file_path: Path to the file.
            to_markdown: Whether to convert PDF to markdown.

        Returns:
            Document instance.
        """
        builder = DocumentBuilder()
        return builder.from_file(file_path, to_markdown=to_markdown)

    @staticmethod
    def create_from_text(text: str, metadata: DocumentMetadata | None = None) -> Document:
        """
        Create document from text with default configuration.

        Args:
            text: Raw text content.
            metadata: Document metadata.

        Returns:
            Document instance.
        """
        builder = DocumentBuilder()
        return builder.from_text(text, metadata=metadata)

    @staticmethod
    def create_consecutive_flow_document(
        file_path: str | Path,
        ollama_client: OllamaClient | None = None,
        **consecutive_flow_kwargs,
    ) -> Document:
        """
        Create document using ConsecutiveFlow splitting.

        Args:
            file_path: Path to the file.
            ollama_client: Optional Ollama client.
            **consecutive_flow_kwargs: Additional arguments for ConsecutiveFlowSplitter.

        Returns:
            Document instance split using ConsecutiveFlow algorithm.
        """
        if ollama_client is None:
            ollama_client = OllamaClient()

        splitter = ConsecutiveFlowSplitter(ollama_client=ollama_client, **consecutive_flow_kwargs)
        builder = DocumentBuilder()
        builder.splitter = splitter
        return builder.from_file(file_path, auto_split=True)

    @staticmethod
    def create_cumulative_flow_document(
        file_path: str | Path,
        ollama_client: OllamaClient | None = None,
        **cumulative_flow_kwargs,
    ) -> Document:
        """
        Create document using CumulativeFlow splitting.

        Args:
            file_path: Path to the file.
            ollama_client: Optional Ollama client.
            **cumulative_flow_kwargs: Additional arguments for CumulativeFlowSplitter.

        Returns:
            Document instance split using CumulativeFlow algorithm.
        """
        if ollama_client is None:
            ollama_client = OllamaClient()

        splitter = CumulativeFlowSplitter(ollama_client=ollama_client, **cumulative_flow_kwargs)
        builder = DocumentBuilder()
        builder.splitter = splitter
        return builder.from_file(file_path, auto_split=True)
