"""Factory module for creating documents with common configurations."""

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

if TYPE_CHECKING:
    from .document import Document

from .document_builder import DocumentBuilder
from .external_clients import MineruClient
from .external_clients import OllamaClient
from .models import DocumentMetadata
from .splitters.consecutive_flow import ConsecutiveFlowSplitter
from .splitters.cumulative_flow import CumulativeFlowSplitter
from .splitters.semantic_splitter import SemanticSplitter


class DocumentFactory:
    """Factory class for creating documents with common configurations."""

    @staticmethod
    def create_from_file(file_path: Union[str, Path], to_markdown: bool = False) -> "Document":
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
    def create_from_text(text: str, metadata: Optional[DocumentMetadata] = None) -> "Document":
        """
        Create document from text with default configuration.

        Args:
            text: Raw text content.
            metadata: Optional document metadata.

        Returns:
            Document instance.
        """
        builder = DocumentBuilder()
        return builder.from_text(text, metadata=metadata)

    @staticmethod
    def create_academic_document(
        file_path: Union[str, Path], use_markdown: bool = True, mineru_client: Optional[MineruClient] = None
    ) -> "Document":
        """
        Create document optimized for academic papers.

        Args:
            file_path: Path to the file.
            use_markdown: Whether to use markdown conversion for PDFs.
            mineru_client: Optional Mineru client.

        Returns:
            Document instance optimized for academic content.
        """
        builder = DocumentBuilder(mineru_client=mineru_client)
        splitter = SemanticSplitter(
            enable_academic_patterns=True,
            enable_markdown_patterns=use_markdown,
            min_chunk_size=300,
            max_chunk_size=2500,
        )
        return builder.from_file(file_path, to_markdown=use_markdown).with_splitter(splitter)

    @staticmethod
    def create_consecutive_flow_document(
        file_path: Union[str, Path], ollama_client: Optional[OllamaClient] = None, **consecutive_flow_kwargs
    ) -> "Document":
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
        file_path: Union[str, Path], ollama_client: Optional[OllamaClient] = None, **cumulative_flow_kwargs
    ) -> "Document":
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
