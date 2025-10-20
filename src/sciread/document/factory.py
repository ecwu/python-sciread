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
from .splitters.semantic_splitter import SemanticSplitter
from .splitters.topic_flow import TopicFlowSplitter


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
    def create_topic_flow_document(
        file_path: Union[str, Path], ollama_client: Optional[OllamaClient] = None, **topic_flow_kwargs
    ) -> "Document":
        """
        Create document using TopicFlow splitting.

        Args:
            file_path: Path to the file.
            ollama_client: Optional Ollama client.
            **topic_flow_kwargs: Additional arguments for TopicFlowSplitter.

        Returns:
            Document instance split using TopicFlow algorithm.
        """
        if ollama_client is None:
            ollama_client = OllamaClient()

        splitter = TopicFlowSplitter(ollama_client=ollama_client, **topic_flow_kwargs)
        builder = DocumentBuilder()
        builder.splitter = splitter
        return builder.from_file(file_path, auto_split=True)
