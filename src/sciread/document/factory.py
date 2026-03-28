"""Factory module for creating documents with common configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

from .document_builder import DocumentBuilder
from .models import DocumentMetadata


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
