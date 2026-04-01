"""Factory module for creating documents with common configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

from .builder import DocumentBuilder
from .models import DocumentMetadata


class DocumentFactory:
    """Factory class for creating documents with common configurations."""

    @staticmethod
    def create_from_file(file_path: str | Path, to_markdown: bool = False, **builder_kwargs) -> Document:
        """
        Create document from file with default configuration.

        Args:
            file_path: Path to the file.
            to_markdown: Whether to convert PDF to markdown.
            **builder_kwargs: Additional arguments forwarded to ``DocumentBuilder.from_file``.

        Returns:
            Document instance.
        """
        return DocumentBuilder().from_file(file_path, to_markdown=to_markdown, **builder_kwargs)

    @staticmethod
    def create_from_text(text: str, metadata: DocumentMetadata | None = None, **builder_kwargs) -> Document:
        """
        Create document from text with default configuration.

        Args:
            text: Raw text content.
            metadata: Document metadata.
            **builder_kwargs: Additional arguments forwarded to ``DocumentBuilder.from_text``.

        Returns:
            Document instance.
        """
        return DocumentBuilder().from_text(text, metadata=metadata, **builder_kwargs)
