"""Document builder class for creating and processing documents."""

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

if TYPE_CHECKING:
    from .document import Document

from ..logging_config import get_logger
from .external_clients import MineruClient
from .external_clients import OllamaClient
from .loaders import BaseLoader
from .loaders.pdf_loader import PdfLoader
from .loaders.txt_loader import TxtLoader
from .models import DocumentMetadata
from .splitters import BaseSplitter
from .splitters.markdown_splitter import MarkdownSplitter
from .splitters.semantic_splitter import SemanticSplitter


class DocumentBuilder:
    """Builder class for creating and processing documents."""

    def __init__(
        self,
        loader: Optional[BaseLoader] = None,
        splitter: Optional[BaseSplitter] = None,
        ollama_client: Optional[OllamaClient] = None,
        mineru_client: Optional[MineruClient] = None,
    ):
        """
        Initialize document builder.

        Args:
            loader: Custom loader to use (optional)
            splitter: Custom splitter to use (optional)
            ollama_client: Ollama client for embedding operations (optional)
            mineru_client: Mineru client for PDF markdown conversion (optional)
        """
        self.logger = get_logger(__name__)
        self.loader = loader
        self.splitter = splitter
        self.ollama_client = ollama_client
        self.mineru_client = mineru_client

    def from_file(
        self,
        file_path: Union[str, Path],
        to_markdown: bool = False,
        auto_split: bool = True,
        **split_kwargs,
    ) -> "Document":
        """
        Create a Document from a file path.

        Args:
            file_path: Path to the file.
            to_markdown: If True, convert PDF to markdown using Mineru API.
            auto_split: Whether to automatically split the document after loading.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with loaded content and optionally split chunks.
        """
        from .document import Document  # Import here to avoid circular imports

        path = Path(file_path)
        self.logger.debug(
            f"Creating document from file: {path} (to_markdown={to_markdown})"
        )

        # Initialize default loader if none provided
        if self.loader is None:
            self.loader = self._create_default_loader(path, to_markdown)

        # Load the document
        load_result = self.loader.load(path)
        if not load_result.success:
            raise RuntimeError(f"Failed to load document: {load_result.errors}")

        # Create document with loaded content
        doc = Document(
            source_path=path,
            text=load_result.text,
            metadata=load_result.metadata,
            _is_markdown=to_markdown,
        )

        # Update processing state
        doc.processing_state.update_timestamp("loaded")
        doc.processing_state.add_note(
            f"Document loaded using {self.loader.loader_name}"
        )

        # Add any warnings to processing state
        for warning in load_result.warnings:
            doc.processing_state.add_note(f"Warning: {warning}")
            self.logger.warning(f"Document loading warning: {warning}")

        self.logger.debug(
            f"Successfully loaded document using {self.loader.loader_name}: {len(load_result.text)} characters"
        )

        # Auto-split if requested
        if auto_split:
            self._split_document(doc, **split_kwargs)

        return doc

    def from_text(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
        auto_split: bool = True,
        is_markdown: bool = False,
        **split_kwargs,
    ) -> "Document":
        """
        Create a Document from raw text.

        Args:
            text: Raw text content.
            metadata: Optional document metadata.
            auto_split: Whether to automatically split the document.
            is_markdown: Whether the text is markdown format.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with text and optionally split chunks.
        """
        from .document import Document  # Import here to avoid circular imports

        self.logger.debug(f"Creating document from text ({len(text)} characters)")

        doc = Document(
            source_path=None,
            text=text,
            metadata=metadata or DocumentMetadata(source_path=None),
            _is_markdown=is_markdown,
        )

        # Update processing state
        doc.processing_state.update_timestamp("loaded")
        doc.processing_state.add_note("Document created from text")

        # Auto-split if requested
        if auto_split:
            self._split_document(doc, **split_kwargs)

        return doc

    def _create_default_loader(self, file_path: Path, to_markdown: bool) -> BaseLoader:
        """Create appropriate loader based on file extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PdfLoader(
                to_markdown=to_markdown,
                mineru_client=self.mineru_client,
            )
        elif suffix in [".txt", ".md", ".rst"]:
            return TxtLoader()
        else:
            # Try to use available loaders
            for loader_class in [PdfLoader, TxtLoader]:
                if suffix in loader_class(None).supported_extensions:
                    return loader_class()

            raise ValueError(f"Unsupported file format: {suffix}")

    def _split_document(
        self, doc: "Document", splitter: Optional[BaseSplitter] = None, **split_kwargs
    ) -> None:
        """
        Split document into chunks.

        Args:
            doc: Document to split.
            splitter: Custom splitter to use (optional).
            **split_kwargs: Additional arguments for splitting.
        """
        if not doc.text.strip():
            raise ValueError("Cannot split empty document")

        # Use provided splitter, default splitter, or create appropriate one
        active_splitter = splitter or self.splitter
        if active_splitter is None:
            active_splitter = self._create_default_splitter(doc)

        # Split the text
        chunks = active_splitter.split(doc.text)
        doc._set_chunks(chunks)
        doc.processing_state.add_note(
            f"Document split using {active_splitter.splitter_name}"
        )

        self.logger.info(
            f"Document split into {len(chunks)} chunks using {active_splitter.splitter_name}"
        )

    def _create_default_splitter(self, doc: "Document") -> BaseSplitter:
        """Create appropriate splitter based on document content."""
        # If document was created with markdown conversion, use the dedicated MarkdownSplitter
        if doc._is_markdown:
            return MarkdownSplitter(
                min_chunk_size=200,
                max_chunk_size=2000,
                preserve_code_blocks=True,
                split_on_headers=True,
                confidence_threshold=0.7,
            )

        # For other documents, use semantic splitter with academic patterns
        return SemanticSplitter(
            enable_academic_patterns=True,
            enable_markdown_patterns=False,
        )

    def with_loader(self, loader: BaseLoader) -> "DocumentBuilder":
        """Set custom loader."""
        self.loader = loader
        return self

    def with_splitter(self, splitter: BaseSplitter) -> "DocumentBuilder":
        """Set custom splitter."""
        self.splitter = splitter
        return self

    def with_ollama_client(self, client: OllamaClient) -> "DocumentBuilder":
        """Set Ollama client for embedding operations."""
        self.ollama_client = client
        return self

    def with_mineru_client(self, client: MineruClient) -> "DocumentBuilder":
        """Set Mineru client for PDF markdown conversion."""
        self.mineru_client = client
        return self
