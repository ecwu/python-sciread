"""Document builder class for creating and processing documents."""

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .document import Document

from ..document_ingestion.external_clients import MineruClient
from ..document_ingestion.loaders import BaseLoader
from ..document_ingestion.loaders import PdfLoader
from ..document_ingestion.loaders import TxtLoader
from ..embedding_provider import OllamaClient
from ..platform.logging import get_logger
from .models import DocumentMetadata
from .splitters import BaseSplitter
from .splitters import MarkdownSplitter
from .splitters import SemanticSplitter

TEXT_FILE_EXTENSIONS = {".txt", ".text", ".md", ".rst"}


class DocumentBuilder:
    """Builder class for creating and processing documents."""

    def __init__(
        self,
        loader: BaseLoader | None = None,
        splitter: BaseSplitter | None = None,
        ollama_client: OllamaClient | None = None,
        mineru_client: MineruClient | None = None,
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
        file_path: str | Path,
        to_markdown: bool = False,
        auto_split: bool = True,
        build_vector_index: bool = False,
        persist_index: bool = False,
        embedding_client: Any | None = None,
        **split_kwargs,
    ) -> "Document":
        """
        Create a Document from a file path.

        Args:
            file_path: Path to the file.
            to_markdown: If True, convert PDF to markdown using Mineru API.
            auto_split: Whether to automatically split the document after loading.
            build_vector_index: Whether to build vector index during builder finalize stage.
            persist_index: Whether to persist vector index to disk when build_vector_index is enabled.
            embedding_client: Optional embedding client used for vector index building.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with loaded content and optionally split chunks.
        """
        from .document import Document  # Import here to avoid circular imports

        path = Path(file_path)
        self.logger.debug(f"Creating document from file: {path} (to_markdown={to_markdown})")

        loader = self.loader or self._create_default_loader(path, to_markdown)

        # Load the document
        load_result = loader.load(path)
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
        doc.processing_state.add_note(f"Document loaded using {loader.loader_name}")

        # Add any warnings to processing state
        for warning in load_result.warnings:
            doc.processing_state.add_note(f"Warning: {warning}")
            self.logger.warning(f"Document loading warning: {warning}")

        self.logger.debug(f"Successfully loaded document using {loader.loader_name}: {len(load_result.text)} characters")

        # Auto-split if requested
        if auto_split:
            self._split_document(doc, **split_kwargs)

        return self.build(
            doc,
            build_vector_index=build_vector_index,
            persist_index=persist_index,
            embedding_client=embedding_client,
        )

    def from_text(
        self,
        text: str,
        metadata: DocumentMetadata | None = None,
        auto_split: bool = True,
        is_markdown: bool = False,
        build_vector_index: bool = False,
        persist_index: bool = False,
        embedding_client: Any | None = None,
        **split_kwargs,
    ) -> "Document":
        """
        Create a Document from raw text.

        Args:
            text: Raw text content.
            metadata: Optional document metadata.
            auto_split: Whether to automatically split the document.
            is_markdown: Whether the text is markdown format.
            build_vector_index: Whether to build vector index during builder finalize stage.
            persist_index: Whether to persist vector index to disk when build_vector_index is enabled.
            embedding_client: Optional embedding client used for vector index building.
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

        return self.build(
            doc,
            build_vector_index=build_vector_index,
            persist_index=persist_index,
            embedding_client=embedding_client,
        )

    def build(
        self,
        doc: "Document",
        build_vector_index: bool = False,
        persist_index: bool = False,
        embedding_client: Any | None = None,
    ) -> "Document":
        """Finalize a loaded/split document into agent-ready chunks.

        Pipeline:
            load raw content -> split into chunks -> enrich chunks -> attach retrieval metadata
            -> optionally build vector index -> return Document

        Args:
            doc: Document instance to finalize.
            build_vector_index: Whether to build vector index.
            persist_index: Whether to persist vector index to disk.
            embedding_client: Optional embedding client for indexing.

        Returns:
            Finalized document.
        """
        if build_vector_index:
            index_client = embedding_client or self.ollama_client
            doc.build_vector_index(persist=persist_index, embedding_client=index_client)

        return doc

    def _create_default_loader(self, file_path: Path, to_markdown: bool) -> BaseLoader:
        """Create appropriate loader based on file extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PdfLoader(
                to_markdown=to_markdown,
                mineru_client=self.mineru_client,
            )
        if suffix in TEXT_FILE_EXTENSIONS:
            return TxtLoader()

        raise ValueError(f"Unsupported file format: {suffix}")

    def _split_document(self, doc: "Document", splitter: BaseSplitter | None = None, **split_kwargs) -> None:
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
        doc.processing_state.add_note(f"Document split using {active_splitter.splitter_name}")

        self.logger.info(f"Document split into {len(chunks)} chunks using {active_splitter.splitter_name}")

    def _create_default_splitter(self, doc: "Document") -> BaseSplitter:
        """Create appropriate splitter based on document content."""
        # If document was created with markdown conversion, use the dedicated MarkdownSplitter
        if doc.is_markdown:
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
