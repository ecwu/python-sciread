"""Document builder class for creating and processing documents."""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document

from ..embedding_provider import OllamaClient
from ..logging_config import get_logger
from .external_clients import MineruClient
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
        embedding_client=None,
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
        embedding_client=None,
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
        embedding_client=None,
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
        if doc.is_split and doc.chunks:
            self._enrich_chunks(doc, doc._chunks)
            doc._set_chunks(doc._chunks)

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
        elif suffix in [".txt", ".md", ".rst"]:
            return TxtLoader()
        else:
            # Try to use available loaders
            for loader_class in [PdfLoader, TxtLoader]:
                if suffix in loader_class(None).supported_extensions:
                    return loader_class()

            raise ValueError(f"Unsupported file format: {suffix}")

    def _split_document(
        self, doc: "Document", splitter: BaseSplitter | None = None, **split_kwargs
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
        self._enrich_chunks(doc, chunks)
        doc._set_chunks(chunks)
        doc.processing_state.add_note(
            f"Document split using {active_splitter.splitter_name}"
        )

        self.logger.info(
            f"Document split into {len(chunks)} chunks using {active_splitter.splitter_name}"
        )

    def _enrich_chunks(self, doc: "Document", chunks: list) -> None:
        """Enrich chunks with normalized retrieval metadata and derived texts."""
        doc_id = self._build_doc_id(doc)

        for i, chunk in enumerate(chunks):
            chunk.doc_id = chunk.doc_id or doc_id
            chunk.para_index = i
            chunk.position = i

            if (
                not chunk.section_path
                and chunk.chunk_name
                and chunk.chunk_name != "unknown"
            ):
                chunk.section_path = [chunk.chunk_name]

            if not chunk.content_plain or chunk.content_plain == chunk.content:
                chunk.content_plain = self._to_plain_text(chunk.content)

            if not chunk.display_text:
                chunk.display_text = chunk.content

            if (
                not chunk.retrieval_text
                or chunk.retrieval_text == chunk.content
                or chunk.retrieval_text == chunk.content_plain
            ):
                chunk.retrieval_text = self._build_retrieval_text(
                    chunk.section_path, chunk.content_plain
                )

            if chunk.token_count is None:
                chunk.token_count = len(chunk.content_plain.split())

            if chunk.page_range is not None:
                if chunk.page_start is None:
                    chunk.page_start = chunk.page_range[0]
                if chunk.page_end is None:
                    chunk.page_end = chunk.page_range[1]
            elif chunk.page_start is not None and chunk.page_end is not None:
                chunk.page_range = (chunk.page_start, chunk.page_end)

            if not chunk.parent_section_id and chunk.section_path:
                chunk.parent_section_id = chunk.section_path[-1]

            if not chunk.citation_key or chunk.citation_key == chunk.chunk_id:
                chunk.citation_key = f"{chunk.doc_id}:{chunk.position}"

            chunk.metadata["section_label"] = (
                " > ".join(chunk.section_path) if chunk.section_path else ""
            )

        for i, chunk in enumerate(chunks):
            chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            chunk.next_chunk_id = (
                chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
            )

    def _build_doc_id(self, doc: "Document") -> str:
        """Build a stable document ID used by chunk metadata."""
        return doc.metadata.file_hash or (
            Path(doc.metadata.source_path).stem
            if doc.metadata.source_path
            else "unnamed_document"
        )

    def _build_retrieval_text(self, section_path: list[str], content_plain: str) -> str:
        """Compose retrieval text used by embeddings/rerank flows."""
        if section_path:
            section_label = " > ".join(section_path)
            return f"[Section] {section_label}\n\n{content_plain}"
        return content_plain

    def _to_plain_text(self, text: str) -> str:
        """Convert markdown-ish text into plain text for lexical retrieval."""
        plain = text
        plain = re.sub(r"```.*?```", " ", plain, flags=re.DOTALL)
        plain = re.sub(r"`([^`]+)`", r"\1", plain)
        plain = re.sub(r"^#{1,6}\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
        plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
        plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
        plain = re.sub(r"__([^_]+)__", r"\1", plain)
        plain = re.sub(r"\*([^*]+)\*", r"\1", plain)
        plain = re.sub(r"_([^_]+)_", r"\1", plain)
        plain = re.sub(r"^\s{0,3}>\s?", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"^\s*[-*+]\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"^\s*\d+\.\s+", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"<[^>]+>", " ", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        return plain

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
