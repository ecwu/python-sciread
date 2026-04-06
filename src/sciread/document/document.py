"""Main Document class for managing document content and chunks."""

from collections.abc import Iterator
from pathlib import Path

from ..embedding_provider import get_embedding_client
from ..platform.config import get_config
from ..platform.logging import get_logger
from .document_builder import DocumentBuilder
from .models import Chunk
from .models import DocumentMetadata
from .models import ProcessingState
from .retrieval.service import build_vector_index as build_document_vector_index
from .retrieval.service import cosine_similarity
from .retrieval.service import semantic_search as semantic_search_document
from .retrieval.vector_index import VectorIndex
from .state import attach_document_chunks
from .state import get_chunk_by_id as get_document_chunk_by_id
from .state import get_chunks as get_document_chunks
from .state import get_chunks_by_section as get_document_chunks_by_section
from .state import get_full_text as get_document_full_text
from .state import get_neighbor_chunks as get_document_neighbor_chunks
from .state import get_section_names as get_document_section_names
from .state import get_section_parts as get_document_section_parts
from .state import get_sections_by_name as get_document_sections_by_name
from .state import initialize_document
from .state import mark_all_processed as mark_document_all_processed
from .state import mark_all_unprocessed as mark_document_all_unprocessed
from .state import mark_chunks_processed as mark_document_chunks_processed
from .state import next_unprocessed_chunk
from .state import update_chunk_index
from .structure.chunking import build_doc_id
from .structure.chunking import build_retrieval_text
from .structure.chunking import calculate_file_hash
from .structure.chunking import enrich_chunks
from .structure.chunking import to_plain_text
from .structure.persistence import load_document
from .structure.persistence import save_document
from .structure.renderers import clean_section_content
from .structure.renderers import collect_sections as collect_document_sections
from .structure.renderers import format_for_human
from .structure.renderers import format_for_llm
from .structure.renderers import get_section_overview as build_section_overview
from .structure.renderers import get_sections_with_confidence as collect_sections_with_confidence
from .structure.renderers import remove_references_section
from .structure.renderers import resolve_section_names
from .structure.sections import get_closest_section_name as resolve_closest_section_name
from .structure.sections import match_section_pattern
from .structure.sections import prefix_similarity
from .structure.sections import word_similarity


class Document:
    """Data container for document content and chunks."""

    def __init__(
        self,
        source_path: Path | None = None,
        text: str | None = None,
        metadata: DocumentMetadata | None = None,
        processing_state: ProcessingState | None = None,
        _is_markdown: bool = False,
    ):
        """Initialize a Document instance.

        Args:
            source_path: Path to the source file (optional).
            text: Raw text content.
            metadata: Document metadata.
            processing_state: Processing state information.
            _is_markdown: Internal flag to track if content is markdown-converted.
        """
        self.logger = get_logger(__name__)
        initialize_document(
            self,
            source_path=source_path,
            text=text,
            metadata=metadata,
            processing_state=processing_state,
            is_markdown=_is_markdown,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
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
        return DocumentBuilder().from_file(
            file_path,
            to_markdown=to_markdown,
            auto_split=auto_split,
            **split_kwargs,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        metadata: DocumentMetadata | None = None,
        auto_split: bool = True,
        **split_kwargs,
    ) -> "Document":
        """
        Create a Document from raw text.

        Args:
            text: Raw text content.
            metadata: Optional document metadata.
            auto_split: Whether to automatically split the document.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with text and optionally split chunks.
        """
        return DocumentBuilder().from_text(
            text,
            metadata=metadata,
            auto_split=auto_split,
            **split_kwargs,
        )

    @property
    def chunks(self) -> list[Chunk]:
        """Get all chunks."""
        return self._chunks.copy()

    @property
    def text(self) -> str:
        """Get the raw text content."""
        return self._raw_text

    @property
    def is_split(self) -> bool:
        """Check if document is split."""
        return self._split

    @property
    def is_markdown(self) -> bool:
        """Check if document content is markdown-converted."""
        return self._is_markdown

    @property
    def vector_index(self) -> VectorIndex | None:
        """Return the runtime vector index."""
        return self._runtime.vector_index

    @vector_index.setter
    def vector_index(self, value: VectorIndex | None) -> None:
        """Store the runtime vector index."""
        self._runtime.vector_index = value

    def get_chunks(
        self,
        processed: bool | None = None,
        chunk_name: str | None = None,
        limit: int | None = None,
        confidence_threshold: float | None = None,
        min_length: int | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[Chunk]:
        """
        Get chunks with flexible filtering criteria.

        This is the unified method for accessing chunks with any combination of filters.
        It replaces the previous multiple specialized methods (get_unprocessed_chunks,
        get_quality_chunks, filter_chunks, etc.).

        Args:
            processed: Filter by processing status (True for processed, False for unprocessed).
            chunk_name: Filter by specific chunk name (section name).
            limit: Maximum number of chunks to return.
            confidence_threshold: Minimum confidence score (0.0-1.0).
            min_length: Minimum character length for chunks.
            exclude_types: Set of chunk types to exclude.

        Returns:
            List of chunks matching the specified criteria.

        Examples:
            # Get all unprocessed chunks
            doc.get_chunks(processed=False)

            # Get high-quality chunks
            doc.get_chunks(confidence_threshold=0.7, min_length=100)

            # Get abstract chunks that haven't been processed yet
            doc.get_chunks(chunk_name="abstract", processed=False)

            # Get first 5 chunks with high confidence
            doc.get_chunks(confidence_threshold=0.8, limit=5)
        """
        return get_document_chunks(
            self,
            processed=processed,
            chunk_name=chunk_name,
            limit=limit,
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
        )

    def get_unprocessed_chunks(self, limit: int | None = None) -> list[Chunk]:
        """Get unprocessed chunks. Convenience method for get_chunks(processed=False)."""
        return self.get_chunks(processed=False, limit=limit)

    def get_quality_chunks(
        self,
        confidence_threshold: float = 0.5,
        min_length: int = 50,
        exclude_types: set[str] | None = None,
        processed: bool | None = None,
    ) -> list[Chunk]:
        """Get high-quality chunks based on quality criteria. Convenience method."""
        return self.get_chunks(
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
            processed=processed,
        )

    def filter_chunks(self, **filter_kwargs) -> list[Chunk]:
        """Filter chunks using any combination of criteria. Alias for get_chunks()."""
        return self.get_chunks(**filter_kwargs)

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by chunk ID.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Matching chunk if found, otherwise None.
        """
        return get_document_chunk_by_id(self, chunk_id)

    def get_chunks_by_section(
        self,
        section: str,
        include_subsections: bool = True,
    ) -> list[Chunk]:
        """Get chunks by section label or section path.

        Args:
            section: Section name/path, e.g. ``methods`` or ``methods > setup``.
            include_subsections: Whether to include descendant sections when a
                section path prefix is matched.

        Returns:
            Chunks that belong to the requested section.
        """
        return get_document_chunks_by_section(
            self,
            section,
            include_subsections=include_subsections,
        )

    def _get_section_parts(self, chunk: Chunk) -> list[str]:
        """Get normalized section path parts for a chunk."""
        return get_document_section_parts(chunk)

    def get_neighbor_chunks(
        self,
        chunk_id: str,
        before: int = 1,
        after: int = 1,
        include_self: bool = True,
    ) -> list[Chunk]:
        """Get neighboring chunks around a chunk.

        Args:
            chunk_id: Center chunk ID.
            before: Number of chunks before the center chunk.
            after: Number of chunks after the center chunk.
            include_self: Whether to include the center chunk in results.

        Returns:
            Neighboring chunks in document order. Returns an empty list when
            chunk_id is not found.
        """
        return get_document_neighbor_chunks(
            self,
            chunk_id,
            before=before,
            after=after,
            include_self=include_self,
        )

    def mark_chunks_processed(
        self,
        confidence_threshold: float | None = None,
        min_length: int | None = None,
        exclude_types: set[str] | None = None,
    ) -> int:
        """
        Mark chunks as processed based on filtering criteria.

        This replaces the old deactivate_low_quality_chunks method with clearer semantics.

        Args:
            confidence_threshold: Mark chunks below this threshold as processed.
            min_length: Mark chunks shorter than this as processed.
            exclude_types: Don't mark these chunk types as processed.

        Returns:
            Number of chunks marked as processed.
        """
        return mark_document_chunks_processed(
            self,
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
        )

    def next_unprocessed(self) -> Chunk | None:
        """Get the next unprocessed chunk."""
        return next_unprocessed_chunk(self)

    def mark_all_processed(self) -> None:
        """Mark all chunks as processed."""
        mark_document_all_processed(self)

    def mark_all_unprocessed(self) -> None:
        """Mark all chunks as unprocessed."""
        mark_document_all_unprocessed(self)

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Get the full text by joining all chunks with separator."""
        return get_document_full_text(self, separator=separator)

    def get_section_names(self) -> list[str]:
        """Return ordered list of section names from all chunks.

        Section names are extracted from chunk metadata if available.
        Chunks without section names are skipped.

        Returns:
            List of section names in document order.
        """
        return get_document_section_names(self)

    def get_sections_by_name(self, section_names: list[str]) -> list[Chunk]:
        """Get chunks matching specific section names.

        Args:
            section_names: List of section names to filter by.

        Returns:
            List of chunks matching the specified section names.
        """
        return get_document_sections_by_name(self, section_names)

    def _resolve_section_names(
        self,
        section_names: list[str] | None = None,
        max_sections: int | None = None,
    ) -> list[str]:
        """Resolve section names to use, honoring ordering and limits."""
        return resolve_section_names(
            self,
            section_names=section_names,
            max_sections=max_sections,
        )

    def _collect_sections(
        self,
        section_names: list[str] | None = None,
        max_sections: int | None = None,
        clean_text: bool = False,
        max_chars_per_section: int | None = None,
    ) -> list[dict]:
        """Collect section data (name, content, chunks, stats) in document order."""
        return collect_document_sections(
            self,
            section_names=section_names,
            max_sections=max_sections,
            clean_text=clean_text,
            max_chars_per_section=max_chars_per_section,
        )

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks."""
        return iter(self._chunks)

    def __len__(self) -> int:
        """Get the number of chunks."""
        return len(self._chunks)

    def __getitem__(self, index: int | slice) -> Chunk | list[Chunk]:
        """Get chunk by index."""
        return self._chunks[index]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        return calculate_file_hash(file_path, self.logger)

    def _update_chunks_by_id(self) -> None:
        """Update the _chunks_by_id dictionary to match current chunks."""
        update_chunk_index(self)

    def _build_doc_id(self) -> str:
        """Build a stable document ID used by chunk metadata."""
        return build_doc_id(self)

    def _enrich_chunks_metadata(self, chunks: list[Chunk]) -> None:
        """Fill normalized chunk metadata fields and maintain linkage."""
        enrich_chunks(self, chunks)

    def _build_retrieval_text(self, section_path: list[str], content_plain: str) -> str:
        """Compose retrieval text used by embeddings/rerank flows."""
        return build_retrieval_text(section_path, content_plain)

    def _to_plain_text(self, text: str) -> str:
        """Convert markdown-ish text into plain text for lexical retrieval."""
        return to_plain_text(text)

    def _set_chunks(self, chunks: list[Chunk]) -> None:
        """Set chunks and update the _chunks_by_id dictionary."""
        attach_document_chunks(self, chunks)

    def build_vector_index(self, persist: bool = False, embedding_client=None) -> None:
        """Builds a semantic vector index from the document's chunks.

        Args:
            persist: Whether to persist the index to disk
            embedding_client: Optional embedding client (OllamaClient or SiliconFlowClient).
                            If not provided, uses OllamaClient from config.
        """
        build_document_vector_index(
            self,
            persist=persist,
            embedding_client=embedding_client,
            get_config_fn=get_config,
            get_embedding_client_fn=get_embedding_client,
            vector_index_cls=VectorIndex,
        )

    def semantic_search(self, query: str, top_k: int = 5, return_scores: bool = False) -> list[Chunk] | list[tuple[Chunk, float]]:
        """Performs a semantic search on the document chunks using cosine similarity.

        This method uses cosine similarity for ranking, which is length-invariant
        and provides better semantic matching than L2 distance. Results are ranked
        by similarity score (higher is better).

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: If True, returns tuples of (chunk, similarity_score)

        Returns:
            List of matching chunks, or list of (chunk, score) tuples if return_scores=True
            Similarity scores are in range [0, 1] where 1 is most similar
        """
        return semantic_search_document(
            self,
            query=query,
            top_k=top_k,
            return_scores=return_scores,
            get_config_fn=get_config,
            get_embedding_client_fn=get_embedding_client,
        )

    # ========== Unified Section Handling Methods ==========

    def print_for_human(
        self,
        section_names: list[str] | None = None,
        max_sections: int | None = None,
        show_metadata: bool = True,
        show_confidence: bool = True,
        use_colors: bool = True,
    ) -> str:
        """Render formatted document sections for human reading.

        Args:
            section_names: Specific section names to display. If None, shows all sections.
            max_sections: Maximum number of sections to display.
            show_metadata: Whether to show document metadata (title, author, etc.).
            show_confidence: Whether to show confidence scores for each section.
            use_colors: Whether to use colored output (if terminal supports it).

        Returns:
            Rendered human-readable content.
        """
        try:
            return format_for_human(
                self,
                section_names=section_names,
                max_sections=max_sections,
                show_metadata=show_metadata,
                show_confidence=show_confidence,
                use_colors=use_colors,
            )

        except Exception as e:
            self.logger.error(f"Failed to print document for human: {e}")
            return f"Error rendering document: {e}"

    def get_for_llm(
        self,
        section_names: list[str] | None = None,
        max_tokens: int | None = None,
        include_headers: bool = True,
        clean_text: bool = True,
        max_chars_per_section: int | None = None,
    ) -> str:
        """Get document content optimized for LLM consumption.

        Args:
            section_names: Specific section names to include. If None, includes all sections.
            max_tokens: Approximate maximum number of tokens (rough estimation: 1 token ≈ 4 chars).
            include_headers: Whether to include section headers.
            clean_text: Whether to clean and normalize text content.
            max_chars_per_section: Maximum characters per section to prevent overflow.

        Returns:
            Formatted document content as string.
        """
        return format_for_llm(
            self,
            section_names=section_names,
            max_tokens=max_tokens,
            include_headers=include_headers,
            clean_text=clean_text,
            max_chars_per_section=max_chars_per_section,
        )

    def get_closest_section_name(
        self,
        target_name: str,
        available_names: list[str] | None = None,
        case_sensitive: bool = False,
        threshold: float = 0.8,
        use_embedding: bool = False,
    ) -> str | None:
        """Find closest matching section name using various similarity measures.

        Args:
            target_name: Name to search for.
            available_names: List of section names to search in. If None, uses document's section names.
            case_sensitive: Whether matching should be case sensitive.
            threshold: Minimum similarity threshold (0.0-1.0).
            use_embedding: Whether to use semantic similarity via embeddings (if available).

        Returns:
            Closest matching section name if above threshold, None otherwise.
        """
        return resolve_closest_section_name(
            self,
            target_name=target_name,
            available_names=available_names,
            case_sensitive=case_sensitive,
            threshold=threshold,
            use_embedding=use_embedding,
        )

    def _match_section_pattern(self, search_name: str, normalized_names: list[str], original_names: list[str]) -> str | None:
        """Match section using common academic paper patterns."""
        return match_section_pattern(search_name, normalized_names, original_names)

    def _word_similarity(self, str1: str, str2: str) -> float:
        """Calculate word-level similarity between two strings."""
        return word_similarity(str1, str2)

    def _prefix_similarity(self, str1: str, str2: str) -> float:
        """Calculate prefix similarity between two strings."""
        return prefix_similarity(str1, str2)

    def get_section_overview(self, include_stats: bool = True, include_quality: bool = True) -> dict:
        """Get comprehensive overview of all sections.

        Args:
            include_stats: Whether to include section statistics.
            include_quality: Whether to include quality metrics.

        Returns:
            Dictionary with section information.
        """
        return build_section_overview(
            self,
            include_stats=include_stats,
            include_quality=include_quality,
        )

    def get_sections_with_confidence(self, min_confidence: float = 0.5, min_length: int = 50) -> list[tuple[str, str]]:
        """Get sections that meet minimum quality criteria.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0).
            min_length: Minimum character length per section.

        Returns:
            List of (section_name, content) tuples that meet criteria.
        """
        return collect_sections_with_confidence(
            self,
            min_confidence=min_confidence,
            min_length=min_length,
        )

    # ========== Helper Methods ==========

    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content."""
        return clean_section_content(content)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return cosine_similarity(vec1, vec2)

    # ========== Agent-Specific Optimization Methods ==========

    def _remove_references_section(self, text: str) -> str:
        """Remove references and bibliography sections from text."""
        return remove_references_section(text)

    def save(self, output_path: Path) -> None:
        """Saves the document's state and its vector index path to a JSON file."""
        save_document(self, output_path)

    @classmethod
    def load(cls, state_path: Path) -> "Document":
        """Loads a document from a state file, re-linking its vector index."""
        return load_document(cls, state_path)
