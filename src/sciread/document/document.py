"""Main Document class for managing document content and chunks."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional
from typing import Union

from ..logging_config import get_logger
from .document_builder import DocumentBuilder
from .document_builder import DocumentFactory
from .models import Chunk
from .models import DocumentMetadata
from .models import ProcessingState


class Document:
    """Data container for document content and chunks."""

    def __init__(
        self,
        source_path: Optional[Path] = None,
        text: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        processing_state: Optional[ProcessingState] = None,
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
        self.source_path = source_path
        self._raw_text = text or ""
        self.metadata = metadata or DocumentMetadata(source_path=source_path)
        self.processing_state = processing_state or ProcessingState()
        self._chunks: list[Chunk] = []
        self._split = False
        self._is_markdown = _is_markdown

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        to_markdown: bool = False,
        auto_split: bool = True,
        **split_kwargs
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
        return DocumentFactory.create_from_file(file_path, to_markdown=to_markdown)

    @classmethod
    def from_text(
        cls,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
        auto_split: bool = True,
        **split_kwargs
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
        return DocumentFactory.create_from_text(text, metadata=metadata)

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

    def get_chunks(
        self,
        processed: Optional[bool] = None,
        chunk_type: Optional[str] = None,
        limit: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
    ) -> list[Chunk]:
        """
        Get chunks with flexible filtering criteria.

        This is the unified method for accessing chunks with any combination of filters.
        It replaces the previous multiple specialized methods (get_unprocessed_chunks,
        get_quality_chunks, filter_chunks, etc.).

        Args:
            processed: Filter by processing status (True for processed, False for unprocessed).
            chunk_type: Filter by specific chunk type.
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
            doc.get_chunks(chunk_type="abstract", processed=False)

            # Get first 5 chunks with high confidence
            doc.get_chunks(confidence_threshold=0.8, limit=5)
        """
        chunks = self._chunks

        # Filter by processing status
        if processed is not None:
            chunks = [chunk for chunk in chunks if chunk.processed == processed]

        # Filter by chunk type
        if chunk_type is not None:
            chunks = [chunk for chunk in chunks if chunk.chunk_type == chunk_type]

        # Filter by confidence threshold
        if confidence_threshold is not None:
            chunks = [chunk for chunk in chunks if (chunk.confidence or 0.0) >= confidence_threshold]

        # Filter by minimum length
        if min_length is not None:
            chunks = [chunk for chunk in chunks if len(chunk.content) >= min_length]

        # Exclude specific chunk types
        if exclude_types:
            chunks = [chunk for chunk in chunks if chunk.chunk_type not in exclude_types]

        # Apply limit
        if limit is not None:
            chunks = chunks[:limit]

        return chunks

    def get_unprocessed_chunks(self, limit: Optional[int] = None) -> list[Chunk]:
        """Get unprocessed chunks. Convenience method for get_chunks(processed=False)."""
        return self.get_chunks(processed=False, limit=limit)

    def get_quality_chunks(
        self,
        confidence_threshold: float = 0.5,
        min_length: int = 50,
        exclude_types: Optional[set[str]] = None,
        processed: Optional[bool] = None,
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

    def mark_chunks_processed(
        self,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
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
        chunks_to_mark = self.get_chunks(
            processed=False,
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
        )

        # Mark chunks as processed
        for chunk in chunks_to_mark:
            chunk.mark_processed()

        if chunks_to_mark:
            self.processing_state.add_note(f"Marked {len(chunks_to_mark)} chunks as processed")
            self.logger.info(f"Marked {len(chunks_to_mark)} chunks as processed")

        return len(chunks_to_mark)

    def next_unprocessed(self) -> Optional[Chunk]:
        """Get the next unprocessed chunk."""
        unprocessed = self.get_unprocessed_chunks(limit=1)
        return unprocessed[0] if unprocessed else None

    def mark_all_processed(self) -> None:
        """Mark all chunks as processed."""
        for chunk in self._chunks:
            chunk.mark_processed()
        self.processing_state.update_timestamp("processed")
        self.processing_state.add_note("All chunks marked as processed")

    def mark_all_unprocessed(self) -> None:
        """Mark all chunks as unprocessed."""
        for chunk in self._chunks:
            chunk.mark_unprocessed()

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Get the full text by joining all chunks with separator."""
        if not self._chunks:
            return self._raw_text
        return separator.join(chunk.content for chunk in self._chunks)

  
    def get_section_names(self) -> list[str]:
        """Return ordered list of section names from all chunks.

        Section names are extracted from chunk metadata if available.
        Chunks without section names are skipped.

        Returns:
            List of section names in document order.
        """
        section_names = []
        for chunk in self._chunks:
            if (chunk.metadata and
                'section_name' in chunk.metadata and
                chunk.metadata['section_name']):
                section_name = chunk.metadata['section_name']
                if section_name not in section_names:  # Avoid duplicates
                    section_names.append(section_name)
        return section_names

    def get_sections_by_name(self, section_names: list[str]) -> list[Chunk]:
        """Get chunks matching specific section names.

        Args:
            section_names: List of section names to filter by.

        Returns:
            List of chunks matching the specified section names.
        """
        matching_chunks = []
        for chunk in self._chunks:
            if (chunk.metadata and
                'section_name' in chunk.metadata and
                chunk.metadata['section_name'] in section_names):
                matching_chunks.append(chunk)
        return matching_chunks

    
    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks."""
        return iter(self._chunks)

    def __len__(self) -> int:
        """Get the number of chunks."""
        return len(self._chunks)

    def __getitem__(self, index: Union[int, slice]) -> Union[Chunk, list[Chunk]]:
        """Get chunk by index."""
        return self._chunks[index]
