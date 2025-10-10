"""Main Document class for managing processed documents."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional
from typing import Union

from ..logging_config import get_logger
from .loaders import BaseLoader
from .loaders import LoadResult
from .loaders.pdf_loader import PdfLoader
from .loaders.txt_loader import TxtLoader
from .models import Chunk
from .models import CoverageStats
from .models import DocumentMetadata
from .models import ProcessingState
from .splitters import BaseSplitter
from .splitters.regex_section_splitter import RegexSectionSplitter


class Document:
    """Main class for managing document processing and state."""

    def __init__(
        self,
        source_path: Optional[Path] = None,
        text: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        processing_state: Optional[ProcessingState] = None,
    ):
        """Initialize a Document instance."""
        self.logger = get_logger(__name__)
        self.source_path = source_path
        self._raw_text = text or ""
        self.metadata = metadata or DocumentMetadata(source_path=source_path)
        self.processing_state = processing_state or ProcessingState()
        self._chunks: list[Chunk] = []
        self._loaded = False
        self._split = False

        # Initialize default loaders
        self._loaders: list[BaseLoader] = [PdfLoader(), TxtLoader()]

        # Initialize default splitter
        self._splitter: BaseSplitter = RegexSectionSplitter()

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Document":
        """Create a Document from a file path."""
        path = Path(file_path)
        logger = get_logger(__name__)
        logger.debug(f"Creating document from file: {path}")
        return cls(source_path=path)

    @classmethod
    def from_text(cls, text: str, metadata: Optional[DocumentMetadata] = None) -> "Document":
        """Create a Document from raw text."""
        logger = get_logger(__name__)
        logger.debug(f"Creating document from text ({len(text)} characters)")
        doc = cls(text=text, metadata=metadata)
        doc._loaded = True  # Text-based documents are already "loaded"
        doc.processing_state.update_timestamp("loaded")
        doc.processing_state.add_note("Document created from text")
        return doc

    def load(self) -> LoadResult:
        """Load the document from the source path."""
        if self.source_path is None:
            self.logger.error("No source path specified for loading")
            raise ValueError("No source path specified for loading")

        self.logger.info(f"Loading document from: {self.source_path}")

        # Find appropriate loader
        loader = None
        for available_loader in self._loaders:
            if available_loader.can_load(self.source_path):
                loader = available_loader
                break

        if loader is None:
            self.logger.error(f"No loader available for file: {self.source_path}")
            raise ValueError(f"No loader available for file: {self.source_path}")

        self.logger.debug(f"Using loader: {loader.loader_name}")

        # Load the document
        result = loader.load(self.source_path)
        if result.success:
            self._raw_text = result.text
            self.metadata = result.metadata
            self._loaded = True
            self.processing_state.update_timestamp("loaded")
            self.processing_state.add_note(f"Document loaded using {loader.loader_name}")
            self.logger.info(f"Successfully loaded document using {loader.loader_name}: {len(result.text)} characters")

            # Add any warnings to processing state
            for warning in result.warnings:
                self.processing_state.add_note(f"Warning: {warning}")
                self.logger.warning(f"Document loading warning: {warning}")
        else:
            self.logger.error("Failed to load document")

        return result

    def split(
        self,
        splitter: Optional[BaseSplitter] = None,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
        auto_filter_quality: bool = False,
        quality_confidence_threshold: float = 0.3,
        quality_min_length: int = 20,
    ) -> list[Chunk]:
        """Split the document into chunks with optional quality filtering."""
        if not self._loaded:
            self.logger.error("Attempted to split document before loading")
            raise ValueError("Document must be loaded before splitting")

        if not self._raw_text.strip():
            self.logger.error("Attempted to split empty document")
            raise ValueError("Cannot split empty document")

        # Use provided splitter or default
        active_splitter = splitter or self._splitter
        self.logger.info(f"Splitting document using {active_splitter.splitter_name}")

        # Split the text
        self._chunks = active_splitter.split(self._raw_text)
        self._split = True
        self.processing_state.update_timestamp("split")
        self.processing_state.add_note(f"Document split using {active_splitter.splitter_name}")

        self.logger.info(f"Document split into {len(self._chunks)} chunks")

        # Apply quality filtering if requested
        if auto_filter_quality:
            deactivated_count = self.deactivate_low_quality_chunks(
                confidence_threshold=quality_confidence_threshold,
                min_length=quality_min_length,
                exclude_types=exclude_types,
            )
            if deactivated_count > 0:
                self.logger.info(f"Auto-filtered {deactivated_count} low-quality chunks")

        # Apply manual filtering if criteria provided
        elif confidence_threshold is not None or min_length is not None or exclude_types:
            # Get filtered chunks for return, but keep all chunks in self._chunks
            filtered_chunks = self.get_chunks(
                confidence_threshold=confidence_threshold,
                min_length=min_length,
                exclude_types=exclude_types,
            )
            self.logger.info(f"Returning {len(filtered_chunks)} filtered chunks out of {len(self._chunks)} total")
            return filtered_chunks

        return self._chunks

    @property
    def chunks(self) -> list[Chunk]:
        """Get all chunks."""
        return self._chunks.copy()

    @property
    def text(self) -> str:
        """Get the raw text content."""
        return self._raw_text

    @property
    def is_loaded(self) -> bool:
        """Check if document is loaded."""
        return self._loaded

    @property
    def is_split(self) -> bool:
        """Check if document is split."""
        return self._split

    def get_chunks(
        self,
        processed: Optional[bool] = None,
        chunk_type: Optional[str] = None,
        limit: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
    ) -> list[Chunk]:
        """Get chunks with optional filtering."""
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
        """Get unprocessed chunks."""
        return self.get_chunks(processed=False, limit=limit)

    def get_quality_chunks(
        self,
        confidence_threshold: float = 0.5,
        min_length: int = 50,
        exclude_types: Optional[set[str]] = None,
        processed: Optional[bool] = None,
    ) -> list[Chunk]:
        """Get high-quality chunks based on quality criteria."""
        return self.get_chunks(
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
            processed=processed,
        )

    def filter_chunks(self, **filter_kwargs) -> list[Chunk]:
        """Filter chunks using any combination of criteria."""
        return self.get_chunks(**filter_kwargs)

    def deactivate_low_quality_chunks(
        self,
        confidence_threshold: float = 0.3,
        min_length: int = 20,
        exclude_types: Optional[set[str]] = None,
    ) -> int:
        """
        Deactivate low-quality chunks by marking them as processed (so they're skipped).

        Returns the number of chunks deactivated.
        """
        low_quality_chunks = self.get_chunks(
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
            processed=False,
        )

        # Mark low-quality chunks as processed so they won't be included in processing
        for chunk in low_quality_chunks:
            chunk.mark_processed()

        if low_quality_chunks:
            self.processing_state.add_note(f"Deactivated {len(low_quality_chunks)} low-quality chunks")
            self.logger.info(f"Deactivated {len(low_quality_chunks)} low-quality chunks")

        return len(low_quality_chunks)

    def next_unprocessed(self) -> Optional[Chunk]:
        """Get the next unprocessed chunk."""
        unprocessed = self.get_unprocessed_chunks(limit=1)
        return unprocessed[0] if unprocessed else None

    def get_coverage(self) -> CoverageStats:
        """Calculate coverage statistics."""
        total_chunks = len(self._chunks)
        processed_chunks = sum(1 for chunk in self._chunks if chunk.processed)

        total_words = sum(chunk.word_count for chunk in self._chunks)
        processed_words = sum(chunk.word_count for chunk in self._chunks if chunk.processed)

        return CoverageStats(
            processed_chunks=processed_chunks,
            total_chunks=total_chunks,
            processed_words=processed_words,
            total_words=total_words,
        )

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

    def search(self, query: str, case_sensitive: bool = False) -> list[Chunk]:
        """Search for text in chunks."""
        if not case_sensitive:
            query = query.lower()

        matching_chunks = []
        for chunk in self._chunks:
            search_text = chunk.content if case_sensitive else chunk.content.lower()
            if query in search_text:
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
