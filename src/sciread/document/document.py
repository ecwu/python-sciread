"""Main Document class for managing processed documents."""

from collections.abc import Iterator
from pathlib import Path
from typing import Optional
from typing import Union

from .loaders import BaseLoader
from .loaders import LoadResult
from .loaders.pdf_loader import PdfLoader
from .loaders.txt_loader import TxtLoader
from .models import Chunk
from .models import CoverageStats
from .models import DocumentMetadata
from .models import ProcessingState
from .splitters import BaseSplitter
from .splitters.hybrid import HybridSplitter


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
        self._splitter: BaseSplitter = HybridSplitter()

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Document":
        """Create a Document from a file path."""
        path = Path(file_path)
        return cls(source_path=path)

    @classmethod
    def from_text(cls, text: str, metadata: Optional[DocumentMetadata] = None) -> "Document":
        """Create a Document from raw text."""
        return cls(text=text, metadata=metadata)

    def load(self) -> LoadResult:
        """Load the document from the source path."""
        if self.source_path is None:
            raise ValueError("No source path specified for loading")

        # Find appropriate loader
        loader = None
        for available_loader in self._loaders:
            if available_loader.can_load(self.source_path):
                loader = available_loader
                break

        if loader is None:
            raise ValueError(f"No loader available for file: {self.source_path}")

        # Load the document
        result = loader.load(self.source_path)
        if result.success:
            self._raw_text = result.text
            self.metadata = result.metadata
            self._loaded = True
            self.processing_state.update_timestamp("loaded")
            self.processing_state.add_note(f"Document loaded using {loader.loader_name}")

            # Add any warnings to processing state
            for warning in result.warnings:
                self.processing_state.add_note(f"Warning: {warning}")

        return result

    def split(self, splitter: Optional[BaseSplitter] = None) -> list[Chunk]:
        """Split the document into chunks."""
        if not self._loaded:
            raise ValueError("Document must be loaded before splitting")

        if not self._raw_text.strip():
            raise ValueError("Cannot split empty document")

        # Use provided splitter or default
        active_splitter = splitter or self._splitter

        # Split the text
        self._chunks = active_splitter.split(self._raw_text)
        self._split = True
        self.processing_state.update_timestamp("split")
        self.processing_state.add_note(f"Document split using {active_splitter.splitter_name}")

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
    ) -> list[Chunk]:
        """Get chunks with optional filtering."""
        chunks = self._chunks

        # Filter by processing status
        if processed is not None:
            chunks = [chunk for chunk in chunks if chunk.processed == processed]

        # Filter by chunk type
        if chunk_type is not None:
            chunks = [chunk for chunk in chunks if chunk.chunk_type == chunk_type]

        # Apply limit
        if limit is not None:
            chunks = chunks[:limit]

        return chunks

    def get_unprocessed_chunks(self, limit: Optional[int] = None) -> list[Chunk]:
        """Get unprocessed chunks."""
        return self.get_chunks(processed=False, limit=limit)

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
