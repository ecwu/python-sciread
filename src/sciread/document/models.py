"""Core data models for document processing."""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Union


@dataclass
class Chunk:
    """A chunk of text from a document with metadata."""

    content: str
    chunk_type: str = "unknown"  # abstract, introduction, methods, etc.
    position: int = 0  # Sequential position in document
    page_range: Optional[tuple[int, int]] = None  # (start_page, end_page)
    char_range: Optional[tuple[int, int]] = None  # (start_char, end_char)
    word_count: int = 0
    confidence: float = 1.0  # Confidence in classification (0.0-1.0)
    processed: bool = False  # Processing status

    def __post_init__(self):
        """Validate and initialize derived fields."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def toggle_processed(self) -> None:
        """Toggle the processed status of this chunk."""
        self.processed = not self.processed

    def mark_processed(self) -> None:
        """Mark this chunk as processed."""
        self.processed = True

    def mark_unprocessed(self) -> None:
        """Mark this chunk as unprocessed."""
        self.processed = False

    @property
    def is_processed(self) -> bool:
        """Check if this chunk is processed."""
        return self.processed


@dataclass
class CoverageStats:
    """Statistics about document coverage."""

    processed_chunks: int = 0
    total_chunks: int = 0
    processed_words: int = 0
    total_words: int = 0

    @property
    def chunk_coverage(self) -> float:
        """Calculate chunk coverage percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    @property
    def word_coverage(self) -> float:
        """Calculate word coverage percentage."""
        if self.total_words == 0:
            return 0.0
        return (self.processed_words / self.total_words) * 100

    def to_dict(self) -> dict[str, Union[int, float]]:
        """Convert to dictionary for serialization."""
        return {
            "processed_chunks": self.processed_chunks,
            "total_chunks": self.total_chunks,
            "processed_words": self.processed_words,
            "total_words": self.total_words,
            "chunk_coverage": self.chunk_coverage,
            "word_coverage": self.word_coverage,
        }


@dataclass
class DocumentMetadata:
    """Metadata about the source document."""

    source_path: Optional[Path] = None
    file_type: Optional[str] = None  # pdf, txt, etc.
    file_size: Optional[int] = None  # in bytes
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.modified_at is None:
            self.modified_at = now


@dataclass
class ProcessingState:
    """State information about document processing."""

    loaded_at: Optional[datetime] = None
    split_at: Optional[datetime] = None
    last_processed_at: Optional[datetime] = None
    processing_version: str = "1.0"
    notes: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        """Add a processing note."""
        self.notes.append(f"{datetime.now().isoformat()}: {note}")

    def update_timestamp(self, operation: str) -> None:
        """Update the appropriate timestamp for an operation."""
        timestamp = datetime.now()
        if operation == "loaded":
            self.loaded_at = timestamp
        elif operation == "split":
            self.split_at = timestamp
        elif operation == "processed":
            self.last_processed_at = timestamp
