"""Core data models for document processing."""

import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    """A chunk of text from a document with metadata."""

    content: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    content_plain: str = ""
    retrieval_text: str = ""
    display_text: str = ""
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    para_index: int | None = None
    char_range: tuple[int, int] | None = None  # (start_char, end_char)
    overlap_prev_chars: int = 0
    overlap_next_chars: int = 0
    token_count: int | None = None

    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    parent_section_id: str | None = None

    citation_key: str = ""
    retrievable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize derived fields."""
        if not self.display_text:
            self.display_text = self.content

        if not self.content_plain:
            self.content_plain = self.content

        if not self.retrieval_text:
            self.retrieval_text = self.content_plain

        if self.token_count is None:
            # Approximate token count fallback when no tokenizer is provided.
            self.token_count = len(self.content_plain.split())

        if self.overlap_prev_chars < 0 or self.overlap_next_chars < 0:
            raise ValueError("Chunk overlap values must be >= 0")

        if self.parent_section_id is None and self.section_path:
            self.parent_section_id = self.section_path[-1]

        if not self.citation_key:
            self.citation_key = self.chunk_id

    @property
    def has_overlap(self) -> bool:
        """Check whether this chunk overlaps with an adjacent chunk."""
        return self.overlap_prev_chars > 0 or self.overlap_next_chars > 0


@dataclass
class DocumentMetadata:
    """Metadata about source document."""

    source_path: Path | None = None
    file_type: str | None = None  # pdf, txt, etc.
    file_size: int | None = None  # in bytes
    file_hash: str | None = None  # hash of file content for identification
    created_at: datetime | None = None
    modified_at: datetime | None = None
    title: str | None = None
    author: str | None = None
    page_count: int | None = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = now
        if self.modified_at is None:
            self.modified_at = now


@dataclass
class ProcessingState:
    """State information about document processing."""

    loaded_at: datetime | None = None
    split_at: datetime | None = None
    processing_version: str = "1.0"
    notes: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        """Add a processing note."""
        self.notes.append(f"{datetime.now(UTC).isoformat()}: {note}")

    def update_timestamp(self, operation: str) -> None:
        """Update the appropriate timestamp for an operation."""
        timestamp = datetime.now(UTC)
        if operation == "loaded":
            self.loaded_at = timestamp
        elif operation == "split":
            self.split_at = timestamp
