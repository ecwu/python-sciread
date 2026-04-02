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
    token_count: int | None = None

    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    parent_section_id: str | None = None

    citation_key: str = ""
    retrievable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy compatibility fields (kept to avoid breaking current APIs/tests)
    chunk_name: str = "unknown"  # abstract, introduction, methods, etc.
    position: int = 0  # Sequential position in document
    page_range: tuple[int, int] | None = None  # (start_page, end_page)
    word_count: int = 0
    confidence: float = 1.0  # Confidence in classification (0.0-1.0)
    processed: bool = False  # Processing status

    def __post_init__(self):
        """Validate and initialize derived fields."""
        if not self.display_text:
            self.display_text = self.content

        if not self.content_plain:
            self.content_plain = self.content

        if not self.retrieval_text:
            self.retrieval_text = self.content_plain

        if self.word_count == 0:
            self.word_count = len(self.content.split())

        if self.token_count is None:
            # Approximate token count fallback when no tokenizer is provided.
            self.token_count = len(self.content_plain.split())

        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        self.sync_page_range()

        if self.para_index is None:
            self.para_index = self.position

        self.sync_section_metadata()

        if not self.citation_key:
            self.citation_key = self.chunk_id

        if self.processed:
            self.retrievable = False

    @property
    def id(self) -> str:
        """Backward-compatible chunk identifier alias."""
        return self.chunk_id

    @id.setter
    def id(self, value: str) -> None:
        """Backward-compatible chunk identifier alias."""
        self.chunk_id = value

    def sync_page_range(self) -> None:
        """Keep page range fields synchronized."""
        if self.page_range is not None:
            if self.page_start is None:
                self.page_start = self.page_range[0]
            if self.page_end is None:
                self.page_end = self.page_range[1]
            return

        if self.page_start is not None and self.page_end is not None:
            self.page_range = (self.page_start, self.page_end)

    def sync_section_metadata(self) -> None:
        """Keep section fields synchronized across legacy and normalized metadata."""
        if not self.section_path and self.chunk_name and self.chunk_name != "unknown":
            self.section_path = [self.chunk_name]
        elif self.chunk_name == "unknown" and self.section_path:
            self.chunk_name = self.section_path[-1]

        if not self.parent_section_id and self.section_path:
            self.parent_section_id = self.section_path[-1]

    def _set_processed_state(self, processed: bool) -> None:
        """Update processed state and the derived retrievable flag together."""
        self.processed = processed
        self.retrievable = not processed

    def toggle_processed(self) -> None:
        """Toggle the processed status of this chunk."""
        self._set_processed_state(not self.processed)

    def mark_processed(self) -> None:
        """Mark this chunk as processed."""
        self._set_processed_state(True)

    def mark_unprocessed(self) -> None:
        """Mark this chunk as unprocessed."""
        self._set_processed_state(False)

    @property
    def is_processed(self) -> bool:
        """Check if this chunk is processed."""
        return self.processed


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
    last_processed_at: datetime | None = None
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
        elif operation == "processed":
            self.last_processed_at = timestamp
