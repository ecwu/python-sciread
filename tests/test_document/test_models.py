"""Tests for core data models."""

from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest

from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.document.models import ProcessingState


class TestChunk:
    """Test cases for the Chunk model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="This is a test chunk.",
            chunk_name="test",
            position=0,
            word_count=5,
            confidence=0.9,
        )

        assert chunk.content == "This is a test chunk."
        assert chunk.chunk_name == "test"
        assert chunk.position == 0
        assert chunk.word_count == 5
        assert chunk.confidence == 0.9
        assert chunk.processed is False

    def test_chunk_word_count_auto_calculation(self):
        """Test automatic word count calculation."""
        chunk = Chunk(content="This has five words exactly.")
        assert chunk.word_count == 5

    def test_chunk_confidence_validation(self):
        """Test confidence value validation."""
        # Valid confidence values
        Chunk(content="test", confidence=0.0)
        Chunk(content="test", confidence=0.5)
        Chunk(content="test", confidence=1.0)

        # Invalid confidence values
        with pytest.raises(ValueError, match="Confidence must be between"):
            Chunk(content="test", confidence=-0.1)
        with pytest.raises(ValueError, match="Confidence must be between"):
            Chunk(content="test", confidence=1.1)

    def test_chunk_toggle_processed(self):
        """Test toggling processed status."""
        chunk = Chunk(content="test", processed=False)
        assert chunk.processed is False

        chunk.toggle_processed()
        assert chunk.processed is True

        chunk.toggle_processed()
        assert chunk.processed is False

    def test_chunk_mark_methods(self):
        """Test marking processed/unprocessed."""
        chunk = Chunk(content="test")
        assert chunk.processed is False

        chunk.mark_processed()
        assert chunk.processed is True
        assert chunk.is_processed is True

        chunk.mark_unprocessed()
        assert chunk.processed is False
        assert chunk.is_processed is False

    def test_chunk_new_metadata_defaults(self):
        """Test default initialization for expanded chunk metadata fields."""
        chunk = Chunk(
            content="This is a test chunk.",
            chunk_name="introduction",
            position=3,
            page_range=(1, 2),
        )

        assert chunk.chunk_id
        assert chunk.id == chunk.chunk_id
        assert chunk.content_plain == chunk.content
        assert chunk.retrieval_text == chunk.content_plain
        assert chunk.display_text == chunk.content
        assert chunk.doc_id == ""
        assert chunk.section_path == ["introduction"]
        assert chunk.page_start == 1
        assert chunk.page_end == 2
        assert chunk.para_index == 3
        assert chunk.token_count == chunk.word_count
        assert chunk.prev_chunk_id is None
        assert chunk.next_chunk_id is None
        assert chunk.parent_section_id == "introduction"
        assert chunk.citation_key == chunk.chunk_id
        assert chunk.retrievable is True

    def test_chunk_retrievable_sync_with_processed(self):
        """Test retrievable flag syncs with processed state transitions."""
        chunk = Chunk(content="test")
        assert chunk.retrievable is True

        chunk.mark_processed()
        assert chunk.processed is True
        assert chunk.retrievable is False

        chunk.mark_unprocessed()
        assert chunk.processed is False
        assert chunk.retrievable is True

        chunk.toggle_processed()
        assert chunk.processed is True
        assert chunk.retrievable is False

    def test_chunk_overlap_metadata_defaults_and_validation(self):
        """Test overlap metadata defaults and validation."""
        chunk = Chunk(content="test")

        assert chunk.overlap_prev_chars == 0
        assert chunk.overlap_next_chars == 0
        assert chunk.has_overlap is False

        overlapping_chunk = Chunk(content="test", overlap_prev_chars=5)
        assert overlapping_chunk.has_overlap is True

        with pytest.raises(ValueError, match="Chunk overlap values must be >= 0"):
            Chunk(content="test", overlap_prev_chars=-1)


class TestDocumentMetadata:
    """Test cases for the DocumentMetadata model."""

    def test_metadata_creation(self):
        """Test metadata creation with explicit values."""
        metadata = DocumentMetadata(
            source_path=Path("/test/path.pdf"),
            file_type="pdf",
            file_size=1024,
            title="Test Document",
            author="Test Author",
            page_count=5,
        )

        assert metadata.source_path == Path("/test/path.pdf")
        assert metadata.file_type == "pdf"
        assert metadata.file_size == 1024
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.page_count == 5
        assert metadata.created_at is not None
        assert metadata.modified_at is not None

    def test_metadata_auto_timestamps(self):
        """Test automatic timestamp initialization."""
        before = datetime.now(UTC)
        metadata = DocumentMetadata()
        after = datetime.now(UTC)

        assert before <= metadata.created_at <= after
        assert before <= metadata.modified_at <= after


class TestProcessingState:
    """Test cases for the ProcessingState model."""

    def test_processing_state_creation(self):
        """Test processing state creation."""
        state = ProcessingState(processing_version="1.0")

        assert state.processing_version == "1.0"
        assert state.loaded_at is None
        assert state.split_at is None
        assert state.last_processed_at is None
        assert state.notes == []

    def test_add_note(self):
        """Test adding processing notes."""
        state = ProcessingState()
        state.add_note("Test note")

        assert len(state.notes) == 1
        assert "Test note" in state.notes[0]

    def test_update_timestamp(self):
        """Test timestamp updates."""
        state = ProcessingState()
        state.update_timestamp("loaded")
        state.update_timestamp("split")
        state.update_timestamp("processed")

        assert state.loaded_at is not None
        assert state.split_at is not None
        assert state.last_processed_at is not None

    def test_update_timestamp_invalid_operation(self):
        """Test timestamp update with invalid operation."""
        state = ProcessingState()
        # Should not raise error for unknown operation
        state.update_timestamp("unknown_operation")
        # All timestamps should remain None
        assert state.loaded_at is None
        assert state.split_at is None
        assert state.last_processed_at is None
