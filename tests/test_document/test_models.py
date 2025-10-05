"""Tests for core data models."""

from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest

from sciread.document.models import Chunk
from sciread.document.models import CoverageStats
from sciread.document.models import DocumentMetadata
from sciread.document.models import ProcessingState


class TestChunk:
    """Test cases for the Chunk model."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            content="This is a test chunk.",
            chunk_type="test",
            position=0,
            word_count=5,
            confidence=0.9,
        )

        assert chunk.content == "This is a test chunk."
        assert chunk.chunk_type == "test"
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


class TestCoverageStats:
    """Test cases for the CoverageStats model."""

    def test_coverage_calculation(self):
        """Test coverage percentage calculation."""
        stats = CoverageStats(
            processed_chunks=5,
            total_chunks=10,
            processed_words=1500,
            total_words=3000,
        )

        assert stats.chunk_coverage == 50.0
        assert stats.word_coverage == 50.0

    def test_coverage_zero_division(self):
        """Test coverage calculation with zero total."""
        stats = CoverageStats(
            processed_chunks=0,
            total_chunks=0,
            processed_words=0,
            total_words=0,
        )

        assert stats.chunk_coverage == 0.0
        assert stats.word_coverage == 0.0

    def test_coverage_to_dict(self):
        """Test dictionary conversion."""
        stats = CoverageStats(
            processed_chunks=3,
            total_chunks=5,
            processed_words=600,
            total_words=1000,
        )

        result = stats.to_dict()
        expected = {
            "processed_chunks": 3,
            "total_chunks": 5,
            "processed_words": 600,
            "total_words": 1000,
            "chunk_coverage": 60.0,
            "word_coverage": 60.0,
        }

        assert result == expected


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
        before = datetime.now(timezone.utc)
        metadata = DocumentMetadata()
        after = datetime.now(timezone.utc)

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
