"""Tests for the main Document class."""

import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata


class TestDocument:
    """Test cases for the Document class."""

    def test_document_from_text(self):
        """Test creating document from text."""
        text = "This is a test document."
        metadata = DocumentMetadata(title="Test Document")
        doc = Document.from_text(text, metadata=metadata)

        assert doc.text == text
        assert doc.metadata.title == "Test Document"
        assert doc.source_path is None
        assert doc.processing_state.loaded_at is not None  # Document is automatically loaded
        assert doc.is_split  # Document is automatically split

    def test_document_from_file(self, sample_txt_file):
        """Test creating document from file path."""
        doc = Document.from_file(sample_txt_file)

        assert doc.source_path == sample_txt_file
        assert doc.processing_state.loaded_at is not None  # Document is automatically loaded
        assert doc.is_split  # Document is automatically split

    def test_load_txt_file(self, sample_txt_file):
        """Test loading a text file."""
        doc = Document.from_file(sample_txt_file)

        # Document is automatically loaded and split by from_file()
        assert len(doc.text) > 0
        assert "Abstract" in doc.text
        assert "Introduction" in doc.text

    def test_document_properties_without_source_path(self):
        """Test document properties when no source path is set."""
        doc = Document.from_text("Some text")
        assert doc.source_path is None
        assert doc.text == "Some text"
        assert doc.processing_state.loaded_at is not None

    def test_split_document(self, sample_txt_file):
        """Test that document is automatically split."""
        doc = Document.from_file(sample_txt_file)

        assert doc.is_split
        assert len(doc.chunks) > 1

        # Check chunk properties
        for chunk in doc.chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0
            assert isinstance(chunk.position, int)

    def test_document_auto_splitting(self, sample_txt_file):
        """Test that document is automatically loaded and split by from_file()."""
        doc = Document.from_file(sample_txt_file)
        # Document should be automatically split
        assert doc.is_split
        assert len(doc.chunks) > 0

    def test_split_empty_document(self):
        """Test splitting empty document raises ValueError."""
        with pytest.raises(ValueError, match="Cannot split empty document"):
            Document.from_text("")

    def test_document_with_default_splitter(self, sample_txt_file):
        """Test document created with default splitter."""
        doc = Document.from_file(sample_txt_file)

        # Document should be automatically split with default splitter
        assert len(doc.chunks) > 0
        # Each chunk should have basic properties
        for chunk in doc.chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0

    def test_get_chunks_filtered(self, sample_txt_file):
        """Test getting chunks with filters."""
        doc = Document.from_file(sample_txt_file)

        # Get all chunks
        all_chunks = doc.get_chunks()
        assert len(all_chunks) > 0

        # Get unprocessed chunks
        unprocessed = doc.get_chunks(processed=False)
        assert len(unprocessed) == len(all_chunks)  # All should be unprocessed initially

        # Get processed chunks (should be empty initially)
        processed = doc.get_chunks(processed=True)
        assert len(processed) == 0

        # Get chunks by name
        # May or may not have abstract chunks depending on splitting
        doc.get_chunks(chunk_name="abstract")

    def test_get_unprocessed_chunks(self, sample_txt_file):
        """Test getting unprocessed chunks."""
        doc = Document.from_file(sample_txt_file)

        unprocessed = doc.get_unprocessed_chunks()
        assert len(unprocessed) > 0

        # Mark one chunk as processed
        unprocessed[0].mark_processed()
        remaining_unprocessed = doc.get_unprocessed_chunks()
        assert len(remaining_unprocessed) == len(unprocessed) - 1

    def test_next_unprocessed(self, sample_txt_file):
        """Test getting next unprocessed chunk."""
        doc = Document.from_file(sample_txt_file)

        # Get first unprocessed chunk
        first = doc.next_unprocessed()
        assert first is not None
        assert not first.processed

        # Mark it as processed
        first.mark_processed()

        # Get next unprocessed chunk
        second = doc.next_unprocessed()
        if len(doc.chunks) > 1:
            assert second is not None
            assert second != first
            assert not second.processed

    def test_mark_all_processed(self, sample_txt_file):
        """Test marking all chunks as processed."""
        doc = Document.from_file(sample_txt_file)

        doc.mark_all_processed()

        for chunk in doc.chunks:
            assert chunk.processed

    def test_mark_all_unprocessed(self, sample_txt_file):
        """Test marking all chunks as unprocessed."""
        doc = Document.from_file(sample_txt_file)
        # Document is automatically loaded and split by from_file()

        # Mark all as processed first
        doc.mark_all_processed()
        # Then mark all as unprocessed
        doc.mark_all_unprocessed()

        for chunk in doc.chunks:
            assert not chunk.processed

    def test_iteration(self, sample_txt_file):
        """Test document iteration."""
        doc = Document.from_file(sample_txt_file)
        # Document is automatically loaded and split by from_file()

        # Test iteration
        chunk_count = 0
        for chunk in doc:
            assert isinstance(chunk, Chunk)
            chunk_count += 1
        assert chunk_count == len(doc.chunks)

    def test_length(self, sample_txt_file):
        """Test document length."""
        doc = Document.from_file(sample_txt_file)

        assert len(doc) == len(doc.chunks)

    def test_indexing(self, sample_txt_file):
        """Test document indexing."""
        doc = Document.from_file(sample_txt_file)

        if len(doc.chunks) > 0:
            # Test single index
            chunk = doc[0]
            assert chunk == doc.chunks[0]

            # Test slice
            if len(doc.chunks) > 2:
                chunk_slice = doc[0:2]
                assert len(chunk_slice) == 2
                assert chunk_slice[0] == doc.chunks[0]
                assert chunk_slice[1] == doc.chunks[1]

    def test_processing_state_updates(self, sample_txt_file):
        """Test that processing state is updated correctly."""
        doc = Document.from_file(sample_txt_file)

        # Document should be automatically loaded and split
        assert doc.processing_state.loaded_at is not None
        assert doc.processing_state.split_at is not None

        # Mark all processed should update last_processed_at
        doc.mark_all_processed()
        assert doc.processing_state.last_processed_at is not None

    def test_processing_notes(self, sample_txt_file):
        """Test processing notes are added."""
        doc = Document.from_file(sample_txt_file)

        # Should have notes for loading and splitting
        assert len(doc.processing_state.notes) >= 2
        assert any("loaded" in note.lower() for note in doc.processing_state.notes)
        assert any("split" in note.lower() for note in doc.processing_state.notes)
