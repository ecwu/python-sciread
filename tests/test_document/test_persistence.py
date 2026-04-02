"""Tests for document persistence and hydration."""

from pathlib import Path

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.document.retrieval.vector_index import VectorIndex


class TestDocumentPersistence:
    """Test document save/load and construction invariants."""

    def test_document_constructor_has_no_source_hash_side_effect(self, sample_txt_file: Path):
        """Direct construction should not derive file metadata from disk."""
        doc = Document(source_path=sample_txt_file, text="plain text")

        assert doc.metadata.source_path == sample_txt_file
        assert doc.metadata.file_hash is None

    def test_document_from_file_populates_file_hash_via_builder(self, sample_txt_file: Path):
        """Builder-backed construction should populate the source hash."""
        doc = Document.from_file(sample_txt_file, auto_split=False)

        assert doc.metadata.source_path == sample_txt_file
        assert doc.metadata.file_hash is not None

    def test_save_and_load_round_trip_preserves_document_state(self, temp_dir: Path):
        """Saving and loading should preserve chunk invariants and runtime index linkage."""
        state_path = temp_dir / "state" / "document.json"
        persist_path = temp_dir / "vector-index"
        persist_path.mkdir()

        doc = Document(
            text="# Intro\n\nPersisted content",
            metadata=DocumentMetadata(title="Persisted Document"),
            _is_markdown=True,
        )
        doc._set_chunks(
            [
                Chunk(content="Chunk one", chunk_name="intro"),
                Chunk(content="Chunk two", chunk_name="results"),
            ]
        )
        doc.vector_index = VectorIndex(collection_name="persisted_document", persist_path=persist_path)

        doc.save(state_path)
        loaded = Document.load(state_path)

        assert loaded.metadata.title == "Persisted Document"
        assert loaded.is_markdown is True
        assert len(loaded.chunks) == 2
        assert loaded.get_chunk_by_id(loaded.chunks[0].chunk_id) == loaded.chunks[0]
        assert loaded.chunks[0].next_chunk_id == loaded.chunks[1].chunk_id
        assert loaded.chunks[1].prev_chunk_id == loaded.chunks[0].chunk_id
        assert loaded.vector_index is not None
        assert loaded.vector_index.persist_path.resolve() == persist_path.resolve()
