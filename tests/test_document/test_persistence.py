"""Tests for document persistence and hydration."""

import json
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest

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
        assert loaded.chunks[0].overlap_prev_chars == 0
        assert loaded.chunks[0].overlap_next_chars == 0
        assert loaded.vector_index is not None
        assert loaded.vector_index.persist_path.resolve() == persist_path.resolve()

    def test_load_preserves_serialized_metadata_and_ignores_missing_vector_index(self, temp_dir: Path):
        """Loading should hydrate metadata types without requiring a stale vector index path."""
        state_path = temp_dir / "document.json"
        source_path = temp_dir / "paper.pdf"
        missing_index_path = temp_dir / "missing-vector-index"
        created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
        modified_at = datetime(2024, 1, 3, 4, 5, 6, tzinfo=UTC)
        state_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "title": "Serialized",
                        "author": "Ada",
                        "source_path": str(source_path),
                        "created_at": created_at.isoformat(),
                        "modified_at": modified_at.isoformat(),
                    },
                    "text": "Serialized body",
                    "chunks": [
                        {
                            "content": "Serialized chunk",
                            "chunk_name": "intro",
                            "section_path": ["intro"],
                            "position": 0,
                        }
                    ],
                    "vector_index_path": str(missing_index_path),
                    "is_markdown": False,
                }
            ),
            encoding="utf-8",
        )

        loaded = Document.load(state_path)

        assert loaded.metadata.title == "Serialized"
        assert loaded.metadata.author == "Ada"
        assert loaded.metadata.source_path == source_path
        assert loaded.metadata.created_at == created_at
        assert loaded.metadata.modified_at == modified_at
        assert loaded.text == "Serialized body"
        assert loaded.chunks[0].doc_id == loaded._build_doc_id()
        assert loaded.vector_index is None

    def test_load_wraps_invalid_state_with_context(self, temp_dir: Path):
        """Invalid persisted state should surface the path and preserve the original exception."""
        state_path = temp_dir / "invalid.json"
        state_path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(RuntimeError, match=f"Failed to load document from {state_path}") as exc_info:
            Document.load(state_path)

        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
