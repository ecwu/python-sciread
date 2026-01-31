"""Tests for vector index functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from sciread.document.models import Chunk
from sciread.document.vector_index import VectorIndex


class TestVectorIndex:
    """Test cases for VectorIndex class."""

    def test_vector_index_init_in_memory(self):
        """Test VectorIndex initialization with in-memory storage."""
        vector_index = VectorIndex("test_collection")
        assert vector_index._collection.name == "test_collection"
        assert vector_index.persist_path is None

    def test_vector_index_init_persistent(self):
        """Test VectorIndex initialization with persistent storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            persist_path = Path(temp_dir)
            vector_index = VectorIndex("test_collection", persist_path=persist_path)
            assert vector_index._collection.name == "test_collection"
            assert vector_index.persist_path == persist_path

    def test_add_chunks_empty(self):
        """Test adding empty chunks list."""
        vector_index = VectorIndex("test_collection")
        vector_index.add_chunks([], [])  # Should not raise any error

    def test_add_chunks_mismatch(self):
        """Test adding chunks with mismatched embeddings count."""
        vector_index = VectorIndex("test_collection")
        chunks = [Chunk(content="test1"), Chunk(content="test2")]
        embeddings = [[0.1, 0.2]]  # Only one embedding for two chunks

        with pytest.raises(ValueError, match=r"Number of chunks .* must match number of embeddings"):
            vector_index.add_chunks(chunks, embeddings)

    def test_add_chunks_success(self):
        """Test successfully adding chunks with embeddings."""
        vector_index = VectorIndex("test_collection")
        chunks = [Chunk(content="First chunk", chunk_name="introduction"), Chunk(content="Second chunk", chunk_name="methods")]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Mock the collection.add method to avoid actual ChromaDB dependency in tests
        vector_index._collection.add = Mock()
        vector_index.add_chunks(chunks, embeddings)

        # Verify add was called with correct parameters
        vector_index._collection.add.assert_called_once_with(
            embeddings=embeddings,
            documents=["First chunk", "Second chunk"],
            metadatas=[
                {"source": "introduction", "position": 0, "word_count": 2, "confidence": 1.0},
                {"source": "methods", "position": 0, "word_count": 2, "confidence": 1.0},
            ],
            ids=[chunks[0].id, chunks[1].id],
        )

    def test_search_no_results(self):
        """Test search with no results."""
        vector_index = VectorIndex("test_collection")
        # Mock empty query result
        vector_index._collection.query = Mock(return_value={"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]})

        results = vector_index.search([0.1, 0.2, 0.3])
        assert results == []

    def test_search_with_results(self):
        """Test search with results."""
        vector_index = VectorIndex("test_collection")
        # Mock query result
        mock_results = {
            "ids": [["chunk1", "chunk2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"source": "intro"}, {"source": "methods"}]],
            "documents": [["Content 1", "Content 2"]],
        }
        vector_index._collection.query = Mock(return_value=mock_results)

        results = vector_index.search([0.1, 0.2, 0.3])

        assert len(results) == 2
        assert results[0]["id"] == "chunk1"
        assert results[0]["distance"] == 0.1
        assert results[0]["content"] == "Content 1"
        assert results[0]["metadata"]["source"] == "intro"

        assert results[1]["id"] == "chunk2"
        assert results[1]["distance"] == 0.2
        assert results[1]["content"] == "Content 2"
        assert results[1]["metadata"]["source"] == "methods"

    def test_search_query_error(self):
        """Test search with query error."""
        vector_index = VectorIndex("test_collection")
        # Mock query to raise an exception
        vector_index._collection.query = Mock(side_effect=Exception("ChromaDB error"))

        with pytest.raises(RuntimeError, match="Failed to query vector index"):
            vector_index.search([0.1, 0.2, 0.3])

    def test_get_collection_info(self):
        """Test getting collection info."""
        vector_index = VectorIndex("test_collection")
        vector_index._collection.count = Mock(return_value=42)

        info = vector_index.get_collection_info()
        assert info["name"] == "test_collection"
        assert info["count"] == 42
        assert info["persist_path"] is None

    def test_get_collection_info_with_error(self):
        """Test getting collection info with error."""
        vector_index = VectorIndex("test_collection")
        vector_index._collection.count = Mock(side_effect=Exception("Count error"))

        info = vector_index.get_collection_info()
        assert "error" in info

    def test_delete_collection(self):
        """Test deleting collection."""
        vector_index = VectorIndex("test_collection")
        vector_index._client.delete_collection = Mock()

        vector_index.delete_collection()
        vector_index._client.delete_collection.assert_called_once_with(name="test_collection")

    def test_delete_collection_error(self):
        """Test deleting collection with error."""
        vector_index = VectorIndex("test_collection")
        vector_index._client.delete_collection = Mock(side_effect=Exception("Delete error"))

        with pytest.raises(RuntimeError, match="Failed to delete collection"):
            vector_index.delete_collection()
