"""Tests for RAG functionality in Document class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from sciread.document import Document
from sciread.document.models import Chunk


class TestDocumentRAG:
    """Test cases for Document RAG functionality."""

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content for hashing")
            temp_file = Path(f.name)

        try:
            doc = Document(source_path=temp_file)
            # Hash should be a non-empty string
            assert doc.metadata.file_hash is not None
            assert len(doc.metadata.file_hash) == 64  # SHA-256 hex length
            assert all(c in "0123456789abcdef" for c in doc.metadata.file_hash.lower())
        finally:
            temp_file.unlink()

    def test_calculate_file_hash_error(self):
        """Test file hash calculation with non-existent file."""
        non_existent = Path("/non/existent/file.txt")
        doc = Document(source_path=non_existent)
        # File hash should remain None for non-existent files (only calculated if file exists)
        assert doc.metadata.file_hash is None

    def test_set_chunks(self):
        """Test setting chunks and updating _chunks_by_id."""
        doc = Document(text="test text")
        chunks = [
            Chunk(content="Chunk 1", chunk_name="intro"),
            Chunk(content="Chunk 2", chunk_name="methods"),
        ]

        doc._set_chunks(chunks)

        assert len(doc._chunks) == 2
        assert len(doc._chunks_by_id) == 2
        assert doc._split is True
        assert chunks[0].id in doc._chunks_by_id
        assert chunks[1].id in doc._chunks_by_id
        assert doc._chunks_by_id[chunks[0].id] == chunks[0]
        assert doc._chunks_by_id[chunks[1].id] == chunks[1]

    def test_update_chunks_by_id(self):
        """Test updating chunks_by_id dictionary."""
        doc = Document(text="test text")
        chunk1 = Chunk(content="Chunk 1")
        chunk2 = Chunk(content="Chunk 2")

        doc._chunks = [chunk1, chunk2]
        doc._update_chunks_by_id()

        assert len(doc._chunks_by_id) == 2
        assert chunk1.id in doc._chunks_by_id
        assert chunk2.id in doc._chunks_by_id

    @patch("sciread.document.document.get_config")
    @patch("sciread.document.document.get_embedding_client")
    def test_build_vector_index_no_chunks(self, mock_get_embedding_client, mock_config):
        """Test building vector index with no chunks."""
        doc = Document(text="test text")
        doc.build_vector_index()

        # Should not attempt to build index
        mock_get_embedding_client.assert_not_called()
        mock_config.assert_not_called()

    @patch("sciread.document.document.get_config")
    @patch("sciread.document.document.get_embedding_client")
    @patch("sciread.document.document.VectorIndex")
    def test_build_vector_index_success(self, mock_vector_index, mock_get_embedding_client, mock_config):
        """Test successful vector index building."""
        # Setup mocks
        mock_config.return_value.vector_store = Mock(
            embedding_model="test-model",
            batch_size=5,
            cache_embeddings=True,
            path="~/.test_vector_store",
        )
        mock_embedding_client = Mock()
        mock_get_embedding_client.return_value = mock_embedding_client
        mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_vector_index_instance = Mock()
        mock_vector_index.return_value = mock_vector_index_instance

        # Create document with chunks
        doc = Document(text="test text")
        chunks = [Chunk(content="Chunk 1"), Chunk(content="Chunk 2")]
        doc._set_chunks(chunks)

        doc.build_vector_index(persist=False)

        # Verify method calls
        mock_get_embedding_client.assert_called_once_with("test-model", cache_embeddings=True)
        mock_embedding_client.get_embeddings.assert_called_once()
        mock_vector_index.assert_called_once()
        mock_vector_index_instance.add_chunks.assert_called_once_with(chunks, [[0.1, 0.2], [0.3, 0.4]])

    @patch("sciread.document.document.get_config")
    @patch("sciread.document.document.get_embedding_client")
    @patch("sciread.document.document.VectorIndex")
    def test_build_vector_index_with_persistence(self, mock_vector_index, mock_get_embedding_client, mock_config):
        """Test building vector index with persistence."""
        # Setup mocks
        with tempfile.TemporaryDirectory(prefix="test_vector_store_") as temp_store:
            mock_config.return_value.vector_store = Mock(
                embedding_model="test-model",
                batch_size=5,
                cache_embeddings=True,
                path=temp_store,
            )
            mock_embedding_client = Mock()
            mock_get_embedding_client.return_value = mock_embedding_client
            mock_embedding_client.get_embeddings.return_value = [[0.1, 0.2]]

            mock_vector_index_instance = Mock()
            mock_vector_index.return_value = mock_vector_index_instance

            # Create document with chunks and file hash
            doc = Document(text="test text")
            doc.metadata.file_hash = "test_hash_123"
            chunks = [Chunk(content="Chunk 1")]
            doc._set_chunks(chunks)

            doc.build_vector_index(persist=True)

            # Verify vector index was called with correct path
            mock_vector_index.assert_called_once()
            args, kwargs = mock_vector_index.call_args
            assert kwargs["collection_name"] == "test_hash_123"
            assert "test_hash_123" in str(kwargs["persist_path"])

    @patch("sciread.document.document.get_config")
    def test_build_vector_index_error(self, mock_config):
        """Test building vector index with error."""
        mock_config.side_effect = Exception("Config error")
        doc = Document(text="test text")
        chunks = [Chunk(content="Chunk 1")]
        doc._set_chunks(chunks)

        with pytest.raises(RuntimeError, match="Failed to build vector index"):
            doc.build_vector_index()

    def test_semantic_search_no_vector_index(self):
        """Test semantic search without vector index."""
        doc = Document(text="test text")
        chunks = [Chunk(content="Chunk 1")]
        doc._set_chunks(chunks)

        results = doc.semantic_search("test query")
        assert results == []

    @patch("sciread.document.document.get_config")
    @patch("sciread.document.document.get_embedding_client")
    def test_semantic_search_success(self, mock_get_embedding_client, mock_config):
        """Test successful semantic search."""
        # Setup mocks
        mock_config.return_value.vector_store = Mock(embedding_model="test-model", cache_embeddings=True)
        mock_embedding_client = Mock()
        mock_get_embedding_client.return_value = mock_embedding_client
        mock_embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3]

        # Create document with chunks and vector index
        doc = Document(text="test text")
        chunks = [
            Chunk(content="Chunk 1", chunk_name="intro"),
            Chunk(content="Chunk 2", chunk_name="methods"),
        ]
        doc._set_chunks(chunks)

        # Mock vector index
        mock_vector_index = Mock()
        mock_vector_index.search.return_value = [
            {"id": chunks[1].id, "distance": 0.1, "metadata": {}, "content": "Chunk 2"},
            {"id": "unknown_id", "distance": 0.2, "metadata": {}, "content": "Unknown"},
        ]
        doc.vector_index = mock_vector_index

        results = doc.semantic_search("test query", top_k=5)

        # Verify results
        assert len(results) == 1  # Only the chunk with matching ID
        assert results[0] == chunks[1]  # Should return the actual Chunk object

        # Verify method calls
        mock_get_embedding_client.assert_called_once_with("test-model", cache_embeddings=True)
        mock_embedding_client.get_embedding.assert_called_once_with("test query")
        mock_vector_index.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=5)

    @patch("sciread.document.document.get_config")
    @patch("sciread.document.document.get_embedding_client")
    def test_semantic_search_embedding_error(self, mock_get_embedding_client, mock_config):
        """Test semantic search with embedding error."""
        mock_config.return_value.vector_store = Mock(embedding_model="test-model", cache_embeddings=True)
        mock_embedding_client = Mock()
        mock_get_embedding_client.return_value = mock_embedding_client
        mock_embedding_client.get_embedding.return_value = None

        doc = Document(text="test text")
        chunks = [Chunk(content="Chunk 1")]
        doc._set_chunks(chunks)

        # Mock vector index
        mock_vector_index = Mock()
        doc.vector_index = mock_vector_index

        results = doc.semantic_search("test query")
        assert results == []

    @patch("sciread.document.document.get_config")
    def test_semantic_search_error(self, mock_config):
        """Test semantic search with general error."""
        mock_config.side_effect = Exception("Config error")

        doc = Document(text="test text")
        chunks = [Chunk(content="Chunk 1")]
        doc._set_chunks(chunks)

        # Mock vector index
        mock_vector_index = Mock()
        doc.vector_index = mock_vector_index

        results = doc.semantic_search("test query")
        assert results == []

    def test_save_success(self):
        """Test successful document saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_doc.json"

            doc = Document(text="test content")
            doc._is_markdown = True
            chunks = [Chunk(content="Chunk 1", chunk_name="intro")]
            doc._set_chunks(chunks)

            # Mock vector index
            mock_vector_index = Mock()
            mock_vector_index.persist_path = Path(temp_dir) / "vector_index"
            doc.vector_index = mock_vector_index

            doc.save(output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with output_path.open() as f:
                saved_data = json.load(f)

            assert saved_data["text"] == "test content"
            assert saved_data["is_markdown"] is True
            assert len(saved_data["chunks"]) == 1
            assert saved_data["chunks"][0]["content"] == "Chunk 1"
            assert saved_data["chunks"][0]["chunk_name"] == "intro"
            assert saved_data["vector_index_path"] is not None

    def test_save_no_vector_index(self):
        """Test document saving without vector index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_doc.json"

            doc = Document(text="test content")
            chunks = [Chunk(content="Chunk 1")]
            doc._set_chunks(chunks)

            doc.save(output_path)

            with output_path.open() as f:
                saved_data = json.load(f)

            assert saved_data["vector_index_path"] is None

    def test_load_success(self):
        """Test successful document loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_data = {
                "metadata": {
                    "source_path": str(Path(temp_dir) / "test.txt"),
                    "file_type": "txt",
                    "file_hash": "test_hash_123",
                    "created_at": "2023-01-01T00:00:00Z",
                    "modified_at": "2023-01-01T00:00:00Z",
                    "title": "Test Document",
                    "author": "Test Author",
                    "page_count": 1,
                },
                "text": "test content",
                "chunks": [
                    {
                        "content": "Chunk 1",
                        "chunk_name": "intro",
                        "position": 0,
                        "word_count": 2,
                        "confidence": 1.0,
                        "processed": False,
                        "metadata": {},
                    }
                ],
                "vector_index_path": None,
                "is_markdown": True,
            }

            # Save test data
            state_path = Path(temp_dir) / "test_state.json"
            with state_path.open("w") as f:
                json.dump(test_data, f)

            # Load document
            doc = Document.load(state_path)

            # Verify loaded document
            assert doc.text == "test content"
            assert doc._is_markdown is True
            assert len(doc._chunks) == 1
            assert doc._chunks[0].content == "Chunk 1"
            assert doc._chunks[0].chunk_name == "intro"
            # Chunk ID is auto-generated, so just check that it exists
            assert doc._chunks[0].id is not None
            assert len(doc._chunks_by_id) == 1
            assert doc._chunks[0] in doc._chunks_by_id.values()
            assert doc.metadata.source_path == Path(temp_dir) / "test.txt"
            assert doc.metadata.file_hash == "test_hash_123"

    @patch("sciread.document.document.VectorIndex")
    def test_load_with_vector_index(self, mock_vector_index):
        """Test document loading with vector index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create vector index directory
            vector_path = Path(temp_dir) / "vector_index"
            vector_path.mkdir()

            # Create test data with vector index
            test_data = {
                "metadata": {
                    "source_path": None,
                    "file_type": None,
                    "created_at": "2023-01-01T00:00:00Z",
                    "modified_at": "2023-01-01T00:00:00Z",
                },
                "text": "test content",
                "chunks": [
                    {
                        "content": "Chunk 1",
                        "chunk_name": "intro",
                        "position": 0,
                        "word_count": 2,
                        "confidence": 1.0,
                        "processed": False,
                        "metadata": {},
                    }
                ],
                "vector_index_path": str(vector_path),
                "is_markdown": False,
            }

            # Save test data
            state_path = Path(temp_dir) / "test_state.json"
            with state_path.open("w") as f:
                json.dump(test_data, f)

            # Load document
            doc = Document.load(state_path)

            # Verify vector index was re-linked
            mock_vector_index.assert_called_once_with(collection_name="vector_index", persist_path=vector_path)
            assert doc.vector_index is not None

    def test_load_vector_index_not_exists(self):
        """Test document loading with non-existent vector index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data with non-existent vector index
            test_data = {
                "metadata": {
                    "source_path": None,
                    "file_type": None,
                    "created_at": "2023-01-01T00:00:00Z",
                    "modified_at": "2023-01-01T00:00:00Z",
                },
                "text": "test content",
                "chunks": [],
                "vector_index_path": str(Path(temp_dir) / "non_existent"),
                "is_markdown": False,
            }

            # Save test data
            state_path = Path(temp_dir) / "test_state.json"
            with state_path.open("w") as f:
                json.dump(test_data, f)

            # Load document
            doc = Document.load(state_path)
            assert doc.vector_index is None

    def test_load_error(self):
        """Test document loading with error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("invalid json")
            temp_path = Path(f.name)

        try:
            with pytest.raises(RuntimeError, match="Failed to load document"):
                Document.load(temp_path)
        finally:
            temp_path.unlink()
