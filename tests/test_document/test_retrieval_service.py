"""Tests for document retrieval service helpers."""

from unittest.mock import Mock

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.retrieval.service import build_vector_index
from sciread.document.retrieval.service import semantic_search
from sciread.document.state import set_runtime_embedding_client


class TestDocumentRetrievalService:
    """Test retrieval runtime behavior."""

    def test_build_vector_index_caches_runtime_embedding_client(self):
        """Vector index construction should cache the runtime embedding client."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", chunk_name="intro")])

        embedding_client = Mock()
        embedding_client.embedding_batch_size = 4
        embedding_client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]

        vector_index = Mock()
        vector_index_cls = Mock(return_value=vector_index)

        build_vector_index(
            doc,
            embedding_client=embedding_client,
            vector_index_cls=vector_index_cls,
        )

        assert doc._runtime.embedding_client is embedding_client
        embedding_client.get_embeddings.assert_called_once()
        vector_index.add_chunks.assert_called_once()

    def test_semantic_search_reuses_cached_runtime_embedding_client(self):
        """Semantic search should reuse the cached embedding client when available."""
        doc = Document.from_text("placeholder", auto_split=False)
        chunk = Chunk(content="Chunk one", chunk_name="intro")
        doc._set_chunks([chunk])

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.9, 0.1]
        set_runtime_embedding_client(doc, embedding_client)

        doc.vector_index = Mock()
        doc.vector_index.search.return_value = [
            {
                "id": chunk.chunk_id,
                "similarity": 0.91,
            }
        ]

        get_embedding_client_fn = Mock(side_effect=AssertionError("unexpected client lookup"))

        results = semantic_search(
            doc,
            query="intro",
            return_scores=True,
            get_embedding_client_fn=get_embedding_client_fn,
        )

        assert results == [(chunk, 0.91)]
        get_embedding_client_fn.assert_not_called()
