"""Tests for document retrieval service helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.retrieval.service import build_vector_index
from sciread.document.retrieval.service import cosine_similarity
from sciread.document.retrieval.service import rerank_search
from sciread.document.retrieval.service import semantic_search
from sciread.document.state import set_runtime_embedding_client


def _provider_config(
    tmp_path=None,
    *,
    embedding_model: str = "dummy/model",
    cache_embeddings: bool = True,
    embedding_batch_size: int = 10,
    rerank_model: str = "dummy/reranker",
    rerank_candidate_multiplier: int = 4,
):
    """Build the provider-focused config shape used by retrieval helpers."""
    vector_store_path = str(tmp_path / "vector_store") if tmp_path is not None else "~/.sciread/vector_store"
    return SimpleNamespace(
        providers=SimpleNamespace(
            embedding=SimpleNamespace(
                default=SimpleNamespace(
                    model=embedding_model,
                    batch_size=embedding_batch_size,
                    cache_embeddings=cache_embeddings,
                )
            ),
            rerank=SimpleNamespace(
                default=SimpleNamespace(
                    model=rerank_model,
                    candidate_multiplier=rerank_candidate_multiplier,
                )
            ),
        ),
        vector_store=SimpleNamespace(path=vector_store_path),
    )


class TestDocumentRetrievalService:
    """Test retrieval runtime behavior."""

    def test_cosine_similarity_handles_normal_and_invalid_vectors(self):
        """Cosine similarity should degrade safely on edge cases."""
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_build_vector_index_returns_early_when_document_has_no_chunks(self):
        """Vector index construction should no-op when no chunks exist."""
        doc = Document.from_text("placeholder", auto_split=False)
        get_embedding_client_fn = Mock()

        build_vector_index(doc, get_embedding_client_fn=get_embedding_client_fn)

        assert doc.vector_index is None
        get_embedding_client_fn.assert_not_called()

    def test_build_vector_index_caches_runtime_embedding_client(self):
        """Vector index construction should cache the runtime embedding client."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", section_path=["intro"])])

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
        vector_index_cls.assert_called_once_with(
            collection_name="unnamed_document",
            persist_path=None,
            reset_collection=True,
        )

    def test_build_vector_index_can_create_persisted_index_from_config(self, tmp_path):
        """Persisted indexes should derive their client and storage path from config."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", section_path=["intro"])])

        embedding_client = Mock()
        embedding_client.get_embeddings.return_value = [[0.1, 0.2]]
        vector_index = Mock()
        vector_index_cls = Mock(return_value=vector_index)
        get_embedding_client_fn = Mock(return_value=embedding_client)
        config = _provider_config(tmp_path)

        build_vector_index(
            doc,
            persist=True,
            get_config_fn=Mock(return_value=config),
            get_embedding_client_fn=get_embedding_client_fn,
            vector_index_cls=vector_index_cls,
        )

        get_embedding_client_fn.assert_called_once_with("dummy/model", cache_embeddings=True)
        vector_index_cls.assert_called_once_with(
            collection_name="unnamed_document",
            persist_path=tmp_path / "vector_store" / "unnamed_document",
            reset_collection=True,
        )
        vector_index.add_chunks.assert_called_once()

    def test_build_vector_index_skips_non_retrievable_chunks(self):
        """Only retrievable chunks should be embedded and indexed."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks(
            [
                Chunk(content="keep", section_path=["intro"], retrievable=True),
                Chunk(content="skip", section_path=["appendix"], retrievable=False),
            ]
        )

        embedding_client = Mock()
        embedding_client.get_embeddings.return_value = [[0.1, 0.2]]
        vector_index = Mock()
        vector_index_cls = Mock(return_value=vector_index)

        build_vector_index(
            doc,
            embedding_client=embedding_client,
            vector_index_cls=vector_index_cls,
        )

        embedding_client.get_embeddings.assert_called_once()
        embedded_texts = embedding_client.get_embeddings.call_args.args[0]
        assert len(embedded_texts) == 1
        assert "keep" in embedded_texts[0]
        assert embedding_client.get_embeddings.call_args.kwargs == {"batch_size": 10}
        indexed_chunks = vector_index.add_chunks.call_args.args[0]
        assert [chunk.content for chunk in indexed_chunks] == ["keep"]

    def test_build_vector_index_wraps_failures(self):
        """Unexpected indexing failures should be wrapped in a RuntimeError."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", section_path=["intro"])])

        embedding_client = Mock()
        embedding_client.get_embeddings.side_effect = ValueError("embedding failed")

        with pytest.raises(RuntimeError, match="Failed to build vector index: embedding failed"):
            build_vector_index(doc, embedding_client=embedding_client)

    def test_semantic_search_reuses_cached_runtime_embedding_client(self):
        """Semantic search should reuse the cached embedding client when available."""
        doc = Document.from_text("placeholder", auto_split=False)
        chunk = Chunk(content="Chunk one", section_path=["intro"])
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

    def test_semantic_search_returns_empty_when_vector_index_is_missing(self):
        """Semantic search should no-op until an index exists."""
        doc = Document.from_text("placeholder", auto_split=False)

        results = semantic_search(doc, query="intro")

        assert results == []

    def test_semantic_search_builds_runtime_client_and_returns_chunks(self):
        """Semantic search should create and cache an embedding client when needed."""
        doc = Document.from_text("placeholder", auto_split=False)
        chunk = Chunk(content="Chunk one", section_path=["intro"])
        doc._set_chunks([chunk])
        doc.vector_index = Mock()
        doc.vector_index.search.return_value = [
            {"id": chunk.chunk_id, "similarity": 0.88},
            {"id": "missing", "similarity": 0.77},
        ]

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.4, 0.6]
        config = _provider_config(cache_embeddings=False)

        results = semantic_search(
            doc,
            query="intro",
            get_config_fn=Mock(return_value=config),
            get_embedding_client_fn=Mock(return_value=embedding_client),
        )

        assert results == [chunk]
        assert doc._runtime.embedding_client is embedding_client

    def test_semantic_search_returns_empty_when_query_embedding_is_missing(self):
        """Missing query embeddings should return an empty result set."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", section_path=["intro"])])
        doc.vector_index = Mock()

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = []
        set_runtime_embedding_client(doc, embedding_client)

        results = semantic_search(doc, query="intro")

        assert results == []
        doc.vector_index.search.assert_not_called()

    def test_semantic_search_filters_non_retrievable_chunks(self):
        """Semantic search should not surface chunks marked as non-retrievable."""
        doc = Document.from_text("placeholder", auto_split=False)
        retrievable_chunk = Chunk(content="Chunk one", section_path=["intro"])
        hidden_chunk = Chunk(content="Chunk two", section_path=["appendix"], retrievable=False)
        doc._set_chunks([retrievable_chunk, hidden_chunk])
        doc.vector_index = Mock()
        doc.vector_index.search.return_value = [
            {"id": hidden_chunk.chunk_id, "similarity": 0.99},
            {"id": retrievable_chunk.chunk_id, "similarity": 0.88},
        ]

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.4, 0.6]
        set_runtime_embedding_client(doc, embedding_client)

        results = semantic_search(doc, query="intro")

        assert results == [retrievable_chunk]

    def test_semantic_search_returns_empty_when_search_raises(self):
        """Unexpected search failures should be swallowed and logged as empty results."""
        doc = Document.from_text("placeholder", auto_split=False)
        doc._set_chunks([Chunk(content="Chunk one", section_path=["intro"])])
        doc.vector_index = Mock()
        doc.vector_index.search.side_effect = RuntimeError("search failed")

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.1, 0.2]
        set_runtime_embedding_client(doc, embedding_client)

        results = semantic_search(doc, query="intro")

        assert results == []

    def test_rerank_search_reranks_semantic_candidates_and_caches_client(self):
        """Rerank search should reorder semantic candidates using the rerank provider."""
        doc = Document.from_text("placeholder", auto_split=False)
        first = Chunk(content="Apple company", section_path=["intro"])
        second = Chunk(content="Apple fruit", section_path=["methods"])
        doc._set_chunks([first, second])
        doc.vector_index = Mock()
        doc.vector_index.search.return_value = [
            {"id": first.chunk_id, "similarity": 0.8},
            {"id": second.chunk_id, "similarity": 0.7},
        ]

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.1, 0.2]
        rerank_client = Mock()
        rerank_client.rerank.return_value = [
            SimpleNamespace(index=1, relevance_score=0.93),
            SimpleNamespace(index=0, relevance_score=0.2),
        ]
        config = _provider_config(rerank_candidate_multiplier=3)

        results = rerank_search(
            doc,
            query="apple as food",
            top_k=1,
            return_scores=True,
            get_config_fn=Mock(return_value=config),
            get_embedding_client_fn=Mock(return_value=embedding_client),
            get_rerank_client_fn=Mock(return_value=rerank_client),
        )

        assert results == [(second, 0.93)]
        assert doc._runtime.rerank_client is rerank_client
        rerank_client.rerank.assert_called_once()
        assert rerank_client.rerank.call_args.args[0] == "apple as food"
        assert rerank_client.rerank.call_args.kwargs == {"top_n": 1}

    def test_rerank_search_falls_back_to_semantic_order_when_rerank_returns_empty(self):
        """Rerank search should keep semantic results when the rerank provider cannot score."""
        doc = Document.from_text("placeholder", auto_split=False)
        chunk = Chunk(content="Chunk one", section_path=["intro"])
        doc._set_chunks([chunk])
        doc.vector_index = Mock()
        doc.vector_index.search.return_value = [{"id": chunk.chunk_id, "similarity": 0.88}]

        embedding_client = Mock()
        embedding_client.get_embedding.return_value = [0.1, 0.2]
        rerank_client = Mock()
        rerank_client.rerank.return_value = []
        config = _provider_config(rerank_candidate_multiplier=4)

        results = rerank_search(
            doc,
            query="intro",
            top_k=1,
            get_config_fn=Mock(return_value=config),
            get_embedding_client_fn=Mock(return_value=embedding_client),
            get_rerank_client_fn=Mock(return_value=rerank_client),
        )

        assert results == [chunk]
