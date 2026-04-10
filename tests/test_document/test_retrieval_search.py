"""Tests for unified multi-strategy retrieval helpers."""


import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.retrieval.models import RetrievedChunk
from sciread.document.retrieval.search import hybrid_search
from sciread.document.retrieval.search import lexical_search
from sciread.document.retrieval.search import semantic_chunk_search
from sciread.document.retrieval.search import tree_search


def test_lexical_search_matches_text_title_and_section_path() -> None:
    """Lexical search should score content text and section labels."""
    doc = Document.from_text("placeholder", auto_split=False)
    doc._set_chunks(
        [
            Chunk(content="We introduce a novel graph method.", section_path=["introduction"]),
            Chunk(content="Detailed setup and training recipe.", section_path=["methods", "setup"]),
            Chunk(content="Ablation results.", section_path=["results"]),
        ]
    )

    results = lexical_search(
        doc,
        "setup method",
        top_k=2,
        neighbor_window=0,
        section_scope=None,
    )

    assert len(results) == 2
    assert results[0].chunk.content == "Detailed setup and training recipe."
    assert "setup" in results[0].matched_terms
    assert "[unnamed_document:1]" in results[0].expanded_context


def test_tree_search_matches_parent_and_child_nodes() -> None:
    """Tree search should return chunks from matching descendant paths."""
    doc = Document.from_text("placeholder", auto_split=False)
    doc._set_chunks(
        [
            Chunk(content="Intro", section_path=["1 introduction"]),
            Chunk(content="Setup", section_path=["1 introduction", "1.1 setup"]),
            Chunk(content="Result", section_path=["2 results"]),
        ]
    )

    results = tree_search(
        doc,
        "setup",
        top_k=2,
        neighbor_window=0,
        section_scope=None,
    )

    assert len(results) == 1
    assert results[0].chunk.content == "Setup"
    assert results[0].section_path == ["1 introduction", "1.1 setup"]


def test_semantic_chunk_search_builds_index_lazily_and_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic search should lazily build the vector index before querying."""
    doc = Document.from_text("placeholder", auto_split=False)
    chunk = Chunk(content="Chunk one", section_path=["intro"])
    doc._set_chunks([chunk])

    build_calls = {"count": 0}

    def fake_build_vector_index() -> None:
        build_calls["count"] += 1
        doc.vector_index = object()

    monkeypatch.setattr(doc, "build_vector_index", fake_build_vector_index)
    monkeypatch.setattr(doc, "semantic_search", lambda query, top_k, return_scores: [(chunk, 0.91)])

    results = semantic_chunk_search(
        doc,
        "intro",
        top_k=1,
        neighbor_window=0,
        section_scope=None,
    )

    assert build_calls["count"] == 1
    assert results[0].score == 0.91


def test_semantic_chunk_search_wraps_index_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic failures should surface a clear retrieval-specific error."""
    doc = Document.from_text("placeholder", auto_split=False)

    def fake_build_vector_index() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(doc, "build_vector_index", fake_build_vector_index)

    with pytest.raises(RuntimeError, match="Semantic retrieval unavailable"):
        semantic_chunk_search(doc, "intro", top_k=1, neighbor_window=0, section_scope=None)


def test_hybrid_search_deduplicates_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid search should fuse and deduplicate hits from different strategies."""
    doc = Document.from_text("placeholder", auto_split=False)
    chunk = Chunk(content="Chunk one", section_path=["intro"])
    doc._set_chunks([chunk])
    lexical_hit = RetrievedChunk(chunk=chunk, score=5.0, strategy="lexical", section_path=["intro"], expanded_context="A")
    semantic_hit = RetrievedChunk(chunk=chunk, score=0.9, strategy="semantic", section_path=["intro"], expanded_context="B")

    monkeypatch.setattr("sciread.document.retrieval.search.lexical_search", lambda *args, **kwargs: [lexical_hit])
    monkeypatch.setattr("sciread.document.retrieval.search.semantic_chunk_search", lambda *args, **kwargs: [semantic_hit])
    monkeypatch.setattr("sciread.document.retrieval.search.tree_search", lambda *args, **kwargs: [lexical_hit])

    results = hybrid_search(
        doc,
        "intro",
        top_k=3,
        neighbor_window=0,
        section_scope=None,
    )

    assert len(results) == 1
    assert results[0].strategy == "hybrid"
