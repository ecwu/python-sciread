"""Integration tests for document construction, rendering, and retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes import FakeEmbeddingClient
from tests.fakes import FakeRerankClient

from sciread.document import Document
from sciread.document.retrieval.evidence import EvidenceRetriever
from sciread.document.retrieval.evidence import format_evidence_results
from sciread.document.retrieval.service import rerank_search

pytestmark = pytest.mark.integration


def test_document_from_text_and_markdown_files_use_real_builder_and_splitters(
    sample_text_file: Path,
    sample_markdown_file: Path,
) -> None:
    """Text and markdown fixtures should load and split through the public Document API."""
    text_document = Document.from_file(sample_text_file, auto_split=True)
    markdown_document = Document.from_file(sample_markdown_file, to_markdown=True, auto_split=True)

    assert text_document.text.startswith("Abstract")
    assert len(text_document.chunks) > 0
    assert text_document.processing_state.loaded_at is not None
    assert any("Document loaded" in note for note in text_document.processing_state.notes)

    assert markdown_document.is_markdown is True
    assert len(markdown_document.chunks) >= 5
    section_names = {name.lower() for name in markdown_document.get_section_names()}
    assert any(name.endswith("abstract") for name in section_names)
    assert any(name.endswith("methods") for name in section_names)
    assert any(name.endswith("results") for name in section_names)
    assert all(chunk.citation_key for chunk in markdown_document.chunks)
    assert all(chunk.retrieval_text for chunk in markdown_document.chunks if chunk.retrievable)


def test_markdown_sections_tree_and_llm_rendering_are_consistent(sample_markdown_file: Path) -> None:
    """Section lookup, tree rendering, and LLM rendering should agree on section labels."""
    document = Document.from_file(sample_markdown_file, to_markdown=True, auto_split=True)

    section_tree = document.build_section_tree()
    rendered_tree = section_tree.render(depth=2)
    methods_section = next(name for name in document.get_section_names() if name.lower().endswith("methods"))
    methods_chunks = document.get_chunks_by_section(methods_section)
    llm_text = document.get_for_llm(section_names=[methods_section], include_headers=True, clean_text=True)

    assert "methods" in rendered_tree
    assert methods_chunks
    assert all("methods" in [part.lower() for part in chunk.section_path] for chunk in methods_chunks)
    assert "retrieval, embeddings, and" in llm_text
    assert "References" not in llm_text


def test_semantic_and_evidence_retrieval_use_fake_embedding_without_external_services(
    sample_markdown_file: Path,
    fake_embedding_client: FakeEmbeddingClient,
) -> None:
    """Semantic retrieval should work with an explicit fake embedding client."""
    document = Document.from_file(sample_markdown_file, to_markdown=True, auto_split=True)
    document.build_vector_index(embedding_client=fake_embedding_client)

    semantic_results = document.semantic_search("retrieval embeddings", top_k=3, return_scores=True)
    empty_results = document.semantic_search("   ", top_k=3)
    evidence_results = EvidenceRetriever(document, strategy="semantic", neighbor_window=1).retrieve(
        "retrieval embeddings",
        top_k=2,
        expand_context=True,
    )
    rendered_evidence = format_evidence_results(evidence_results, query="retrieval embeddings", strategy="semantic")

    assert semantic_results
    assert all(score >= 0 for _chunk, score in semantic_results)
    assert empty_results == []
    assert evidence_results
    assert evidence_results[0].citation_key
    assert "Evidence strategy: semantic" in rendered_evidence
    assert fake_embedding_client.batch_calls


def test_rerank_search_handles_empty_and_duplicate_reranker_outputs(
    sample_markdown_file: Path,
    fake_embedding_client: FakeEmbeddingClient,
) -> None:
    """Rerank integration should fall back on empty output and de-duplicate rerank results."""
    document = Document.from_file(sample_markdown_file, to_markdown=True, auto_split=True)
    document.build_vector_index(embedding_client=fake_embedding_client)

    fallback_client = FakeRerankClient(return_empty=True)
    fallback_results = document.rerank_search(
        "retrieval embeddings",
        top_k=2,
        return_scores=True,
        rerank_client=fallback_client,
    )

    duplicate_client = FakeRerankClient(scores=[0.1, 0.9, 0.2, 0.3], duplicate_first=True)
    deduped_results = rerank_search(
        document,
        "retrieval embeddings",
        top_k=2,
        return_scores=True,
        rerank_client=duplicate_client,
    )

    assert len(fallback_results) <= 2
    assert fallback_client.calls
    assert len(deduped_results) == 2
    assert len({chunk.chunk_id for chunk, _score in deduped_results}) == len(deduped_results)
    assert duplicate_client.calls
