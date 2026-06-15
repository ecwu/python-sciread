"""Tests for evidence-level retrieval."""

from __future__ import annotations

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.retrieval.evidence import EvidenceRetriever
from sciread.document.retrieval.evidence import format_evidence_results
from sciread.document.retrieval.models import Evidence
from sciread.document.retrieval.models import RetrievedChunk


def test_evidence_retriever_returns_expanded_agent_facing_evidence(monkeypatch) -> None:
    """Evidence retrieval should hide chunk internals and expand same-section neighbors."""
    document = Document.from_text("placeholder", auto_split=False)
    chunks = [
        Chunk(content="Previous context.", section_path=["methods"], page_range=(1, 1)),
        Chunk(content="The retrieved method uses contrastive training.", section_path=["methods"], page_range=(2, 2)),
        Chunk(content="Next context.", section_path=["methods"], page_range=(3, 3)),
        Chunk(content="Different section.", section_path=["results"], page_range=(4, 4)),
    ]
    document._set_chunks(chunks)

    calls = []

    def fake_retrieve_chunks(**kwargs):
        calls.append(kwargs)
        return [RetrievedChunk(chunk=chunks[1], score=0.87, strategy="semantic", section_path=["methods"])]

    monkeypatch.setattr(document, "retrieve_chunks", fake_retrieve_chunks)

    evidence = EvidenceRetriever(document, max_context_tokens=100).retrieve("contrastive method", top_k=1)

    assert calls == [
        {
            "query": "contrastive method",
            "strategy": "semantic",
            "top_k": 1,
            "neighbor_window": 0,
            "section_scope": None,
        }
    ]
    assert len(evidence) == 1
    assert evidence[0].evidence_id == "E1"
    assert evidence[0].chunk_id == chunks[1].chunk_id
    assert evidence[0].citation_key == chunks[1].citation_key
    assert evidence[0].section_path == ["methods"]
    assert evidence[0].section_label == "methods"
    assert "Previous context." in evidence[0].text
    assert "The retrieved method uses contrastive training." in evidence[0].text
    assert "Next context." in evidence[0].text
    assert "Different section." not in evidence[0].text
    assert evidence[0].page_range == (1, 3)
    assert evidence[0].expanded_from == [chunks[0].chunk_id, chunks[1].chunk_id, chunks[2].chunk_id]
    assert evidence[0].score == 0.87
    assert evidence[0].rank == 1


def test_evidence_retriever_respects_section_filter(monkeypatch) -> None:
    """Section filters should keep only matching evidence candidates."""
    document = Document.from_text("placeholder", auto_split=False)
    chunks = [
        Chunk(content="Method text.", section_path=["methods"]),
        Chunk(content="Result text.", section_path=["results"]),
    ]
    document._set_chunks(chunks)
    monkeypatch.setattr(
        document,
        "retrieve_chunks",
        lambda **_kwargs: [
            RetrievedChunk(chunk=chunks[0], score=0.7, strategy="semantic", section_path=["methods"]),
            RetrievedChunk(chunk=chunks[1], score=0.9, strategy="semantic", section_path=["results"]),
        ],
    )

    evidence = EvidenceRetriever(document).retrieve("effect size", section_filter=["results"])

    assert [item.chunk_id for item in evidence] == [chunks[1].chunk_id]


def test_evidence_retriever_keeps_hit_chunk_when_context_budget_is_small(monkeypatch) -> None:
    """Token budgeting should prioritize the matched chunk over its neighbors."""
    document = Document.from_text("placeholder", auto_split=False)
    chunks = [
        Chunk(content="large previous context that should not replace the hit", section_path=["intro"]),
        Chunk(content="hit chunk has essential answer tokens", section_path=["intro"]),
        Chunk(content="large next context that should not replace the hit", section_path=["intro"]),
    ]
    document._set_chunks(chunks)
    monkeypatch.setattr(
        document,
        "retrieve_chunks",
        lambda **_kwargs: [RetrievedChunk(chunk=chunks[1], score=1.0, strategy="semantic", section_path=["intro"])],
    )

    evidence = EvidenceRetriever(document, max_context_tokens=4).retrieve("answer", top_k=1)

    assert evidence[0].text == "hit chunk has essential"
    assert evidence[0].expanded_from is None
    assert evidence[0].chunk_id == chunks[1].chunk_id


def test_format_evidence_results_uses_evidence_fields() -> None:
    """Evidence formatting should not require raw chunk objects."""
    formatted = format_evidence_results(
        [
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                citation_key="doc:1",
                section_path=["intro"],
                section_label="intro",
                text="Evidence text.",
                display_text="Evidence text.",
                score=0.5,
                rank=1,
            )
        ],
        query="claim",
        strategy="semantic",
    )

    assert "Evidence strategy: semantic" in formatted
    assert "id=E1" in formatted
    assert "citation=doc:1" in formatted
    assert "Evidence text." in formatted
