"""Tests for coordinate-agent helper modules."""

from pathlib import Path

import pytest

from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.planner import extract_abstract
from sciread.agent.coordinate.planner import select_sections_for_expert
from sciread.agent.coordinate.synthesis import build_execution_summary
from sciread.agent.coordinate.synthesis import validate_pdf_document
from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.platform.logging import get_logger


@pytest.fixture
def sample_pdf_document() -> Document:
    """Create a split PDF-like document for coordinate-agent tests."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract\nPaper summary\n\nMethodology\nDetailed methods",
        metadata=DocumentMetadata(title="Test Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks(
        [
            Chunk(content="This is the abstract section.", chunk_name="abstract"),
            Chunk(content="Introduction content.", chunk_name="introduction"),
            Chunk(content="Method details.", chunk_name="methodology"),
            Chunk(content="Experiment results.", chunk_name="results"),
        ]
    )
    return document


def test_extract_abstract_prefers_abstract_section(sample_pdf_document: Document):
    """Abstract extraction should prefer the named abstract section."""
    abstract = extract_abstract(sample_pdf_document, get_logger(__name__))

    assert abstract == "This is the abstract section."


def test_select_sections_for_expert_uses_fuzzy_matching(sample_pdf_document: Document):
    """Planner should map expert preferences to available section names."""
    sections = select_sections_for_expert(sample_pdf_document, "methodology", None)

    assert sections == ["methodology"]


def test_validate_pdf_document_rejects_non_pdf():
    """Coordinate synthesis guard should reject non-PDF inputs."""
    document = Document.from_text("plain text")

    with pytest.raises(ValueError, match="CoordinateAgent only supports PDF files"):
        validate_pdf_document(document)


def test_build_execution_summary_counts_successes_and_failures():
    """Execution summary should ignore the internal sections entry."""
    summary = build_execution_summary(
        {
            "metadata": {"success": True},
            "methodology": {"success": False},
            "_sections_analyzed": {"metadata": ["abstract"]},
        }
    )

    assert summary["total_agents_executed"] == 2
    assert summary["successful_agents"] == 1
    assert summary["failed_agents"] == 1


def test_analysis_plan_model_still_accepts_empty_section_lists():
    """Fallback plans should remain valid after the refactor."""
    plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=True,
        analyze_research_questions=True,
        analyze_methodology=True,
        analyze_experiments=True,
        analyze_future_directions=True,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="fallback",
    )

    assert plan.reasoning == "fallback"
