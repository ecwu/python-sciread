"""Tests for coordinate-agent helper modules."""

from pathlib import Path

import pytest

from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.models import PreviousMethodsResult
from sciread.agent.coordinate.planner import extract_abstract
from sciread.agent.coordinate.planner import select_sections_for_expert
from sciread.agent.coordinate.prompts import build_analysis_planning_prompt
from sciread.agent.coordinate.synthesis import build_comprehensive_result
from sciread.agent.coordinate.synthesis import build_execution_summary
from sciread.agent.coordinate.synthesis import validate_pdf_document
from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.document.structure.renderers import format_section_choices
from sciread.document.structure.renderers import get_section_length_map
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


@pytest.fixture
def hierarchical_section_document() -> Document:
    """Create a document where a top-level heading has little content."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Method\nHeading only\n\nProposed Method\nActual content",
        metadata=DocumentMetadata(title="Hierarchical Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks(
        [
            Chunk(content="Short bridge.", chunk_name="3. Method"),
            Chunk(content="Detailed proposed method with architecture and training details.", chunk_name="3.1 Proposed Method"),
            Chunk(content="Extensive quantitative evaluation with baselines and ablations.", chunk_name="4. Experiments"),
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


def test_select_sections_for_expert_prefers_longer_non_heading_section(hierarchical_section_document: Document):
    """Planner should avoid choosing a heading-only parent section when a richer child section exists."""
    sections = select_sections_for_expert(hierarchical_section_document, "methodology", None)

    assert "3.1 Proposed Method" in sections
    assert "3. Method" not in sections


def test_build_analysis_planning_prompt_includes_section_lengths_and_short_section_warning(hierarchical_section_document: Document):
    """Planning prompt should expose section lengths so the controller can avoid heading-only sections."""
    section_names = hierarchical_section_document.get_section_names()
    section_lengths = get_section_length_map(hierarchical_section_document, section_names)
    sections_text = format_section_choices(section_names, section_lengths)

    prompt = build_analysis_planning_prompt("Paper abstract", sections_text)

    assert "3. Method |" in prompt
    assert "可能仅标题" in prompt
    assert "只填写章节名本身" in prompt


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


def test_build_comprehensive_result_maps_results_by_agent_name():
    """Comprehensive result fields should map from stable agent names."""
    plan = AnalysisPlan(
        analyze_metadata=False,
        analyze_previous_methods=True,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=["introduction"],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="focused",
    )
    previous_methods = PreviousMethodsResult(related_work=["Prior System"])

    result = build_comprehensive_result(
        analysis_plan=plan,
        sub_agent_results={
            "previous_methods": {"success": True, "result": previous_methods},
            "_sections_analyzed": {"previous_methods": ["introduction"]},
        },
        final_report="report",
        total_execution_time=1.23,
    )

    assert result.previous_methods_result == previous_methods
    assert result.sections_analyzed == {"previous_methods": ["introduction"]}


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
