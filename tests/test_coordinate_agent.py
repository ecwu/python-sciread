"""Tests for coordinate-agent helper modules."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from sciread.agent.coordinate.agent import CoordinateAgent
from sciread.agent.coordinate.executor import execute_sub_agents
from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.models import MetadataExtractionResult
from sciread.agent.coordinate.models import MethodologyResult
from sciread.agent.coordinate.models import PreviousMethodsResult
from sciread.agent.coordinate.planner import build_expert_content
from sciread.agent.coordinate.planner import create_planning_agent
from sciread.agent.coordinate.planner import extract_abstract
from sciread.agent.coordinate.planner import plan_analysis
from sciread.agent.coordinate.planner import select_sections_for_expert
from sciread.agent.coordinate.prompts import build_analysis_planning_prompt
from sciread.agent.coordinate.runtime import CoordinateDeps
from sciread.agent.coordinate.synthesis import build_comprehensive_result
from sciread.agent.coordinate.synthesis import build_execution_summary
from sciread.agent.coordinate.synthesis import synthesize_report
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


def test_extract_abstract_falls_back_to_first_chunk_and_text() -> None:
    """Abstract extraction should degrade through chunk and raw-text fallbacks."""
    single_chunk_document = Document(
        source_path=Path("paper.pdf"),
        text="Fallback text",
        metadata=DocumentMetadata(title="Chunk fallback", source_path=Path("paper.pdf")),
    )
    single_chunk_document._set_chunks([Chunk(content="Only chunk content.", chunk_name="body")])

    text_only_document = Document(
        source_path=Path("paper.pdf"),
        text="Full text fallback content",
        metadata=DocumentMetadata(title="Text fallback", source_path=Path("paper.pdf")),
    )

    empty_document = Document(
        source_path=Path("paper.pdf"),
        text="",
        metadata=DocumentMetadata(title="Empty", source_path=Path("paper.pdf")),
    )

    logger = get_logger(__name__)

    assert extract_abstract(single_chunk_document, logger) == "Only chunk content."
    assert extract_abstract(text_only_document, logger) == "Full text fallback content"
    assert extract_abstract(empty_document, logger) == ""


def test_select_sections_for_expert_uses_fuzzy_matching(sample_pdf_document: Document):
    """Planner should map expert preferences to available section names."""
    sections = select_sections_for_expert(sample_pdf_document, "methodology", None)

    assert sections == ["methodology"]


def test_select_sections_for_expert_prefers_longer_non_heading_section(hierarchical_section_document: Document):
    """Planner should avoid choosing a heading-only parent section when a richer child section exists."""
    sections = select_sections_for_expert(hierarchical_section_document, "methodology", None)

    assert "3.1 Proposed Method" in sections
    assert "3. Method" not in sections


def test_select_sections_for_expert_falls_back_to_non_heading_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should return substantive sections when fuzzy matching yields nothing."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Body",
        metadata=DocumentMetadata(title="Fallback Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks(
        [
            Chunk(content="Tiny", chunk_name="1. Heading"),
            Chunk(content="Substantial introduction content for the paper.", chunk_name="2. Introduction"),
            Chunk(content="Detailed experiment content with extensive findings.", chunk_name="3. Results"),
        ]
    )

    monkeypatch.setattr(Document, "get_closest_section_name", lambda self, target, threshold=0.7: None)

    sections = select_sections_for_expert(document, "unknown_analysis", None)

    assert sections == ["2. Introduction", "3. Results"]


def test_select_sections_for_expert_falls_back_to_available_sections_when_all_are_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should still return available sections when every section looks heading-only."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Body",
        metadata=DocumentMetadata(title="Short Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks(
        [
            Chunk(content="A", chunk_name="1. Intro"),
            Chunk(content="B", chunk_name="2. Method"),
            Chunk(content="C", chunk_name="3. Result"),
        ]
    )

    monkeypatch.setattr(Document, "get_closest_section_name", lambda self, target, threshold=0.7: None)

    sections = select_sections_for_expert(document, "unknown_analysis", None)

    assert sections == ["1. Intro", "2. Method", "3. Result"]


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


def test_build_expert_content_uses_selected_sections_and_standard_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expert content assembly should pass normalized section choices into document rendering."""
    captured: dict[str, object] = {}

    class DummyDocument:
        def get_for_llm(self, **kwargs: object) -> str:
            captured.update(kwargs)
            return "expert content"

    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.select_sections_for_expert",
        lambda document, analysis_type, planned_sections: ["Methods", "Results"],
    )

    content = build_expert_content(DummyDocument(), "methodology", ["Methods"], max_tokens=2048)

    assert content == "expert content"
    assert captured == {
        "section_names": ["Methods", "Results"],
        "max_tokens": 2048,
        "include_headers": True,
        "clean_text": True,
        "max_chars_per_section": 2500,
    }


def test_coordinate_agent_init_with_model_name_and_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """String model names should resolve once and be reflected in repr."""
    captured: dict[str, object] = {}
    fake_model = object()

    def fake_create_planning_agent(model: object, max_retries: int, logger) -> str:
        captured["planning_args"] = (model, max_retries)
        return "planner"

    monkeypatch.setattr("sciread.agent.coordinate.agent.get_model", lambda model_name: fake_model)
    monkeypatch.setattr("sciread.agent.coordinate.agent.create_planning_agent", fake_create_planning_agent)

    agent = CoordinateAgent("mock-model", max_retries=4, timeout=12.5)

    assert agent.model is fake_model
    assert agent.model_identifier == "mock-model"
    assert agent.planning_agent == "planner"
    assert captured["planning_args"] == (fake_model, 4)
    assert repr(agent) == "CoordinateAgent(model=mock-model)"


def test_coordinate_agent_init_with_model_object_uses_model_name_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct model objects should avoid get_model and keep their declared name."""
    model = SimpleNamespace(model_name="inline-model")

    monkeypatch.setattr(
        "sciread.agent.coordinate.agent.create_planning_agent",
        lambda model, max_retries, logger: "planner",
    )

    agent = CoordinateAgent(model, max_retries=2)

    assert agent.model is model
    assert agent.model_identifier == "inline-model"
    assert agent.planning_agent == "planner"


def test_coordinate_agent_extract_abstract_delegates_to_planner(monkeypatch: pytest.MonkeyPatch, sample_pdf_document: Document) -> None:
    """The thin extract wrapper should forward the document and logger."""
    agent = CoordinateAgent.__new__(CoordinateAgent)
    agent.logger = get_logger(__name__)

    monkeypatch.setattr("sciread.agent.coordinate.agent.extract_abstract", lambda document, logger: "wrapped abstract")

    assert agent.extract_abstract(sample_pdf_document) == "wrapped abstract"


@pytest.mark.asyncio
async def test_coordinate_agent_wrapper_methods_delegate_to_module_helpers(
    sample_pdf_document: Document, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrapper methods should pass through stateful configuration to helper modules."""
    agent = CoordinateAgent.__new__(CoordinateAgent)
    agent.logger = get_logger(__name__)
    agent.timeout = 9.0
    agent.max_retries = 3
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.planning_agent = object()
    plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="wrapped",
    )

    async def fake_plan_analysis(**kwargs):
        assert kwargs["document"] is sample_pdf_document
        assert kwargs["section_names"] == ["abstract", "introduction", "methodology", "results"]
        assert kwargs["planning_agent"] is agent.planning_agent
        assert kwargs["timeout"] == 9.0
        assert kwargs["model_identifier"] == "mock-model"
        return plan

    async def fake_execute_sub_agents(**kwargs):
        assert kwargs["document"] is sample_pdf_document
        assert kwargs["analysis_plan"] is plan
        assert kwargs["model"] is agent.model
        assert kwargs["max_retries"] == 3
        return {"metadata": {"success": True}}

    async def fake_synthesize_report(**kwargs):
        assert kwargs["analysis_plan"] is plan
        assert kwargs["sub_agent_results"] == {"metadata": {"success": True}}
        assert kwargs["document"] is sample_pdf_document
        assert kwargs["model"] is agent.model
        assert kwargs["timeout"] == 9.0
        return "wrapped report"

    monkeypatch.setattr("sciread.agent.coordinate.agent.plan_analysis", fake_plan_analysis)
    monkeypatch.setattr("sciread.agent.coordinate.agent.execute_sub_agents", fake_execute_sub_agents)
    monkeypatch.setattr("sciread.agent.coordinate.agent.synthesize_report", fake_synthesize_report)

    planned = await agent.plan_analysis(sample_pdf_document, sample_pdf_document.get_section_names())
    executed = await agent.execute_sub_agents(sample_pdf_document, plan)
    report = await agent.synthesize_report(plan, executed, sample_pdf_document)

    assert planned is plan
    assert executed == {"metadata": {"success": True}}
    assert report == "wrapped report"


@pytest.mark.asyncio
async def test_plan_analysis_returns_default_plan_when_abstract_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should short-circuit to the default plan when no abstract can be extracted."""
    document = object()
    planning_agent = SimpleNamespace()
    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "")

    plan = await plan_analysis(
        document=document,
        section_names=["Intro"],
        planning_agent=planning_agent,
        timeout=10.0,
        model_identifier="mock-model",
        logger=get_logger(__name__),
    )

    assert plan.analyze_metadata is True
    assert plan.analyze_methodology is True
    assert "No abstract available" in plan.reasoning


@pytest.mark.asyncio
async def test_plan_analysis_returns_agent_output_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should return the model-generated plan when the call succeeds."""
    expected_plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=True,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=["Methods"],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="model plan",
    )

    class FakePlanningAgent:
        async def run(self, prompt: str, deps) -> _FakeAgentRunResult:
            assert prompt == "请生成分析计划"
            return _FakeAgentRunResult(expected_plan)

    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "Paper abstract")
    monkeypatch.setattr("sciread.agent.coordinate.planner.get_section_length_map", lambda document, section_names: {"Methods": 240})
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.format_section_choices", lambda section_names, section_lengths: "Methods | 240 chars"
    )

    document = object()
    logger = get_logger(__name__)
    plan = await plan_analysis(
        document=document,
        section_names=["Methods"],
        planning_agent=FakePlanningAgent(),
        timeout=10.0,
        model_identifier="mock-model",
        logger=logger,
    )

    assert plan == expected_plan


@pytest.mark.asyncio
async def test_plan_analysis_falls_back_to_default_plan_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should recover with the default plan when the model call times out."""

    class SlowPlanningAgent:
        async def run(self, prompt: str, deps):
            raise TimeoutError("timed out")

    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "Paper abstract")
    monkeypatch.setattr("sciread.agent.coordinate.planner.get_section_length_map", lambda document, section_names: {"Methods": 240})
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.format_section_choices", lambda section_names, section_lengths: "Methods | 240 chars"
    )

    document = object()
    plan = await plan_analysis(
        document=document,
        section_names=["Methods"],
        planning_agent=SlowPlanningAgent(),
        timeout=1.0,
        model_identifier="mock-model",
        logger=get_logger(__name__),
    )

    assert plan.analyze_future_directions is True
    assert "Planning timed out after 1.0 seconds" in plan.reasoning


@pytest.mark.asyncio
async def test_plan_analysis_returns_agent_output_even_with_empty_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty reasoning field should warn but still return the generated plan."""
    expected_plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=True,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="",
    )

    class FakePlanningAgent:
        async def run(self, prompt: str, deps) -> _FakeAgentRunResult:
            return _FakeAgentRunResult(expected_plan)

    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "Paper abstract")
    monkeypatch.setattr("sciread.agent.coordinate.planner.get_section_length_map", lambda document, section_names: {"Intro": 240})
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.format_section_choices", lambda section_names, section_lengths: "Intro | 240 chars"
    )

    plan = await plan_analysis(
        document=object(),
        section_names=["Intro"],
        planning_agent=FakePlanningAgent(),
        timeout=10.0,
        model_identifier="mock-model",
        logger=get_logger(__name__),
    )

    assert plan is expected_plan


@pytest.mark.asyncio
async def test_plan_analysis_falls_back_to_default_plan_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Planner should recover with the default plan when the model call fails generically."""

    class BrokenPlanningAgent:
        async def run(self, prompt: str, deps):
            raise RuntimeError("planner exploded")

    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "Paper abstract")
    monkeypatch.setattr("sciread.agent.coordinate.planner.get_section_length_map", lambda document, section_names: {"Methods": 240})
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.format_section_choices", lambda section_names, section_lengths: "Methods | 240 chars"
    )

    plan = await plan_analysis(
        document=object(),
        section_names=["Methods"],
        planning_agent=BrokenPlanningAgent(),
        timeout=1.0,
        model_identifier="mock-model",
        logger=get_logger(__name__),
    )

    assert plan.analyze_metadata is True
    assert "Planning failed (planner exploded)" in plan.reasoning


@pytest.mark.asyncio
async def test_create_planning_agent_builds_prompt_and_retries_for_empty_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planning agent's system prompt should embed abstract/sections and retry on empty content."""

    class FakeAgent:
        def __init__(self, model, deps_type, output_type, retries) -> None:
            self.model = model
            self.deps_type = deps_type
            self.output_type = output_type
            self.retries = retries
            self.prompt_factory = None

        def system_prompt(self, func):
            self.prompt_factory = func
            return func

    monkeypatch.setattr("sciread.agent.coordinate.planner.Agent", FakeAgent)
    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "Paper abstract")
    monkeypatch.setattr("sciread.agent.coordinate.planner.get_section_length_map", lambda document, section_names: {"Methods": 480})
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.format_section_choices", lambda section_names, section_lengths: "Methods | 480 chars"
    )
    monkeypatch.setattr(
        "sciread.agent.coordinate.planner.build_analysis_planning_prompt", lambda abstract, sections: f"Prompt::{abstract}::{sections}"
    )

    planning_agent = create_planning_agent(model="model", max_retries=2, logger=get_logger(__name__))
    ctx = SimpleNamespace(deps=CoordinateDeps(document=SimpleNamespace(get_section_names=lambda: ["Methods"])))

    prompt = await planning_agent.prompt_factory(ctx)

    assert planning_agent.model == "model"
    assert planning_agent.retries == 2
    assert "Paper abstract" in prompt
    assert "Methods | 480 chars" in prompt
    assert "Prompt::Paper abstract::Methods | 480 chars" in prompt

    monkeypatch.setattr("sciread.agent.coordinate.planner.extract_abstract", lambda document, logger: "")
    empty_ctx = SimpleNamespace(deps=CoordinateDeps(document=SimpleNamespace(get_section_names=list)))

    with pytest.raises(ModelRetry):
        await planning_agent.prompt_factory(empty_ctx)


class _FakeAgentRunResult:
    def __init__(self, output) -> None:
        self.output = output


@pytest.mark.asyncio
async def test_execute_sub_agents_collects_success_failures_and_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executor should aggregate partial failures without losing section assignments."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract\nBody",
        metadata=DocumentMetadata(title="Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks(
        [
            Chunk(content="Abstract text", chunk_name="abstract"),
            Chunk(content="Methods text", chunk_name="methods"),
            Chunk(content="Conclusion text", chunk_name="conclusion"),
        ]
    )
    plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=True,
        analyze_experiments=False,
        analyze_future_directions=True,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=["methods"],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="focused",
    )

    def fake_create_expert_agent(model, max_retries: int, analysis_type: str, logger):
        class FakeExpertAgent:
            async def run(self, prompt: str, deps) -> _FakeAgentRunResult:
                assert prompt == "请执行专家分析"
                assert deps.analysis_type == analysis_type
                if analysis_type == "metadata":
                    return _FakeAgentRunResult(MetadataExtractionResult(title="Paper"))
                if analysis_type == "methodology":
                    return _FakeAgentRunResult(MethodologyResult(approach="Structured"))
                raise RuntimeError("future failed")

        return FakeExpertAgent()

    monkeypatch.setattr("sciread.agent.coordinate.executor.create_expert_agent", fake_create_expert_agent)

    results = await execute_sub_agents(
        document=document,
        analysis_plan=plan,
        model=object(),
        max_retries=2,
        logger=get_logger(__name__),
    )

    assert results["metadata"]["success"] is True
    assert results["metadata"]["result"].title == "Paper"
    assert results["methodology"]["success"] is True
    assert results["methodology"]["result"].approach == "Structured"
    assert results["future_directions"]["success"] is False
    assert "future failed" in results["future_directions"]["error"]
    assert results["_sections_analyzed"] == {
        "metadata": ["First 3 chunks"],
        "methodology": ["methods"],
        "future_directions": ["All sections"],
    }


@pytest.mark.asyncio
async def test_coordinate_agent_analyze_uses_custom_plan_without_replanning() -> None:
    """Custom plans should bypass planner calls and still build the final result."""
    agent = CoordinateAgent.__new__(CoordinateAgent)
    agent.logger = get_logger(__name__)
    agent.timeout = 30.0
    agent.max_retries = 1
    agent.model = object()
    agent.model_identifier = "mock-model"

    custom_plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="custom",
    )
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract\nBody",
        metadata=DocumentMetadata(title="Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks([Chunk(content="Abstract text", chunk_name="abstract")])

    async def fail_if_called(_document: Document, _section_names: list[str]) -> AnalysisPlan:
        raise AssertionError("plan_analysis should not be called when custom_plan is provided")

    async def fake_execute(_document: Document, analysis_plan: AnalysisPlan) -> dict[str, object]:
        assert analysis_plan is custom_plan
        return {
            "metadata": {"success": True, "result": MetadataExtractionResult(title="Paper")},
            "_sections_analyzed": {"metadata": ["First 3 chunks"]},
        }

    async def fake_synthesize(_analysis_plan: AnalysisPlan, sub_agent_results: dict[str, object], _document: Document) -> str:
        assert sub_agent_results["metadata"]["success"] is True
        return "Final synthesized report"

    agent.plan_analysis = fail_if_called
    agent.execute_sub_agents = fake_execute
    agent.synthesize_report = fake_synthesize

    result = await agent.analyze(document, custom_plan=custom_plan)

    assert result.analysis_plan == custom_plan
    assert result.metadata_result is not None
    assert result.metadata_result.title == "Paper"
    assert result.final_report == "Final synthesized report"
    assert result.sections_analyzed == {"metadata": ["First 3 chunks"]}


@pytest.mark.asyncio
async def test_coordinate_agent_analyze_reraises_execution_failures() -> None:
    """Top-level analyze should log and re-raise orchestration failures."""
    agent = CoordinateAgent.__new__(CoordinateAgent)
    agent.logger = get_logger(__name__)
    agent.timeout = 30.0
    agent.max_retries = 1
    agent.model = object()
    agent.model_identifier = "mock-model"

    custom_plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="custom",
    )
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract\nBody",
        metadata=DocumentMetadata(title="Paper", source_path=Path("paper.pdf")),
    )
    document._set_chunks([Chunk(content="Abstract text", chunk_name="abstract")])

    async def fake_execute(_document: Document, analysis_plan: AnalysisPlan) -> dict[str, object]:
        raise RuntimeError("sub-agent failure")

    agent.execute_sub_agents = fake_execute
    agent.synthesize_report = None

    with pytest.raises(RuntimeError, match="sub-agent failure"):
        await agent.analyze(document, custom_plan=custom_plan)


@pytest.mark.asyncio
async def test_synthesize_report_returns_fallback_when_no_successful_analyses() -> None:
    """Synthesis should return a stable fallback when every expert failed."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract",
        metadata=DocumentMetadata(title="Paper", source_path=Path("paper.pdf")),
    )
    logger = get_logger(__name__)

    report = await synthesize_report(
        analysis_plan=AnalysisPlan(
            analyze_metadata=True,
            analyze_previous_methods=False,
            analyze_research_questions=False,
            analyze_methodology=False,
            analyze_experiments=False,
            analyze_future_directions=False,
            previous_methods_sections=[],
            research_questions_sections=[],
            methodology_sections=[],
            experiments_sections=[],
            future_directions_sections=[],
            reasoning="reasoning",
        ),
        sub_agent_results={"metadata": {"success": False, "error": "failed"}},
        document=document,
        model=object(),
        max_retries=2,
        timeout=5.0,
        logger=logger,
    )

    assert report == "No successful analyses available for report synthesis. Please try again with different settings."


@pytest.mark.asyncio
async def test_synthesize_report_uses_metadata_title_and_agent_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesis should use successful analyses and return the synthesis agent output."""
    captured: dict[str, object] = {}

    class FakeSynthesisAgent:
        def __init__(self, **kwargs) -> None:
            captured["agent_init"] = kwargs

        async def run(self, prompt: str, deps: dict[str, object]) -> _FakeAgentRunResult:
            captured["prompt"] = prompt
            captured["deps"] = deps
            return _FakeAgentRunResult("Synthesized report")

    monkeypatch.setattr("sciread.agent.coordinate.synthesis.Agent", FakeSynthesisAgent)
    monkeypatch.setattr("sciread.agent.coordinate.synthesis.extract_abstract", lambda document, logger: "Paper abstract")

    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract",
        metadata=DocumentMetadata(title=None, source_path=Path("paper.pdf")),
    )
    metadata_result = MetadataExtractionResult(title="Recovered title")

    report = await synthesize_report(
        analysis_plan=AnalysisPlan(
            analyze_metadata=True,
            analyze_previous_methods=False,
            analyze_research_questions=False,
            analyze_methodology=False,
            analyze_experiments=False,
            analyze_future_directions=False,
            previous_methods_sections=[],
            research_questions_sections=[],
            methodology_sections=[],
            experiments_sections=[],
            future_directions_sections=[],
            reasoning="reasoning",
        ),
        sub_agent_results={
            "metadata": {"success": True, "result": metadata_result},
            "methodology": {"success": True, "result": MethodologyResult(approach="Structured")},
        },
        document=document,
        model=object(),
        max_retries=2,
        timeout=5.0,
        logger=get_logger(__name__),
    )

    assert report == "Synthesized report"
    assert captured["deps"] == {"successful_analyses": ["metadata", "methodology"]}
    assert "Recovered title" in captured["prompt"]


@pytest.mark.asyncio
async def test_synthesize_report_handles_timeout_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesis should return readable fallback messages for timeouts and generic failures."""
    document = Document(
        source_path=Path("paper.pdf"),
        text="Abstract",
        metadata=DocumentMetadata(title="Paper", source_path=Path("paper.pdf")),
    )
    sub_agent_results = {"metadata": {"success": True, "result": MetadataExtractionResult(title="Paper")}}
    plan = AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=False,
        analyze_research_questions=False,
        analyze_methodology=False,
        analyze_experiments=False,
        analyze_future_directions=False,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning="reasoning",
    )

    class TimeoutSynthesisAgent:
        def __init__(self, **kwargs) -> None:
            pass

        async def run(self, prompt: str, deps: dict[str, object]):
            raise TimeoutError("timed out")

    monkeypatch.setattr("sciread.agent.coordinate.synthesis.Agent", TimeoutSynthesisAgent)
    monkeypatch.setattr("sciread.agent.coordinate.synthesis.extract_abstract", lambda document, logger: "Paper abstract")
    timeout_report = await synthesize_report(
        analysis_plan=plan,
        sub_agent_results=sub_agent_results,
        document=document,
        model=object(),
        max_retries=2,
        timeout=3.0,
        logger=get_logger(__name__),
    )

    class BrokenSynthesisAgent:
        def __init__(self, **kwargs) -> None:
            pass

        async def run(self, prompt: str, deps: dict[str, object]):
            raise RuntimeError("boom")

    monkeypatch.setattr("sciread.agent.coordinate.synthesis.Agent", BrokenSynthesisAgent)
    error_report = await synthesize_report(
        analysis_plan=plan,
        sub_agent_results=sub_agent_results,
        document=document,
        model=object(),
        max_retries=2,
        timeout=3.0,
        logger=get_logger(__name__),
    )

    assert timeout_report == "Report synthesis timed out after 3.0 seconds. Please try again or use a shorter document."
    assert error_report == "Report synthesis failed: boom. Please try again."
