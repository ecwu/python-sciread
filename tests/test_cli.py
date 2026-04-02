"""Tests for CLI command line interface."""

import subprocess

import pytest
from rich.console import Console

from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.models import ComprehensiveAnalysisResult
from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import ConsensusPoint
from sciread.agent.discussion.models import DiscussionResult
from sciread.agent.discussion.models import DivergentView
from sciread.entrypoints import cli
from sciread.entrypoints.cli import _render_coordinate_plan
from sciread.entrypoints.cli import _render_discussion_overview
from sciread.entrypoints.cli import _resolve_plan_sections


def test_main():
    """Test that invalid CLI commands fail with proper error codes."""
    # Test that the old CLI interface (without subcommands) now fails
    with pytest.raises(subprocess.CalledProcessError):
        # Invalid command should fail since new CLI requires subcommands
        subprocess.check_output(["sciread", "foo", "foobar"], text=True)


def test_render_coordinate_plan_shows_subagents_and_sections():
    """Coordinate CLI should render a table for sub-agents and section assignments."""
    result = ComprehensiveAnalysisResult(
        analysis_plan=AnalysisPlan(
            analyze_metadata=True,
            analyze_previous_methods=True,
            analyze_research_questions=False,
            analyze_methodology=True,
            analyze_experiments=False,
            analyze_future_directions=True,
            previous_methods_sections=["Introduction", "Related Work"],
            research_questions_sections=[],
            methodology_sections=["Method"],
            experiments_sections=[],
            future_directions_sections=["Discussion", "Conclusion"],
            reasoning="Planner selected the most relevant sections for each expert.",
        ),
        final_report="## Report",
        sections_analyzed={
            "metadata": ["First 3 chunks"],
            "previous_methods": ["Introduction", "Related Work"],
            "methodology": ["Method"],
            "future_directions": ["Discussion", "Conclusion"],
        },
    )
    test_console = Console(record=True, width=120)

    _render_coordinate_plan(result, target_console=test_console)

    rendered = test_console.export_text()
    assert "Coordinate Analysis Plan" in rendered
    assert "Sub-Agent Section Plan" in rendered
    assert "Metadata" in rendered
    assert "Previous Methods" in rendered
    assert "Research Questions" in rendered
    assert "First 3 chunks" in rendered
    assert "Introduction" in rendered
    assert "Discussion" in rendered


def test_resolve_plan_sections_handles_disabled_executed_and_fallback_states() -> None:
    """Plan section rendering should prefer executed sections and fall back predictably."""
    result = ComprehensiveAnalysisResult(
        analysis_plan=AnalysisPlan(
            analyze_metadata=False,
            analyze_previous_methods=True,
            analyze_research_questions=True,
            analyze_methodology=True,
            analyze_experiments=True,
            analyze_future_directions=True,
            previous_methods_sections=["Related Work"],
            research_questions_sections=[],
            methodology_sections=["Method"],
            experiments_sections=[],
            future_directions_sections=["Conclusion"],
            reasoning="reasoning",
        ),
        sections_analyzed={"previous_methods": ["Executed Section"]},
    )

    assert _resolve_plan_sections(result, "metadata", "analyze_metadata", None) == "-"
    assert _resolve_plan_sections(result, "previous_methods", "analyze_previous_methods", "previous_methods_sections") == "Executed Section"
    assert _resolve_plan_sections(result, "methodology", "analyze_methodology", "methodology_sections") == "Method"
    assert _resolve_plan_sections(result, "experiments", "analyze_experiments", "experiments_sections") == "All sections"


def test_render_discussion_overview_handles_non_list_sections() -> None:
    """Overview rendering should tolerate malformed section payloads."""
    test_console = Console(record=True, width=120)

    original_console = cli.console
    cli.console = test_console
    try:
        _render_discussion_overview(
            {
                "document_title": "Paper",
                "total_content_chars": 123,
                "chunk_count": 4,
                "section_names": "not-a-list",
            }
        )
    finally:
        cli.console = original_console

    rendered = test_console.export_text()
    assert "Discussion Analysis" in rendered
    assert "No named sections found" in rendered


def test_run_returns_help_for_missing_command(capsys) -> None:
    """The top-level CLI should print help when no subcommand is supplied."""
    exit_code = cli.run(["sciread"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Academic Paper Analysis Tool" in captured.out


def test_run_coordinate_success_renders_plan_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate CLI should invoke the use case and render the report."""
    test_console = Console(record=True, width=120)
    result = ComprehensiveAnalysisResult(
        analysis_plan=AnalysisPlan(
            analyze_metadata=True,
            analyze_previous_methods=True,
            analyze_research_questions=False,
            analyze_methodology=True,
            analyze_experiments=False,
            analyze_future_directions=True,
            previous_methods_sections=["Related Work"],
            research_questions_sections=[],
            methodology_sections=["Method"],
            experiments_sections=[],
            future_directions_sections=["Conclusion"],
            reasoning="Planner reasoning",
        ),
        final_report="## Final Coordinate Report",
        sections_analyzed={"previous_methods": ["Related Work"]},
    )

    async def fake_run_coordinate_analysis(document_file: str, model: str) -> ComprehensiveAnalysisResult:
        assert document_file == "paper.pdf"
        assert model == "test-model"
        return result

    monkeypatch.setattr(cli, "run_coordinate_analysis", fake_run_coordinate_analysis)
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.run(["sciread", "coordinate", "paper.pdf", "--model", "test-model"])

    rendered = test_console.export_text()
    assert exit_code == 0
    assert "Coordinate Analysis Plan" in rendered
    assert "Final Coordinate Report" in rendered


@pytest.mark.parametrize(
    ("argv", "expected_output"),
    [
        (
            ["sciread", "react", "paper.pdf", "What changed?", "--model", "react-model", "--max-loops", "7", "--no-progress"],
            "react",
        ),
        (["sciread", "simple", "paper.pdf", "--model", "simple-model"], "simple"),
    ],
)
def test_run_react_and_simple_success_paths(monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_output: str) -> None:
    """React and simple commands should return zero on success."""
    test_console = Console(record=True, width=120)
    calls: dict[str, object] = {}

    async def fake_run_react_analysis(
        document_file: str,
        task: str,
        *,
        model: str,
        max_loops: int,
        show_progress: bool,
    ) -> str:
        calls["react"] = {
            "document_file": document_file,
            "task": task,
            "model": model,
            "max_loops": max_loops,
            "show_progress": show_progress,
        }
        return "react result"

    async def fake_run_simple_analysis(document_file: str, model: str) -> str:
        calls["simple"] = {"document_file": document_file, "model": model}
        return "simple result"

    monkeypatch.setattr(cli, "run_react_analysis", fake_run_react_analysis)
    monkeypatch.setattr(cli, "run_simple_analysis", fake_run_simple_analysis)
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.run(argv)

    assert exit_code == 0
    if expected_output == "react":
        assert calls["react"] == {
            "document_file": "paper.pdf",
            "task": "What changed?",
            "model": "react-model",
            "max_loops": 7,
            "show_progress": False,
        }
    else:
        assert calls["simple"] == {"document_file": "paper.pdf", "model": "simple-model"}


def test_run_discussion_success_renders_overview_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discussion CLI should render overview metadata and the synthesized report."""
    test_console = Console(record=True, width=120)
    overview = {
        "document_title": "Paper Title",
        "total_content_chars": 321,
        "chunk_count": 7,
        "section_names": ["intro", "results"],
    }
    result = DiscussionResult(
        document_title="Paper Title",
        summary="Summary text",
        key_contributions=["Contribution A"],
        significance="Important",
        consensus_points=[
            ConsensusPoint(
                topic="Topic A",
                content="Consensus content",
                supporting_agents=[AgentPersonality.CRITICAL_EVALUATOR],
                strength=0.8,
            )
        ],
        divergent_views=[
            DivergentView(
                topic="Topic B",
                content="Divergent content",
                holding_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
                reasoning="Different assumptions",
            )
        ],
        final_insights=[
            AgentInsight(
                agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
                content="Insight content",
                importance_score=0.9,
                confidence=0.85,
                supporting_evidence=["Evidence A"],
                related_sections=["Results"],
            )
        ],
        confidence_score=0.75,
        discussion_metadata={"rounds": 3, "error": "ignored"},
    )

    async def fake_run_discussion_analysis(document_file: str, model: str) -> tuple[dict[str, object], DiscussionResult]:
        assert document_file == "paper.pdf"
        assert model == "discussion-model"
        return overview, result

    monkeypatch.setattr(cli, "run_discussion_analysis", fake_run_discussion_analysis)
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.run(["sciread", "discussion", "paper.pdf", "--model", "discussion-model"])

    rendered = test_console.export_text()
    assert exit_code == 0
    assert "Discussion Analysis" in rendered
    assert "Paper Title" in rendered
    assert "Contribution A" in rendered
    assert "Consensus content" in rendered
    assert "Divergent content" in rendered
    assert "Evidence A" in rendered
    assert "Results" in rendered


@pytest.mark.parametrize(
    ("argv", "patch_target", "expected_message"),
    [
        (
            ["sciread", "coordinate", "paper.pdf"],
            "run_coordinate_analysis",
            "Error: coordinate failure",
        ),
        (
            ["sciread", "react", "paper.pdf"],
            "run_react_analysis",
            "Error: react failure",
        ),
        (
            ["sciread", "simple", "paper.pdf"],
            "run_simple_analysis",
            "Error: simple failure",
        ),
        (
            ["sciread", "discussion", "paper.pdf"],
            "run_discussion_analysis",
            "Error: discussion failure",
        ),
    ],
)
def test_run_returns_error_code_when_command_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    argv: list[str],
    patch_target: str,
    expected_message: str,
) -> None:
    """Each subcommand should report failures and return a non-zero exit code."""

    async def fake_failure(*_args, **_kwargs):
        failure_messages = {
            "run_coordinate_analysis": "coordinate failure",
            "run_react_analysis": "react failure",
            "run_simple_analysis": "simple failure",
            "run_discussion_analysis": "discussion failure",
        }
        raise RuntimeError(failure_messages[patch_target])

    monkeypatch.setattr(cli, patch_target, fake_failure)

    exit_code = cli.run(argv)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert expected_message in captured.out
