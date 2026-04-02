"""Tests for CLI command line interface."""

import subprocess

import pytest
from rich.console import Console

from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.models import ComprehensiveAnalysisResult
from sciread.entrypoints.cli import _render_coordinate_plan


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
