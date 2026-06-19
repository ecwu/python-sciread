"""System smoke tests for CLI entrypoints using fake external boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from sciread.agent.coordinate.models import AnalysisPlan
from sciread.agent.coordinate.models import ComprehensiveAnalysisResult
from sciread.agent.react.models import ReActIterationOutput
from sciread.agent.search_react.models import SearchReactIterationOutput
from sciread.agent.search_react.models import SearchReactStrategyRun
from sciread.entrypoints import cli

pytestmark = pytest.mark.system


def test_python_module_help_smoke() -> None:
    """The installed module entrypoint should expose CLI help."""
    result = subprocess.run(
        [sys.executable, "-m", "sciread", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Academic Paper Analysis Tool" in result.stdout
    assert "search-react" in result.stdout


def test_simple_cli_runs_through_document_loading_with_fake_agent(
    monkeypatch: pytest.MonkeyPatch,
    sample_markdown_file: Path,
) -> None:
    """Simple mode should load a real document and cross into a fake agent boundary."""
    calls: dict[str, object] = {}

    async def fake_run_analysis(self, document, task_prompt: str, **kwargs):
        calls["source_path"] = document.source_path
        calls["chunk_count"] = len(document.chunks)
        calls["task_prompt"] = task_prompt
        calls["kwargs"] = kwargs
        return "Fake simple final report"

    def fake_init(self, model: str, *args, **kwargs) -> None:
        self.model_identifier = model

    monkeypatch.setattr("sciread.agent.simple.agent.SimpleAgent.__init__", fake_init)
    monkeypatch.setattr("sciread.agent.simple.agent.SimpleAgent.run_analysis", fake_run_analysis)

    exit_code = cli.run(["sciread", "simple", str(sample_markdown_file), "--model", "fake-model"])

    assert exit_code == 0
    assert calls["source_path"] == sample_markdown_file
    assert calls["chunk_count"] > 0
    assert "论文" in str(calls["task_prompt"])


def test_coordinate_cli_renders_plan_and_report_with_fake_agent(
    monkeypatch: pytest.MonkeyPatch,
    sample_markdown_file: Path,
) -> None:
    """Coordinate mode should load a document through the use case and render the result."""
    test_console = Console(record=True, width=120)
    calls: dict[str, object] = {}

    class FakeCoordinateAgent:
        def __init__(self, model: str) -> None:
            calls["model"] = model

        async def analyze(self, document):
            calls["source_path"] = document.source_path
            calls["sections"] = document.get_section_names()
            return ComprehensiveAnalysisResult(
                analysis_plan=AnalysisPlan(
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
                    reasoning="Fake planner selected method coverage.",
                ),
                final_report="Fake coordinate final report",
                execution_summary={
                    "total_agents_executed": 2,
                    "successful_agents": 2,
                },
                sections_analyzed={"metadata": ["First 3 chunks"], "methodology": ["Methods"]},
            )

    monkeypatch.setattr("sciread.application.use_cases.coordinate.CoordinateAgent", FakeCoordinateAgent)
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.run(["sciread", "coordinate", str(sample_markdown_file), "--model", "fake-model"])

    rendered = test_console.export_text()
    assert exit_code == 0
    assert calls["model"] == "fake-model"
    assert calls["source_path"] == sample_markdown_file
    assert any(str(section).lower().endswith("methods") for section in calls["sections"])
    assert "Coordinate Analysis Plan" in rendered
    assert "Fake coordinate final report" in rendered


def test_react_cli_runs_through_document_loading_with_fake_agent(
    monkeypatch: pytest.MonkeyPatch,
    sample_markdown_file: Path,
) -> None:
    """React mode should pass CLI options through the use case after loading the document."""
    calls: dict[str, object] = {}

    async def fake_run_analysis(self, document, task: str, max_loops: int, show_progress: bool):
        calls["source_path"] = document.source_path
        calls["task"] = task
        calls["max_loops"] = max_loops
        calls["show_progress"] = show_progress
        return ReActIterationOutput(thoughts="done", should_continue=False, report="Fake react final report")

    monkeypatch.setattr("sciread.agent.react.agent.get_model", lambda model: object())
    monkeypatch.setattr("sciread.agent.react.agent.ReActAgent.run_analysis", fake_run_analysis)

    exit_code = cli.run(
        [
            "sciread",
            "react",
            str(sample_markdown_file),
            "What are the methods?",
            "--model",
            "fake-model",
            "--max-loops",
            "2",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "source_path": sample_markdown_file,
        "task": "What are the methods?",
        "max_loops": 2,
        "show_progress": False,
    }


def test_search_react_cli_runs_strategy_and_compare_paths_with_fake_agent(
    monkeypatch: pytest.MonkeyPatch,
    sample_markdown_file: Path,
) -> None:
    """Search-react should load the document and run each requested strategy."""
    calls: list[dict[str, object]] = []

    async def fake_run_analysis(
        self,
        document,
        task: str,
        *,
        strategy: str,
        top_k: int,
        neighbor_window: int,
        max_loops: int,
        show_progress: bool,
    ):
        calls.append(
            {
                "source_path": document.source_path,
                "task": task,
                "strategy": strategy,
                "top_k": top_k,
                "neighbor_window": neighbor_window,
                "max_loops": max_loops,
                "show_progress": show_progress,
            }
        )
        return SearchReactStrategyRun(
            strategy=strategy,
            output=SearchReactIterationOutput(thoughts="done", should_continue=False, report=f"Fake {strategy} report"),
            retrieved_chunks=[],
            total_time_seconds=0.01,
        )

    monkeypatch.setattr("sciread.agent.search_react.agent.SearchReactAgent.run_analysis", fake_run_analysis)

    exit_code = cli.run(
        [
            "sciread",
            "search-react",
            str(sample_markdown_file),
            "What evidence supports the results?",
            "--model",
            "fake-model",
            "--max-loops",
            "3",
            "--retriever",
            "lexical",
            "--compare",
            "lexical,tree",
            "--top-k",
            "4",
            "--neighbor-window",
            "2",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert [call["strategy"] for call in calls] == ["lexical", "tree"]
    assert calls[0] == {
        "source_path": sample_markdown_file,
        "task": "What evidence supports the results?",
        "strategy": "lexical",
        "top_k": 4,
        "neighbor_window": 2,
        "max_loops": 3,
        "show_progress": False,
    }


def test_cli_missing_file_returns_stable_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Missing user input files should produce a stable non-zero CLI result."""
    missing_file = tmp_path / "missing.md"

    exit_code = cli.run(["sciread", "simple", str(missing_file), "--model", "fake-model"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Document file not found" in captured.out
