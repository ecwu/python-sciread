"""Tests for the simple agent public API."""

import pytest

from sciread.agent.simple import SimpleAgent
from sciread.agent.simple import analyze_file_with_simple
from sciread.agent.simple import analyze_file_with_simple_sync


def test_simple_public_api_uses_clarified_names() -> None:
    """The simple agent API should use analysis-oriented method names."""
    assert analyze_file_with_simple.__name__ == "analyze_file_with_simple"
    assert analyze_file_with_simple_sync.__name__ == "analyze_file_with_simple_sync"
    assert hasattr(SimpleAgent, "run_analysis")
    assert hasattr(SimpleAgent, "run_structured_analysis")
    assert not hasattr(SimpleAgent, "analyze")
    assert not hasattr(SimpleAgent, "analyze_structured")


@pytest.mark.asyncio
async def test_analyze_file_with_simple_delegates_to_run_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The file entrypoint should load the document and delegate to SimpleAgent.run_analysis."""
    captured: dict[str, object] = {}
    sentinel_document = object()

    def fake_validate(file_path: str) -> None:
        captured["validated_file_path"] = file_path

    def fake_load(file_path: str, to_markdown: bool = False) -> object:
        captured["loaded_file_path"] = file_path
        captured["loaded_to_markdown"] = to_markdown
        return sentinel_document

    class FakeSimpleAgent:
        def __init__(self, model: str) -> None:
            captured["model"] = model

        async def run_analysis(self, **kwargs: object) -> str:
            captured["run_analysis_kwargs"] = kwargs
            return "simple result"

    monkeypatch.setattr("sciread.agent.simple.agent._validate_document_file", fake_validate)
    monkeypatch.setattr("sciread.agent.simple.agent.load_document_for_simple_analysis", fake_load)
    monkeypatch.setattr("sciread.agent.simple.agent.SimpleAgent", FakeSimpleAgent)

    result = await analyze_file_with_simple(
        file_path="paper.txt",
        task_prompt="Summarize this paper",
        model="test-model",
        to_markdown=True,
        include_metadata=False,
    )

    assert result == "simple result"
    assert captured["validated_file_path"] == "paper.txt"
    assert captured["loaded_file_path"] == "paper.txt"
    assert captured["loaded_to_markdown"] is True
    assert captured["model"] == "test-model"
    assert captured["run_analysis_kwargs"] == {
        "document": sentinel_document,
        "task_prompt": "Summarize this paper",
        "include_metadata": False,
        "remove_references": True,
        "clean_text": True,
    }
