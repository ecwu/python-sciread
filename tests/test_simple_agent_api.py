"""Tests for the simple agent public API."""

from types import SimpleNamespace

import pytest

from sciread.agent.simple import SimpleAgent
from sciread.agent.simple import analyze_file_with_simple
from sciread.agent.simple import analyze_file_with_simple_sync
from sciread.agent.simple.agent import _build_simple_content
from sciread.agent.simple.models import SimpleAnalysisResult


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


def test_build_simple_content_uses_sections_and_removes_references() -> None:
    """Section-based content assembly should drop reference sections before rendering."""
    captured: dict[str, object] = {}

    class DummyDocument:
        def __init__(self) -> None:
            self.chunks = ["chunk"]
            self.text = "raw text"

        def get_section_names(self) -> list[str]:
            return ["Abstract", "References", "Method"]

        def get_for_llm(self, **kwargs: object) -> str:
            captured.update(kwargs)
            return "Section content\nReferences\n" + ("citation\n" * 400)

    content = _build_simple_content(
        document=DummyDocument(),
        include_metadata=True,
        remove_references=True,
        clean_text=True,
        max_tokens=1024,
    )

    assert captured == {
        "section_names": ["Abstract", "Method"],
        "max_tokens": 1024,
        "include_headers": True,
        "clean_text": True,
    }
    assert content == "Section content"


def test_build_simple_content_falls_back_to_cleaned_raw_text() -> None:
    """When section rendering is empty, the helper should fall back to cleaned raw text."""

    class DummyDocument:
        def __init__(self) -> None:
            self.chunks: list[object] = []
            self.text = "Body text\nReferences\n" + ("citation\n" * 400)

        def get_section_names(self) -> list[str]:
            return []

        def get_for_llm(self, **kwargs: object) -> str:
            return "   "

    content = _build_simple_content(
        document=DummyDocument(),
        include_metadata=False,
        remove_references=True,
        clean_text=True,
        max_tokens=256,
    )

    assert content == "Body text"


@pytest.mark.asyncio
async def test_simple_run_analysis_uses_safe_agent_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """The main analysis path should delegate to safe_agent_execution and return the agent output."""
    captured: dict[str, object] = {}
    agent = SimpleAgent.__new__(SimpleAgent)
    agent.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    agent.timeout = 12.0

    class FakeInnerAgent:
        def run(self, prompt: str, deps) -> str:
            captured["prompt"] = prompt
            captured["deps"] = deps
            return "run-coro"

    async def fake_safe_agent_execution(coro, timeout: float, operation_name: str):
        captured["safe_call"] = {
            "coro": coro,
            "timeout": timeout,
            "operation_name": operation_name,
        }
        return SimpleNamespace(output="analysis result")

    monkeypatch.setattr("sciread.agent.simple.agent.safe_agent_execution", fake_safe_agent_execution)
    monkeypatch.setattr("sciread.agent.simple.agent.console.print", lambda *args, **kwargs: None)
    agent.agent = FakeInnerAgent()

    document = SimpleNamespace(source_path="paper.txt")
    result = await agent.run_analysis(document=document, task_prompt="Summarize", audience="researchers")

    assert result == "analysis result"
    assert captured["prompt"] == "请根据任务要求分析文档"
    assert captured["deps"].task_prompt == "Summarize"
    assert captured["deps"].additional_context == {"audience": "researchers"}
    assert captured["safe_call"] == {
        "coro": "run-coro",
        "timeout": 12.0,
        "operation_name": "document analysis",
    }


@pytest.mark.asyncio
async def test_simple_run_structured_analysis_returns_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structured analysis should return the parsed structured model from the safe execution wrapper."""
    captured: dict[str, object] = {}
    agent = SimpleAgent.__new__(SimpleAgent)
    agent.logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    agent.timeout = 15.0
    agent.max_retries = 2
    agent.model = object()

    class FakeStructuredAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["agent_init"] = kwargs

        def system_prompt(self, func):
            captured["system_prompt_registered"] = True
            return func

        def run(self, prompt: str, deps):
            captured["prompt"] = prompt
            captured["deps"] = deps
            return "structured-coro"

    async def fake_safe_agent_execution(coro, timeout: float, operation_name: str):
        captured["safe_call"] = {
            "coro": coro,
            "timeout": timeout,
            "operation_name": operation_name,
        }
        return SimpleNamespace(output=SimpleAnalysisResult(report="structured result"))

    monkeypatch.setattr("sciread.agent.simple.agent.Agent", FakeStructuredAgent)
    monkeypatch.setattr("sciread.agent.simple.agent.safe_agent_execution", fake_safe_agent_execution)
    monkeypatch.setattr("sciread.agent.simple.agent.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_build_system_prompt", lambda ctx: "prompt")

    result = await agent.run_structured_analysis(
        document=SimpleNamespace(source_path="paper.txt"),
        task_prompt="Summarize",
        audience="researchers",
    )

    assert result.report == "structured result"
    assert captured["agent_init"]["output_type"] is SimpleAnalysisResult
    assert captured["system_prompt_registered"] is True
    assert captured["prompt"] == "请根据任务要求分析文档"
    assert captured["deps"].additional_context == {"audience": "researchers"}
    assert captured["safe_call"] == {
        "coro": "structured-coro",
        "timeout": 15.0,
        "operation_name": "structured document analysis",
    }


def test_analyze_file_with_simple_sync_uses_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync wrapper should delegate to asyncio.run with the async entrypoint coroutine."""
    captured: dict[str, object] = {}

    async def fake_async_entrypoint(**kwargs: object) -> str:
        return "unused"

    def fake_asyncio_run(coro) -> str:
        captured["coro"] = coro
        coro.close()
        return "sync result"

    monkeypatch.setattr("sciread.agent.simple.agent.analyze_file_with_simple", fake_async_entrypoint)
    monkeypatch.setattr("sciread.agent.simple.agent.asyncio.run", fake_asyncio_run)

    result = analyze_file_with_simple_sync("paper.txt", "Summarize", model="mock-model")

    assert result == "sync result"
    assert captured["coro"] is not None
