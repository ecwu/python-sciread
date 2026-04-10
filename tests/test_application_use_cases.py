"""Tests for application-layer use cases and shared helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sciread.agent.simple import DEFAULT_TASK_PROMPT
from sciread.application.use_cases import common
from sciread.application.use_cases.coordinate import run_coordinate_analysis
from sciread.application.use_cases.discussion import run_discussion_analysis
from sciread.application.use_cases.react import run_react_analysis
from sciread.application.use_cases.search_react import run_search_react_analysis
from sciread.application.use_cases.simple import run_simple_analysis


def test_ensure_file_exists_returns_path_for_existing_file(tmp_path: Path) -> None:
    """Existing files should pass validation unchanged."""
    document_path = tmp_path / "paper.txt"
    document_path.write_text("content")

    result = common.ensure_file_exists(str(document_path))

    assert result == document_path


def test_ensure_file_exists_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing files should raise a useful error."""
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Document file not found"):
        common.ensure_file_exists(str(missing_path))


def test_load_document_uses_project_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document loading should always enable auto splitting."""
    captured: dict[str, object] = {}
    sentinel_document = object()

    def fake_ensure(path: str) -> Path:
        captured["validated_path"] = path
        return Path(path)

    def fake_from_file(path: str, *, to_markdown: bool, auto_split: bool) -> object:
        captured["document_path"] = path
        captured["to_markdown"] = to_markdown
        captured["auto_split"] = auto_split
        return sentinel_document

    monkeypatch.setattr(common, "ensure_file_exists", fake_ensure)
    monkeypatch.setattr(common.Document, "from_file", staticmethod(fake_from_file))

    result = common.load_document("paper.pdf", to_markdown=True)

    assert result is sentinel_document
    assert captured == {
        "validated_path": "paper.pdf",
        "document_path": "paper.pdf",
        "to_markdown": True,
        "auto_split": True,
    }


@pytest.mark.asyncio
async def test_run_simple_analysis_delegates_with_expected_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simple analysis should call the public simple-agent entrypoint with defaults."""
    captured: dict[str, object] = {}

    async def fake_analyze_file_with_simple(**kwargs: object) -> str:
        captured.update(kwargs)
        return "simple result"

    monkeypatch.setattr(
        "sciread.application.use_cases.simple.analyze_file_with_simple",
        fake_analyze_file_with_simple,
    )

    result = await run_simple_analysis("paper.pdf", model="provider/model")

    assert result == "simple result"
    assert captured == {
        "file_path": "paper.pdf",
        "task_prompt": DEFAULT_TASK_PROMPT,
        "model": "provider/model",
        "remove_references": True,
        "clean_text": True,
    }


@pytest.mark.asyncio
async def test_run_simple_analysis_reraises_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simple analysis should surface agent failures unchanged."""

    async def fake_analyze_file_with_simple(**_: object) -> str:
        raise RuntimeError("simple failure")

    monkeypatch.setattr(
        "sciread.application.use_cases.simple.analyze_file_with_simple",
        fake_analyze_file_with_simple,
    )

    with pytest.raises(RuntimeError, match="simple failure"):
        await run_simple_analysis("paper.pdf")


@pytest.mark.asyncio
async def test_run_react_analysis_delegates_with_expected_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """ReAct analysis should pass through task and loop settings."""
    captured: dict[str, object] = {}

    async def fake_analyze_file_with_react(*args: object, **kwargs: object) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "react result"

    monkeypatch.setattr(
        "sciread.application.use_cases.react.analyze_file_with_react",
        fake_analyze_file_with_react,
    )

    result = await run_react_analysis(
        "paper.pdf",
        "What matters?",
        model="deepseek-chat",
        max_loops=7,
        show_progress=False,
    )

    assert result == "react result"
    assert captured["args"] == ("paper.pdf", "What matters?")
    assert captured["kwargs"] == {
        "model": "deepseek-chat",
        "max_loops": 7,
        "to_markdown": True,
        "show_progress": False,
    }


@pytest.mark.asyncio
async def test_run_react_analysis_reraises_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """ReAct analysis should surface agent failures unchanged."""

    async def fake_analyze_file_with_react(*_: object, **__: object) -> str:
        raise RuntimeError("react failure")

    monkeypatch.setattr(
        "sciread.application.use_cases.react.analyze_file_with_react",
        fake_analyze_file_with_react,
    )

    with pytest.raises(RuntimeError, match="react failure"):
        await run_react_analysis("paper.pdf", "task")


@pytest.mark.asyncio
async def test_run_search_react_analysis_delegates_with_expected_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search-react analysis should pass through retrieval settings."""
    captured: dict[str, object] = {}

    async def fake_analyze_file_with_search_react(*args: object, **kwargs: object) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "search-react result"

    monkeypatch.setattr(
        "sciread.application.use_cases.search_react.analyze_file_with_search_react",
        fake_analyze_file_with_search_react,
    )

    result = await run_search_react_analysis(
        "paper.pdf",
        "What matters?",
        model="deepseek-chat",
        max_loops=7,
        show_progress=False,
        retriever="tree",
        compare=["lexical", "tree"],
        top_k=6,
        neighbor_window=2,
    )

    assert result == "search-react result"
    assert captured["args"] == ("paper.pdf", "What matters?")
    assert captured["kwargs"] == {
        "model": "deepseek-chat",
        "max_loops": 7,
        "to_markdown": True,
        "show_progress": False,
        "retriever": "tree",
        "compare": ["lexical", "tree"],
        "top_k": 6,
        "neighbor_window": 2,
    }


@pytest.mark.asyncio
async def test_run_search_react_analysis_reraises_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search-react analysis should surface agent failures unchanged."""

    async def fake_analyze_file_with_search_react(*_: object, **__: object) -> str:
        raise RuntimeError("search-react failure")

    monkeypatch.setattr(
        "sciread.application.use_cases.search_react.analyze_file_with_search_react",
        fake_analyze_file_with_search_react,
    )

    with pytest.raises(RuntimeError, match="search-react failure"):
        await run_search_react_analysis("paper.pdf", "task")


@pytest.mark.asyncio
async def test_run_coordinate_analysis_loads_document_and_runs_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate analysis should build the agent, load the document, and return the agent result."""
    captured: dict[str, object] = {}
    document = SimpleNamespace(
        text="Document body",
        chunks=["chunk-1", "chunk-2"],
        get_section_names=lambda: ["Introduction", "Method"],
    )
    expected_result = SimpleNamespace(
        total_execution_time=1.25,
        execution_summary={"total_agents_executed": 3, "successful_agents": 3},
        final_report="Final coordinate report",
    )

    class FakeCoordinateAgent:
        def __init__(self, model: str) -> None:
            captured["model"] = model

        async def analyze(self, loaded_document: object) -> object:
            captured["document"] = loaded_document
            return expected_result

    def fake_load_document(path: str, *, to_markdown: bool) -> object:
        captured["path"] = path
        captured["to_markdown"] = to_markdown
        return document

    monkeypatch.setattr("sciread.application.use_cases.coordinate.CoordinateAgent", FakeCoordinateAgent)
    monkeypatch.setattr("sciread.application.use_cases.coordinate.load_document", fake_load_document)

    result = await run_coordinate_analysis("paper.pdf", model="coordinate-model")

    assert result is expected_result
    assert captured == {
        "model": "coordinate-model",
        "path": "paper.pdf",
        "to_markdown": True,
        "document": document,
    }


@pytest.mark.asyncio
async def test_run_coordinate_analysis_rejects_empty_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate analysis should fail fast when document extraction returns no content."""
    document = SimpleNamespace(text="   ", chunks=[], get_section_names=list)

    monkeypatch.setattr("sciread.application.use_cases.coordinate.load_document", lambda *_args, **_kwargs: document)
    monkeypatch.setattr("sciread.application.use_cases.coordinate.CoordinateAgent", lambda _model: object())

    with pytest.raises(ValueError, match="Failed to load PDF: no text content extracted"):
        await run_coordinate_analysis("paper.pdf")


@pytest.mark.asyncio
async def test_run_coordinate_analysis_reraises_agent_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate analysis should surface analysis failures unchanged."""
    document = SimpleNamespace(text="Document body", chunks=["chunk"], get_section_names=lambda: ["Intro"])

    class FakeCoordinateAgent:
        def __init__(self, _model: str) -> None:
            pass

        async def analyze(self, _document: object) -> object:
            raise RuntimeError("coordinate failure")

    monkeypatch.setattr("sciread.application.use_cases.coordinate.load_document", lambda *_args, **_kwargs: document)
    monkeypatch.setattr("sciread.application.use_cases.coordinate.CoordinateAgent", FakeCoordinateAgent)

    with pytest.raises(RuntimeError, match="coordinate failure"):
        await run_coordinate_analysis("paper.pdf")


@pytest.mark.asyncio
async def test_run_discussion_analysis_builds_overview_and_runs_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discussion analysis should return both document overview and agent result."""
    captured: dict[str, object] = {}
    metadata = SimpleNamespace(title="Paper Title")
    document = SimpleNamespace(
        text="Document body",
        chunks=["chunk-1", "chunk-2", "chunk-3"],
        metadata=metadata,
        get_section_names=lambda: ["Introduction", "Results"],
    )
    expected_result = object()

    class FakeDiscussionAgent:
        def __init__(self, model: str) -> None:
            captured["model"] = model

        async def analyze_document(self, loaded_document: object) -> object:
            captured["document"] = loaded_document
            return expected_result

    def fake_load_document(path: str, *, to_markdown: bool) -> object:
        captured["path"] = path
        captured["to_markdown"] = to_markdown
        return document

    monkeypatch.setattr("sciread.application.use_cases.discussion.DiscussionAgent", FakeDiscussionAgent)
    monkeypatch.setattr("sciread.application.use_cases.discussion.load_document", fake_load_document)

    overview, result = await run_discussion_analysis("paper.pdf", model="discussion-model")

    assert result is expected_result
    assert overview == {
        "document_title": "Paper Title",
        "total_content_chars": len(document.text),
        "chunk_count": len(document.chunks),
        "section_names": ["Introduction", "Results"],
    }
    assert captured == {
        "model": "discussion-model",
        "path": "paper.pdf",
        "to_markdown": True,
        "document": document,
    }


@pytest.mark.asyncio
async def test_run_discussion_analysis_uses_untitled_when_metadata_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discussion analysis should provide a stable fallback title."""
    document = SimpleNamespace(
        text="Document body",
        chunks=["chunk"],
        metadata=SimpleNamespace(title=None),
        get_section_names=list,
    )

    class FakeDiscussionAgent:
        def __init__(self, _model: str) -> None:
            pass

        async def analyze_document(self, _loaded_document: object) -> str:
            return "discussion result"

    monkeypatch.setattr("sciread.application.use_cases.discussion.DiscussionAgent", FakeDiscussionAgent)
    monkeypatch.setattr("sciread.application.use_cases.discussion.load_document", lambda *_args, **_kwargs: document)

    overview, result = await run_discussion_analysis("paper.pdf")

    assert overview["document_title"] == "Untitled"
    assert result == "discussion result"


@pytest.mark.asyncio
async def test_run_discussion_analysis_rejects_empty_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discussion analysis should fail fast when document extraction returns no content."""
    document = SimpleNamespace(
        text="  ",
        chunks=[],
        metadata=SimpleNamespace(title="Title"),
        get_section_names=list,
    )

    monkeypatch.setattr("sciread.application.use_cases.discussion.load_document", lambda *_args, **_kwargs: document)
    monkeypatch.setattr("sciread.application.use_cases.discussion.DiscussionAgent", lambda _model: object())

    with pytest.raises(ValueError, match="Failed to load document: no text content extracted"):
        await run_discussion_analysis("paper.pdf")


@pytest.mark.asyncio
async def test_run_discussion_analysis_reraises_agent_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discussion analysis should surface analysis failures unchanged."""
    document = SimpleNamespace(
        text="Document body",
        chunks=["chunk"],
        metadata=SimpleNamespace(title="Title"),
        get_section_names=lambda: ["Intro"],
    )

    class FakeDiscussionAgent:
        def __init__(self, _model: str) -> None:
            pass

        async def analyze_document(self, _loaded_document: object) -> object:
            raise RuntimeError("discussion failure")

    monkeypatch.setattr("sciread.application.use_cases.discussion.DiscussionAgent", FakeDiscussionAgent)
    monkeypatch.setattr("sciread.application.use_cases.discussion.load_document", lambda *_args, **_kwargs: document)

    with pytest.raises(RuntimeError, match="discussion failure"):
        await run_discussion_analysis("paper.pdf")
