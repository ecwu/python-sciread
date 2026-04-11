"""Tests for search-react tools and compare workflow."""

from types import SimpleNamespace

import pytest

from sciread.agent.search_react import analyze_file_with_search_react
from sciread.agent.search_react.agent import SearchReactAgent
from sciread.agent.search_react.agent import add_memory
from sciread.agent.search_react.agent import get_all_memory
from sciread.agent.search_react.agent import inspect_section_tree
from sciread.agent.search_react.agent import search_document
from sciread.agent.search_react.models import SearchReactDeps
from sciread.agent.search_react.models import SearchReactIterationInput
from sciread.agent.search_react.models import SearchReactIterationOutput
from sciread.agent.search_react.models import SearchReactIterationState
from sciread.document.models import Chunk
from sciread.document.retrieval.models import RetrievedChunk


class _FakeRunResult:
    def __init__(self, output) -> None:
        self.output = output


@pytest.mark.asyncio
async def test_search_react_tools_guard_against_repeated_search_and_memory() -> None:
    """Each iteration should allow one retrieval action and one memory write."""

    class DummyTree:
        def render(self, path=None, depth=2) -> str:
            return "tree"

    class DummyDocument:
        def retrieve_chunks(self, **kwargs):
            chunk = Chunk(content="Body", section_path=["intro"])
            return [
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0,
                    strategy="lexical",
                    section_path=["intro"],
                    expanded_context="Body",
                )
            ]

        def build_section_tree(self):
            return DummyTree()

    deps = SearchReactDeps(
        document=DummyDocument(),
        task="Summarize the paper",
        iteration_input=SearchReactIterationInput(task="Summarize the paper"),
        current_loop=1,
        max_loops=3,
        show_progress=False,
    )
    iteration_state = SearchReactIterationState()
    ctx = SimpleNamespace(deps=deps, metadata={"iteration_state": iteration_state})

    first_search = await search_document(ctx, "main contribution", strategy="lexical")
    second_search = await search_document(ctx, "results", strategy="tree")
    first_memory = await add_memory(ctx, "- [CLAIM] Main finding.")
    second_memory = await add_memory(ctx, "- [RESULT] Another finding.")

    assert "Retrieval strategy: lexical" in first_search
    assert "已经执行过检索或结构浏览" in second_search
    assert first_memory == "记忆已记录"
    assert "已调用过 add_memory" in second_memory


@pytest.mark.asyncio
async def test_search_react_tools_enforce_order_and_final_memory_access() -> None:
    """Memory retrieval should remain final-only and tree inspection should be available."""

    class DummyTree:
        def render(self, path=None, depth=2) -> str:
            return "intro\n  - setup"

    regular_deps = SearchReactDeps(
        document=SimpleNamespace(build_section_tree=lambda: DummyTree()),
        task="Summarize the paper",
        iteration_input=SearchReactIterationInput(task="Summarize the paper"),
        current_loop=1,
        max_loops=3,
        accumulated_memory="- [CLAIM] Prior finding.",
        show_progress=False,
    )
    regular_state = SearchReactIterationState()
    regular_ctx = SimpleNamespace(deps=regular_deps, metadata={"iteration_state": regular_state})

    inspect_result = await inspect_section_tree(regular_ctx, depth=2)
    add_before_search = await add_memory(regular_ctx, "- [CLAIM] Main finding.")
    get_memory_too_early = await get_all_memory(regular_ctx)

    assert "setup" in inspect_result
    assert "请先检索" in add_before_search
    assert "仅允许在最终迭代调用" in get_memory_too_early

    final_deps = SearchReactDeps(
        document=SimpleNamespace(build_section_tree=lambda: DummyTree()),
        task="Summarize the paper",
        iteration_input=SearchReactIterationInput(task="Summarize the paper"),
        current_loop=3,
        max_loops=3,
        accumulated_memory="- [CLAIM] Prior finding.",
        show_progress=False,
    )
    final_state = SearchReactIterationState()
    final_ctx = SimpleNamespace(deps=final_deps, metadata={"iteration_state": final_state})

    first_get_memory = await get_all_memory(final_ctx)
    second_get_memory = await get_all_memory(final_ctx)

    assert "完整累计记忆" in first_get_memory
    assert "已调用过 get_all_memory" in second_get_memory


@pytest.mark.asyncio
async def test_search_react_compare_runs_strategies_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare mode should return one result per requested strategy."""
    expected_runs = []

    async def fake_run_analysis(self, **kwargs):
        strategy = kwargs["strategy"]
        run = SimpleNamespace(
            strategy=strategy,
            output=SearchReactIterationOutput(thoughts=f"{strategy} done", should_continue=False, report=f"{strategy} report"),
            retrieved_chunks=[],
            total_time_seconds=0.1,
            error="",
        )
        expected_runs.append(strategy)
        return run

    monkeypatch.setattr("sciread.agent.search_react.agent.load_document", lambda *args, **kwargs: SimpleNamespace(chunks=[], metadata=None))
    monkeypatch.setattr("sciread.agent.search_react.agent._validate_document_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("sciread.agent.search_react.agent._render_analysis_overview", lambda *args, **kwargs: None)
    monkeypatch.setattr("sciread.agent.search_react.agent.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(SearchReactAgent, "run_analysis", fake_run_analysis)

    result = await analyze_file_with_search_react(
        "paper.pdf",
        "What changed?",
        compare=["lexical", "tree"],
        show_progress=False,
    )

    assert [run.strategy for run in result.runs] == ["lexical", "tree"]
    assert expected_runs == ["lexical", "tree"]


@pytest.mark.asyncio
async def test_search_react_run_iteration_falls_back_when_agent_run_raises() -> None:
    """Iteration execution should use the safe fallback when the retrieval model call fails."""

    class FailingAgent:
        async def run(self, *args, **kwargs):
            raise RuntimeError("model offline")

    agent = SearchReactAgent.__new__(SearchReactAgent)
    agent.logger = SimpleNamespace(error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None)
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.agent = FailingAgent()

    output, iteration_state = await agent.run_iteration(
        document=SimpleNamespace(),
        iteration_input=SearchReactIterationInput(task="Summarize"),
        current_loop=1,
        max_loops=3,
        accumulated_memory="",
        show_progress=False,
    )

    assert output.thoughts == "Iteration failed: could not complete retrieval-driven analysis."
    assert output.should_continue is True
    assert iteration_state.queries_run == []


@pytest.mark.asyncio
async def test_search_react_compare_keeps_failures_without_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare mode should keep failed strategies in the result while returning successful ones."""
    monkeypatch.setattr("sciread.agent.search_react.agent.load_document", lambda *args, **kwargs: SimpleNamespace(chunks=[], metadata=None))
    monkeypatch.setattr("sciread.agent.search_react.agent._validate_document_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("sciread.agent.search_react.agent._render_analysis_overview", lambda *args, **kwargs: None)
    monkeypatch.setattr("sciread.agent.search_react.agent._render_compare_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("sciread.agent.search_react.agent.console.print", lambda *args, **kwargs: None)

    async def fake_run_analysis(self, **kwargs):
        strategy = kwargs["strategy"]
        if strategy == "tree":
            raise RuntimeError("tree failed")
        return SimpleNamespace(
            strategy=strategy,
            output=SearchReactIterationOutput(thoughts=f"{strategy} done", should_continue=False, report=f"{strategy} report"),
            retrieved_chunks=[],
            total_time_seconds=0.1,
            error="",
        )

    monkeypatch.setattr(SearchReactAgent, "run_analysis", fake_run_analysis)

    result = await analyze_file_with_search_react(
        "paper.pdf",
        "What changed?",
        compare=["lexical", "tree"],
        show_progress=False,
    )

    assert [run.strategy for run in result.runs] == ["lexical", "tree"]
    assert result.runs[0].error == ""
    assert result.runs[0].output.report == "lexical report"
    assert result.runs[1].error == "tree failed"
    assert result.runs[1].output.thoughts == "Strategy failed: tree failed"


@pytest.mark.asyncio
async def test_search_react_run_analysis_stops_after_completed_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The multi-iteration loop should stop once one iteration reports completion."""
    agent = SearchReactAgent.__new__(SearchReactAgent)
    agent.logger = SimpleNamespace(error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None)
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.agent = object()

    chunk = Chunk(content="Body", chunk_name="intro")
    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.9,
        strategy="hybrid",
        section_path=["intro"],
        expanded_context="Body",
    )

    async def fake_run_iteration(
        document,
        iteration_input: SearchReactIterationInput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
        show_progress: bool,
    ) -> tuple[SearchReactIterationOutput, SearchReactIterationState]:
        assert current_loop == 1
        return (
            SearchReactIterationOutput(thoughts="Enough evidence.", should_continue=False, report=""),
            SearchReactIterationState(
                queries_run=["main contribution"],
                retrieved_chunks=[retrieved],
                memory_text="- [CLAIM] Main contribution found.",
            ),
        )

    monkeypatch.setattr(agent, "run_iteration", fake_run_iteration)

    result = await agent.run_analysis(
        document=SimpleNamespace(),
        task="Summarize",
        strategy="hybrid",
        max_loops=3,
        show_progress=False,
    )

    assert result.strategy == "hybrid"
    assert result.output.should_continue is False
    assert result.output.report == "- [CLAIM] Main contribution found."
    assert [item.chunk.chunk_id for item in result.retrieved_chunks] == [chunk.chunk_id]
