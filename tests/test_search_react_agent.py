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
