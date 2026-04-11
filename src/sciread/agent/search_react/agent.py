"""Search-react agent with multi-strategy retrieval tools."""

from __future__ import annotations

import asyncio
import time
import traceback
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import RunContext
from rich.console import Console

from ...application.use_cases.common import ensure_file_exists
from ...application.use_cases.common import load_document
from ...document import Document
from ...document.retrieval import SUPPORTED_RETRIEVERS
from ...document.retrieval import format_retrieval_results
from ...llm_provider import get_model
from ...platform.logging import get_logger
from ...platform.rich_output import TableColumnSpec
from ...platform.rich_output import build_data_table
from ...platform.rich_output import build_key_value_table
from ...platform.rich_output import build_markdown_panel
from ...platform.rich_output import build_mode_banner
from ...platform.rich_output import build_stage_banner
from .models import SearchReactAnalysisResult
from .models import SearchReactAnalysisState
from .models import SearchReactDeps
from .models import SearchReactIterationInput
from .models import SearchReactIterationOutput
from .models import SearchReactIterationState
from .models import SearchReactStrategyRun
from .prompts import build_iteration_system_prompt
from .prompts import build_iteration_user_prompt

logger = get_logger(__name__)
console = Console()


def _render_iteration_header(current_loop: int, max_loops: int, processed_queries: int, retrieved_chunks: int, strategy: str) -> None:
    """Render a consistent iteration header."""
    console.print(
        build_stage_banner(
            title=f"Search-ReAct Iteration {current_loop}/{max_loops}",
            summary_lines=[
                f"Strategy: {strategy}",
                f"Queries executed: {processed_queries}",
                f"Retrieved chunks: {retrieved_chunks}",
            ],
            border_style="cyan",
        )
    )


def _render_retrieval_summary(strategy: str, results) -> None:
    """Render one retrieval summary table."""
    rows = [
        (
            str(index),
            result.section_path_text or "unknown",
            result.chunk.citation_key,
            f"{result.score:.3f}",
        )
        for index, result in enumerate(results, start=1)
    ]
    console.print(
        build_data_table(
            title=f"Retrieved Chunks ({strategy})",
            columns=[
                TableColumnSpec("#", style="green", justify="right", no_wrap=True, width=4),
                TableColumnSpec("Section", style="yellow"),
                TableColumnSpec("Citation", style="cyan", no_wrap=True),
                TableColumnSpec("Score", style="green", justify="right", no_wrap=True, width=8),
            ],
            rows=rows or [("-", "No hits", "-", "-")],
        )
    )


def _render_compare_summary(result: SearchReactAnalysisResult) -> None:
    """Render compare-mode strategy summary."""
    rows = []
    for run in result.runs:
        rows.append(
            (
                run.strategy,
                "error" if run.error else "ok",
                str(len(run.retrieved_chunks)),
                f"{run.total_time_seconds:.2f}s",
            )
        )
    console.print(
        build_data_table(
            title="Strategy Comparison",
            columns=[
                TableColumnSpec("Strategy", style="cyan", no_wrap=True),
                TableColumnSpec("Status", style="yellow", no_wrap=True),
                TableColumnSpec("Retrieved", style="green", justify="right", no_wrap=True),
                TableColumnSpec("Time", style="green", justify="right", no_wrap=True),
            ],
            rows=rows,
        )
    )


def _get_iteration_state(ctx: RunContext[SearchReactDeps]) -> SearchReactIterationState:
    """Retrieve per-iteration mutable state from agent metadata."""
    metadata = ctx.metadata or {}
    state = metadata.get("iteration_state")
    if not isinstance(state, SearchReactIterationState):
        raise RuntimeError("Search-react iteration state is missing from run metadata.")
    return state


def _normalize_strategy(strategy: str | None, default: str) -> str:
    """Normalize a retrieval strategy name."""
    candidate = (strategy or default).strip().lower()
    if candidate not in SUPPORTED_RETRIEVERS:
        raise ValueError(f"Unsupported retrieval strategy: {candidate}")
    return candidate


search_react_iteration_agent = Agent(
    deps_type=SearchReactDeps,
    output_type=SearchReactIterationOutput,
    retries=3,
)


@search_react_iteration_agent.system_prompt
async def search_react_iteration_system_prompt(ctx: RunContext[SearchReactDeps]) -> str:
    return build_iteration_system_prompt(ctx.deps)


@search_react_iteration_agent.tool
async def search_document(
    ctx: RunContext[SearchReactDeps],
    query: str,
    strategy: str | None = None,
    top_k: int | None = None,
    neighbor_window: int | None = None,
    section_scope: str | None = None,
) -> str:
    """Search the document using the configured retrieval strategies."""
    deps = ctx.deps
    iteration_state = _get_iteration_state(ctx)

    if iteration_state.queries_run or iteration_state.tree_inspected:
        return "提示：本轮已经执行过检索或结构浏览。请调用 add_memory() 或直接返回迭代输出。"

    if iteration_state.memory_text.strip():
        return "提示：本轮已调用过 add_memory()。请直接返回 SearchReactIterationOutput。"

    active_strategy = _normalize_strategy(strategy, deps.iteration_input.strategy)
    active_top_k = top_k or deps.iteration_input.top_k
    active_neighbor_window = neighbor_window if neighbor_window is not None else deps.iteration_input.neighbor_window

    results = deps.document.retrieve_chunks(
        query=query,
        strategy=active_strategy,
        top_k=active_top_k,
        neighbor_window=active_neighbor_window,
        section_scope=section_scope,
    )

    iteration_state.queries_run.append(query)
    iteration_state.retrieved_chunks = results

    if deps.show_progress:
        _render_retrieval_summary(active_strategy, results)

    return format_retrieval_results(results, query=query, strategy=active_strategy)


@search_react_iteration_agent.tool
async def inspect_section_tree(
    ctx: RunContext[SearchReactDeps],
    path: str | None = None,
    depth: int = 2,
) -> str:
    """Inspect the current section tree for debugging and query planning."""
    iteration_state = _get_iteration_state(ctx)
    if iteration_state.queries_run or iteration_state.tree_inspected:
        return "提示：本轮已经执行过结构浏览或检索。请直接返回迭代输出，或在下一轮继续。"
    if iteration_state.memory_text.strip():
        return "提示：本轮已调用过 add_memory()。请直接返回 SearchReactIterationOutput。"

    iteration_state.tree_inspected = True
    section_tree = ctx.deps.document.build_section_tree()
    return section_tree.render(path=path, depth=depth)


@search_react_iteration_agent.tool
async def add_memory(ctx: RunContext[SearchReactDeps], memory: str) -> str:
    """Record memory from the most recent retrieval result."""
    iteration_state = _get_iteration_state(ctx)
    if not iteration_state.retrieved_chunks:
        return "提示：本轮还没有执行 search_document()。请先检索，再记录记忆。"
    if iteration_state.memory_text.strip():
        return "提示：本轮已调用过 add_memory()。请直接返回 SearchReactIterationOutput。"

    fragment = memory.strip()
    if not fragment:
        return "警告：记忆文本为空。"

    iteration_state.memory_text = fragment
    return "记忆已记录"


@search_react_iteration_agent.tool
async def get_all_memory(ctx: RunContext[SearchReactDeps]) -> str:
    """Return the complete accumulated memory from all earlier iterations."""
    deps = ctx.deps
    iteration_state = _get_iteration_state(ctx)

    if deps.current_loop < deps.max_loops:
        return "提示：get_all_memory() 仅允许在最终迭代调用。"
    if iteration_state.all_memory_read:
        return "提示：本轮已调用过 get_all_memory()。请直接返回最终 SearchReactIterationOutput。"

    iteration_state.all_memory_read = True
    if not deps.accumulated_memory.strip():
        return "提示：目前还没有累计记忆。"
    return f"完整累计记忆：\n\n{deps.accumulated_memory}\n\n请基于这份记忆生成最终报告。"


class SearchReactAgent:
    """ReAct-style agent that reads retrieval bundles instead of raw sections."""

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.logger = get_logger(__name__)
        self.model_identifier = model
        self.model = None
        self.agent = search_react_iteration_agent
        self.logger.info(f"Initialized SearchReactAgent with model: {model}")

    def _get_or_create_model(self):
        """Create the configured model lazily so tests can replace run paths without real provider setup."""
        if self.model is None:
            self.model = get_model(self.model_identifier)
        return self.model

    def _build_iteration_deps(
        self,
        document: Document,
        iteration_input: SearchReactIterationInput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
        show_progress: bool,
    ) -> SearchReactDeps:
        """Create the dependency bundle for one iteration."""
        return SearchReactDeps(
            document=document,
            task=iteration_input.task,
            iteration_input=iteration_input,
            current_loop=current_loop,
            max_loops=max_loops,
            accumulated_memory=accumulated_memory,
            show_progress=show_progress,
        )

    def _fallback_iteration_output(self) -> SearchReactIterationOutput:
        """Create a safe fallback output when the model call fails."""
        return SearchReactIterationOutput(
            thoughts="Iteration failed: could not complete retrieval-driven analysis.",
            should_continue=True,
        )

    def _finalize_iteration_output(
        self,
        output: SearchReactIterationOutput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
    ) -> SearchReactIterationOutput:
        """Force convergence on the final loop."""
        if current_loop < max_loops or not output.should_continue:
            return output

        output.should_continue = False
        if not output.report.strip() and accumulated_memory.strip():
            output.report = accumulated_memory.strip()
        return output

    def _log_iteration_exception(self, error: Exception) -> None:
        """Log one iteration failure."""
        self.logger.error(f"Search-react iteration failed: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def run_iteration(
        self,
        document: Document,
        iteration_input: SearchReactIterationInput,
        current_loop: int = 1,
        max_loops: int = 5,
        accumulated_memory: str = "",
        show_progress: bool = True,
    ) -> tuple[SearchReactIterationOutput, SearchReactIterationState]:
        """Run a single search-react iteration."""
        deps = self._build_iteration_deps(
            document=document,
            iteration_input=iteration_input,
            current_loop=current_loop,
            max_loops=max_loops,
            accumulated_memory=accumulated_memory,
            show_progress=show_progress,
        )
        iteration_state = SearchReactIterationState()

        try:
            result = await self.agent.run(
                build_iteration_user_prompt(current_loop, max_loops),
                deps=deps,
                model=self._get_or_create_model(),
                metadata={"iteration_state": iteration_state},
            )
            output = result.output if isinstance(result.output, SearchReactIterationOutput) else None
        except (RuntimeError, ValueError, TypeError) as exc:
            self._log_iteration_exception(exc)
            output = None

        if output is None:
            output = self._fallback_iteration_output()

        return (
            self._finalize_iteration_output(output, current_loop, max_loops, accumulated_memory),
            iteration_state,
        )

    async def run_analysis(
        self,
        document: Document,
        task: str,
        *,
        strategy: str = "hybrid",
        top_k: int = 5,
        neighbor_window: int = 1,
        max_loops: int = 5,
        show_progress: bool = True,
    ) -> SearchReactStrategyRun:
        """Run the multi-iteration analysis for a single retrieval strategy."""
        analysis_state = SearchReactAnalysisState(
            task=task,
            strategy=strategy,
            top_k=top_k,
            neighbor_window=neighbor_window,
        )
        started_at = time.perf_counter()

        for loop_num in range(1, max_loops + 1):
            if show_progress:
                _render_iteration_header(
                    current_loop=loop_num,
                    max_loops=max_loops,
                    processed_queries=len(analysis_state.processed_queries),
                    retrieved_chunks=len(analysis_state.retrieved_chunks),
                    strategy=strategy,
                )

            iteration_output, iteration_state = await self.run_iteration(
                document=document,
                iteration_input=analysis_state.build_iteration_input(),
                current_loop=loop_num,
                max_loops=max_loops,
                accumulated_memory=analysis_state.accumulated_memory,
                show_progress=show_progress,
            )
            analysis_state.apply_iteration(iteration_output, iteration_state)

            if show_progress:
                console.print(
                    build_markdown_panel(
                        title=f"Iteration Notes {loop_num}/{max_loops}",
                        content=analysis_state.current_thoughts,
                        border_style="blue",
                        subtitle=f"Queries: {len(analysis_state.processed_queries)} | Retrieved chunks: {len(analysis_state.retrieved_chunks)}",
                    )
                )

            if not iteration_output.should_continue:
                break

        finished_at = time.perf_counter()
        output = analysis_state.build_final_output()
        return SearchReactStrategyRun(
            strategy=strategy,
            output=output,
            retrieved_chunks=analysis_state.retrieved_chunks,
            total_time_seconds=finished_at - started_at,
        )


def _validate_document_file(document_file: str | Path) -> None:
    """Validate that the input file exists."""
    ensure_file_exists(str(document_file))


def _render_analysis_overview(document: Document, task: str, strategy: str, max_loops: int, top_k: int, neighbor_window: int) -> None:
    """Render the top-level search-react analysis overview."""
    metadata = getattr(document, "metadata", None)
    document_title = getattr(metadata, "title", None) or getattr(document, "source_path", None) or "unknown"
    console.print()
    console.print(
        build_mode_banner(
            "Search-ReAct Analysis",
            subtitle="Retrieval-driven iterative analysis with comparable strategies",
        )
    )
    console.print(
        build_key_value_table(
            "Analysis Overview",
            [
                ("Document", str(document_title)),
                ("Task", task),
                ("Default Strategy", strategy),
                ("Chunks", str(len(document.chunks))),
                ("Max Loops", str(max_loops)),
                ("Top K", str(top_k)),
                ("Neighbor Window", str(neighbor_window)),
            ],
        )
    )


async def analyze_file_with_search_react(
    file_path: str,
    task: str,
    *,
    model: str = "deepseek-chat",
    max_loops: int = 5,
    to_markdown: bool = True,
    show_progress: bool = True,
    retriever: str = "hybrid",
    compare: list[str] | None = None,
    top_k: int = 5,
    neighbor_window: int = 1,
) -> SearchReactAnalysisResult:
    """Analyze one file with search-react, optionally comparing multiple retrievers."""
    _validate_document_file(file_path)
    document = load_document(str(file_path), to_markdown=to_markdown)
    _render_analysis_overview(document, task, retriever, max_loops, top_k, neighbor_window)

    strategies = compare or [retriever]
    runs: list[SearchReactStrategyRun] = []
    for strategy in strategies:
        agent = SearchReactAgent(model=model)
        try:
            run = await agent.run_analysis(
                document=document,
                task=task,
                strategy=_normalize_strategy(strategy, retriever),
                top_k=top_k,
                neighbor_window=neighbor_window,
                max_loops=max_loops,
                show_progress=show_progress,
            )
        except Exception as exc:
            if not compare:
                raise
            runs.append(
                SearchReactStrategyRun(
                    strategy=strategy,
                    output=SearchReactIterationOutput(thoughts=f"Strategy failed: {exc}", should_continue=False, report=""),
                    retrieved_chunks=[],
                    total_time_seconds=0.0,
                    error=str(exc),
                )
            )
            continue

        runs.append(run)
        console.print()
        console.print(build_markdown_panel(f"Final Report ({strategy})", run.output.report, border_style="green"))

    if compare:
        _render_compare_summary(SearchReactAnalysisResult(task=task, primary_strategy=retriever, runs=runs))

    if not runs:
        raise RuntimeError("No search-react strategy completed successfully.")

    return SearchReactAnalysisResult(
        task=task,
        primary_strategy=retriever,
        runs=runs,
    )


def analyze_file_with_search_react_sync(
    file_path: str,
    task: str,
    *,
    model: str = "deepseek-chat",
    max_loops: int = 5,
    to_markdown: bool = True,
    show_progress: bool = True,
    retriever: str = "hybrid",
    compare: list[str] | None = None,
    top_k: int = 5,
    neighbor_window: int = 1,
) -> SearchReactAnalysisResult:
    """Synchronous wrapper for CLI callers."""
    return asyncio.run(
        analyze_file_with_search_react(
            file_path=file_path,
            task=task,
            model=model,
            max_loops=max_loops,
            to_markdown=to_markdown,
            show_progress=show_progress,
            retriever=retriever,
            compare=compare,
            top_k=top_k,
            neighbor_window=neighbor_window,
        )
    )
