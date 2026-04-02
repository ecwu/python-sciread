"""ReAct agent for intelligent document analysis with single-iteration loops."""

import asyncio
import json
import traceback
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import RunContext
from rich.console import Console
from rich.markdown import Markdown

from ...document_structure import Document
from ...document_structure.renderers import get_sections_content
from ...llm_provider import get_model
from ...platform.logging import get_logger
from .models import ReActAnalysisState
from .models import ReActIterationDeps
from .models import ReActIterationInput
from .models import ReActIterationOutput
from .models import ReActIterationState
from .prompts import build_iteration_system_prompt
from .prompts import build_iteration_user_prompt

logger = get_logger(__name__)

console = Console()


def _get_iteration_state(ctx: RunContext[ReActIterationDeps]) -> ReActIterationState:
    """Retrieve per-iteration mutable state from agent metadata."""
    metadata = ctx.metadata or {}
    state = metadata.get("iteration_state")
    if not isinstance(state, ReActIterationState):
        raise RuntimeError("ReAct iteration state is missing from run metadata.")
    return state


def _validate_document_file(document_file: str | Path) -> None:
    """Validate that the document file exists before processing."""
    if not Path(document_file).exists():
        raise FileNotFoundError(f"Document file not found: {document_file}")


def _clean_section_names(section_names: list[object]) -> list[str] | None:
    """Normalize a mixed list into a deduplicated section-name list."""
    cleaned_names: list[str] = []
    for name in section_names:
        if not isinstance(name, str):
            continue

        normalized_name = name.strip()
        if normalized_name and normalized_name not in cleaned_names:
            cleaned_names.append(normalized_name)

    return cleaned_names or None


def _resolve_sections(document: Document, requested_sections: list[str], available_sections: list[str]) -> list[str]:
    """Resolve requested section names against exact and fuzzy document matches."""
    resolved_sections: list[str] = []
    for section in requested_sections:
        resolved_section = section if section in available_sections else document.get_closest_section_name(section, threshold=0.7)
        if resolved_section and resolved_section not in resolved_sections:
            resolved_sections.append(resolved_section)

    return resolved_sections


def load_and_process_document(file_path: str | Path, to_markdown: bool = True) -> Document:
    """Load and process a document using markdown conversion and natural section splitting.

    Args:
        file_path: Path to the PDF file
        to_markdown: Whether to convert PDF to markdown using Mineru API

    Returns:
        Document instance with processed chunks using natural markdown sections
    """
    logger.debug(f"Loading document from {file_path} (to_markdown={to_markdown})")

    # Create document with markdown conversion and auto-splitting
    # Document.from_file() automatically loads and splits the document when auto_split=True
    document = Document.from_file(file_path, to_markdown=to_markdown, auto_split=True)

    logger.debug(f"Document processed into {len(document.chunks)} chunks with natural markdown sections")
    logger.debug(f"Available sections: {document.get_section_names()}")

    return document


def get_section_content(document: Document, section_names: list[str]) -> str:
    """Get content for specified sections.

    Args:
        document: Processed document instance
        section_names: List of section names to retrieve content for

    Returns:
        Combined content from all specified sections
    """
    if not section_names:
        return ""

    sections = get_sections_content(
        document,
        section_names=section_names,
        clean_text=True,
    )
    if not sections:
        logger.warning(f"No content found for sections: {section_names}")
        return ""

    combined_content = "\n\n".join(f"=== {name.upper()} ===\n{content}" for name, content in sections)
    logger.debug(f"Retrieved content for sections {section_names}: {len(combined_content)} characters")

    return combined_content


def normalize_section_names(section_names: list[str] | str | None) -> list[str] | None:
    """Normalize tool input to a list of section names.

    Some models send a JSON-encoded list as a string. This parser keeps the
    tool callable instead of failing at schema validation time.
    """
    if section_names is None:
        return None

    if isinstance(section_names, list):
        return _clean_section_names(section_names)

    if isinstance(section_names, str):
        raw = section_names.strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return _clean_section_names(parsed)
            if isinstance(parsed, str):
                return _clean_section_names([parsed])
        except json.JSONDecodeError:
            # Fallback: treat as a single section name.
            return [raw]

    return None


async def analyze_file_with_react(
    file_path: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> ReActIterationOutput:
    """Analyze a file using the ReAct agent with multi-iteration loops.

    Args:
        file_path: Path to the document file (PDF or TXT)
        task: Analysis task or question about the document
        model: Model identifier for the LLM provider
        max_loops: Maximum number of analysis iterations
        to_markdown: Whether to convert PDF to markdown using Mineru API
        show_progress: Whether to show progress during analysis

    Returns:
        Structured analysis report generated by the ReAct agent

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the analysis fails
    """
    logger.debug(f"Starting ReAct analysis for file: {file_path}")
    logger.debug(f"Task: {task[:100]}...")
    logger.debug(f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}")

    _validate_document_file(file_path)

    document = load_and_process_document(file_path, to_markdown=to_markdown)

    agent = ReActAgent(model=model)
    result = await agent.run_analysis(document, task, max_loops=max_loops, show_progress=show_progress)

    console.print(Markdown(result.report))

    logger.debug("ReAct analysis completed successfully!")
    return result


def analyze_file_with_react_sync(
    file_path: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> ReActIterationOutput:
    """Synchronous wrapper for top-level callers such as the CLI."""
    return asyncio.run(
        analyze_file_with_react(
            file_path=file_path,
            task=task,
            model=model,
            max_loops=max_loops,
            to_markdown=to_markdown,
            show_progress=show_progress,
        )
    )


react_iteration_agent = Agent(
    deps_type=ReActIterationDeps,
    output_type=ReActIterationOutput,
    retries=3,
)


@react_iteration_agent.system_prompt
async def react_iteration_system_prompt(
    ctx: RunContext[ReActIterationDeps],
) -> str:
    return build_iteration_system_prompt(ctx.deps)


@react_iteration_agent.tool
async def read_section(ctx: RunContext[ReActIterationDeps], section_names: list[str] | str | None = None) -> str:
    """Read one or more sections.

    Args:
        section_names: Section names to read (list or comma-separated string).
                      If None, reads the first available section.
    """
    deps = ctx.deps
    iteration_state = _get_iteration_state(ctx)
    iteration_input = deps.iteration_input
    available_sections = iteration_input.available_sections

    if not available_sections:
        return "提示：所有章节都已处理，没有可读取内容。"

    normalized_section_names = normalize_section_names(section_names)
    requested_sections = normalized_section_names or [available_sections[0]]
    resolved_sections = _resolve_sections(deps.document, requested_sections, available_sections)

    next_sections = resolved_sections
    if not next_sections:
        return f"提示：未能匹配到请求章节。可用章节：{available_sections}"

    section_content = get_section_content(deps.document, next_sections)
    if not section_content.strip():
        return f"警告：在章节 {next_sections} 中未找到内容，请尝试其他章节。"

    iteration_state.sections_read.extend(next_sections)

    if deps.show_progress:
        print(f"\n[Iteration] Read sections: {', '.join(next_sections)}")

    return f"章节内容（读取自：{', '.join(next_sections)}）：\n\n{section_content}\n\n现在请调用 add_memory() 记录提取到的发现。"


@react_iteration_agent.tool
async def add_memory(ctx: RunContext[ReActIterationDeps], memory: str) -> str:
    """Record discovered memory from the current read.

    Args:
        memory: Extracted findings and key information from newly read sections.
    """
    iteration_state = _get_iteration_state(ctx)

    fragment = memory.strip()
    if not fragment:
        return "警告：记忆文本为空。"

    iteration_state.memory_text = f"{iteration_state.memory_text}\n\n{fragment}".strip() if iteration_state.memory_text else fragment

    return "记忆已记录"


@react_iteration_agent.tool
async def get_all_memory(ctx: RunContext[ReActIterationDeps]) -> str:
    """Read the complete accumulated memory from all previous iterations.

    Use this ONLY on the final iteration when preparing the complete structured analysis.
    Returns the full synthesized memory built up across all iterations.
    """
    deps = ctx.deps

    if not deps.accumulated_memory.strip():
        return "提示：目前还没有来自前几轮的累计记忆。"

    return f"完整累计记忆：\n\n{deps.accumulated_memory}\n\n请基于这份完整记忆进行最终推理并生成最终报告。"


class ReActAgent:
    """ReAct agent for intelligent document analysis with single-iteration loops.

    This agent performs one read + one memory update per iteration, enabling
    controlled iterative analysis with reduced context accumulation.
    """

    def __init__(self, model: str = "deepseek-chat"):
        """Initialize the ReAct agent.

        Args:
            model: Model identifier for the LLM provider
        """
        self.logger = get_logger(__name__)
        self.model = get_model(model)
        self.model_identifier = model
        self.agent = react_iteration_agent

        self.logger.info(f"Initialized ReActAgent with model: {model}")

    @staticmethod
    def _build_iteration_deps(
        document: Document,
        iteration_input: ReActIterationInput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
        show_progress: bool,
    ) -> ReActIterationDeps:
        """Create the dependency bundle for one iteration."""
        return ReActIterationDeps(
            document=document,
            task=iteration_input.task,
            iteration_input=iteration_input,
            current_loop=current_loop,
            max_loops=max_loops,
            accumulated_memory=accumulated_memory,
            show_progress=show_progress,
        )

    def _fallback_iteration_output(self) -> ReActIterationOutput:
        """Create a safe fallback output when the model call fails."""
        return ReActIterationOutput(
            thoughts="Iteration failed: could not complete analysis.",
            should_continue=True,
        )

    def _finalize_iteration_output(
        self,
        output: ReActIterationOutput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
    ) -> ReActIterationOutput:
        """Force convergence once the maximum loop count is reached."""
        if current_loop < max_loops or not output.should_continue:
            return output

        self.logger.info("Final loop reached with should_continue=True; forcing should_continue=False")
        output.should_continue = False
        if not output.report.strip() and accumulated_memory.strip():
            output.report = accumulated_memory.strip()

        return output

    def _log_iteration_exception(self, error: Exception) -> None:
        """Log iteration failures with traceback context."""
        self.logger.error(f"Iteration failed: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def run_iteration(
        self,
        document: Document,
        iteration_input: ReActIterationInput,
        current_loop: int = 1,
        max_loops: int = 8,
        accumulated_memory: str = "",
        show_progress: bool = True,
    ) -> tuple[ReActIterationOutput, ReActIterationState]:
        """Run a single ReAct iteration and return its output plus tool state.

        Args:
            document: The document to analyze
            iteration_input: Input state from previous iteration(s)
            current_loop: Current iteration number (1-based)
            max_loops: Maximum number of iterations allowed
            accumulated_memory: Complete accumulated memory from all previous iterations
            show_progress: Whether to print debug progress

        Returns:
            Tuple of iteration output and internal iteration state
        """
        self.logger.debug("Starting single ReAct iteration")

        deps = self._build_iteration_deps(
            document=document,
            iteration_input=iteration_input,
            current_loop=current_loop,
            max_loops=max_loops,
            accumulated_memory=accumulated_memory,
            show_progress=show_progress,
        )
        iteration_state = ReActIterationState()

        try:
            result = await self.agent.run(
                build_iteration_user_prompt(current_loop, max_loops),
                deps=deps,
                model=self.model,
                metadata={"iteration_state": iteration_state},
            )
            output = result.output if isinstance(result.output, ReActIterationOutput) else None
        except (RuntimeError, ValueError, TypeError) as e:
            self._log_iteration_exception(e)
            output = None

        if output is None:
            output = self._fallback_iteration_output()

        return self._finalize_iteration_output(output, current_loop, max_loops, accumulated_memory), iteration_state

    async def run_analysis(
        self,
        document: Document,
        task: str,
        max_loops: int = 8,
        show_progress: bool = True,
    ) -> ReActIterationOutput:
        """Run the multi-iteration ReAct analysis over a prepared document.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            max_loops: Maximum number of iterations to perform
            show_progress: Whether to print progress information

        Returns:
            Final ReAct iteration output with should_continue=False and final report populated
        """
        self.logger.debug(f"Starting ReAct multi-iteration analysis (max_loops={max_loops})")

        analysis_state = ReActAnalysisState(
            task=task,
            available_sections=document.get_section_names(),
        )

        for loop_num in range(1, max_loops + 1):
            self.logger.debug(f"Starting iteration {loop_num}/{max_loops}")

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
                console.print(Markdown(analysis_state.current_thoughts))

            if not iteration_output.should_continue:
                self.logger.info(f"Agent decided to stop after iteration {loop_num}")
                if show_progress:
                    print("Agent completed analysis")
                break

            if not analysis_state.remaining_sections:
                self.logger.info(f"All sections processed after iteration {loop_num}")
                if show_progress:
                    print("All available sections have been read")
                break

        self.logger.info(f"Multi-iteration analysis completed after {loop_num} iterations")
        return analysis_state.build_final_output()

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier})"
