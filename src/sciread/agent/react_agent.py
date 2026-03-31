"""ReAct agent for intelligent document analysis with single-iteration loops."""

import asyncio
import json
import traceback
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import RunContext
from rich.console import Console
from rich.markdown import Markdown

from ..document import Document
from ..document.document_renderers import get_sections_content
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.react_models import ReActIterationInput
from .models.react_models import ReActIterationOutput

logger = get_logger(__name__)

console = Console()


@dataclass
class ReActIterationDeps:
    """Immutable dependencies for a single ReAct iteration."""

    document: Document
    task: str
    iteration_input: ReActIterationInput
    current_loop: int = 1
    max_loops: int = 8
    accumulated_memory: str = ""
    show_progress: bool = True


@dataclass
class ReActIterationState:
    """Mutable per-iteration state: track tool invocations to enforce single-call rules."""

    read_section_called: bool = False
    add_memory_called: bool = False
    sections_read: list[str] = field(default_factory=list)
    section_content: str = ""
    memory_text: str = ""


def _get_iteration_state(ctx: RunContext[ReActIterationDeps]) -> ReActIterationState:
    """Retrieve per-iteration mutable state from agent metadata."""
    metadata = ctx.metadata or {}
    state = metadata.get("iteration_state")
    if not isinstance(state, ReActIterationState):
        raise RuntimeError("ReAct iteration state is missing from run metadata.")
    return state


# ReAct agent implementation


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

    content_parts = []
    for name, content in sections:
        section_name = name if name != "unknown" else "unknown"
        content_parts.append(f"=== {section_name.upper()} ===\n{content}")

    combined_content = "\n\n".join(content_parts)
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
        return [name for name in section_names if isinstance(name, str) and name.strip()]

    if isinstance(section_names, str):
        raw = section_names.strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [name for name in parsed if isinstance(name, str) and name.strip()]
        except json.JSONDecodeError:
            # Fallback: treat as a single section name.
            return [raw]

    return None


async def analyze_document_with_react(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> ReActIterationOutput:
    """Analyze a document using the ReAct agent with multi-iteration loops.

    Args:
        document_file: Path to the document file (PDF or TXT)
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
    logger.debug(f"Starting ReAct analysis for file: {document_file}")
    logger.debug(f"Task: {task[:100]}...")
    logger.debug(f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}")

    # Check if file exists
    if not Path(document_file).exists():
        raise FileNotFoundError(f"Document file not found: {document_file}")

    # Load and process the document
    document = load_and_process_document(document_file, to_markdown=to_markdown)

    # Create and run the ReAct agent
    agent = ReActAgent(model=model)
    result = await agent.analyze_document(document, task, max_loops=max_loops, show_progress=show_progress)

    console.print(Markdown(result.report))

    logger.debug("ReAct analysis completed successfully!")
    return result


def analyze_document_with_react_sync(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> ReActIterationOutput:
    """Synchronous wrapper for top-level callers such as the CLI."""
    return asyncio.run(
        analyze_document_with_react(
            document_file=document_file,
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
    deps = ctx.deps
    iteration_input = deps.iteration_input
    remaining_loops = max(deps.max_loops - deps.current_loop, 0)
    is_final_iteration = deps.current_loop >= deps.max_loops
    is_first_iteration = deps.current_loop == 1

    previous_context = f"上一轮思考：\n{iteration_input.previous_thoughts}" if iteration_input.previous_thoughts else ""
    processed_sections_str = (
        f"已处理章节：{', '.join(iteration_input.processed_sections)}" if iteration_input.processed_sections else "尚未处理任何章节。"
    )
    unprocessed = [s for s in iteration_input.available_sections if s not in iteration_input.processed_sections]
    unprocessed_str = f"剩余未处理章节：{', '.join(unprocessed)}" if unprocessed else "所有章节都已阅读。"

    # ── FINAL ITERATION: synthesis only, no new reading ──────────────────────
    if is_final_iteration:
        return (
            f"任务：{deps.task}\n\n"
            f"=== 最终迭代（{deps.current_loop}/{deps.max_loops}）——仅做综合 ===\n\n"
            f"{processed_sections_str}\n\n"
            f"{previous_context}\n\n"
            f"本轮严格规则：\n"
            f"  ✗ 不要调用 read_section() —— 所有阅读已结束。\n"
            f"  ✓ 第 1 步：调用 get_all_memory()，获取到目前为止累计的全部发现。\n"
            f"  ✓ 第 2 步：基于这些发现综合生成最终结构化报告（见下方格式）。\n"
            f"  ✓ 第 3 步：返回 ReActIterationOutput，并设置 should_continue=False 且填写 report。\n\n"
            f"=== 最终报告格式 ===\n"
            f"撰写简洁、聚焦贡献的报告。不要按章节逐段复述。\n"
            f"请按以下结构输出：\n\n"
            f"1. **核心研究问题与主张**\n"
            f"   论文试图填补什么空白？核心论断是什么？\n\n"
            f"2. **关键贡献**（最重要部分）\n"
            f"   列出 3-5 条具体且明确的贡献。\n"
            f"   语言要精确：指出方法、数据集、指标、性能提升等要点。\n\n"
            f"3. **方法论（概念层）**\n"
            f"   从架构/算法层描述方法，不展开实现细节。\n\n"
            f"4. **主要结果与意义**\n"
            f"   实验显示了什么？关键数字代表什么意义？\n\n"
            f"5. **局限性与开放问题**\n"
            f"   论文承认了哪些尚未解决或不在范围内的问题？\n\n"
            f"语气要求：精确、学术、分析性。避免“论文讨论了……”这类空泛表达，直接陈述结论。"
        )

    # ── NORMAL ITERATION ──────────────────────────────────────────────────────
    planning_block = (
        "=== 首轮迭代：先制定阅读策略 ===\n"
        "在开始阅读前，先浏览所有可用章节并判断：\n"
        "  • 哪些章节最能直接揭示论文的主张（CLAIMS）与贡献（CONTRIBUTIONS）？\n"
        "    （摘要、引言、结论，以及任何“贡献”小节优先级最高。）\n"
        "  • 哪些章节包含你后续需要的实验证据？\n"
        "  • 哪些章节对当前任务价值较低（如附录、致谢）？\n"
        "先读信息密度最高的章节。你可以在一次 read_section() 调用中批量读取多个相关章节。\n"
        "请在输出的 thoughts 字段中记录你的阅读计划。\n\n"
        if is_first_iteration
        else ""
    )

    return (
        f"任务：{deps.task}\n\n"
        f"=== 迭代 {deps.current_loop}/{deps.max_loops} "
        f"（本轮结束后剩余轮次：{remaining_loops}）===\n\n"
        f"{planning_block}"
        f"{processed_sections_str}\n"
        f"{unprocessed_str}\n\n"
        f"{previous_context}\n\n"
        f"=== 本轮规则 ===\n"
        f"1. 必须且仅能调用一次 read_section()。\n"
        f"   • 章节选择要有策略，不要只按顺序读下一个。\n"
        f"   • 优先选择能直接回答：论文主张了什么？创新点是什么？\n"
        f"   • 可在一次调用中批量读取多个主题相关章节。\n"
        f"   • 剩余轮次：{remaining_loops}。如果本轮后只剩 1 轮，\n"
        f"     请聚焦最高价值的未读章节，跳过低优先级内容。\n\n"
        f"2. 必须且仅能调用一次 add_memory()。\n"
        f"   • 记录贡献（CONTRIBUTIONS）、主张（CLAIMS）和关键发现（KEY FINDINGS），不要写内容摘要。\n"
        f"   • 记忆内容请使用要点格式：\n"
        f"     - [CLAIM] <论文提出的主张>\n"
        f"     - [CONTRIBUTION] <论文的新颖贡献>\n"
        f"     - [RESULT] <关键实验结果，尽量包含数字>\n"
        f"     - [METHOD] <核心技术，仅在架构层面重要时记录>\n"
        f"   • 不要记录背景、动机或模板化描述。\n\n"
        f"3. 立即返回 ReActIterationOutput，不要继续调用其他工具。\n"
        f"   • thoughts：说明本轮阅读依据，以及下一步准备读什么（和原因）。\n"
        f"   • should_continue：\n"
        f"     - 若仍有高价值未读章节，设为 True。\n"
        f"     - 仅当你已准备好在本次输出中亲自写出最终报告时，才设为 False。\n"
        f"       警告：设置 should_continue=False 意味着这是你产出报告的最后机会。\n"
        f"       若设为 should_continue=False，你必须同时完整填写 report 字段，\n"
        f"       之后不会再有自动综合步骤。\n"
        f"       若 report 为空，不要设置 should_continue=False。\n\n"
        f"   • report：保持为空（仅最终迭代才进行综合）。\n\n"
        f"=== 允许的工具 ===\n"
        f"  ✓ read_section(section_names)  —— 仅调用一次\n"
        f"  ✓ add_memory(memory)           —— 仅调用一次\n"
        f"  ✓ get_all_memory()             —— 常规迭代请不要使用，仅生成报告的迭代才应使用\n"
    )


@react_iteration_agent.tool
async def read_section(ctx: RunContext[ReActIterationDeps], section_names: list[str] | str | None = None) -> str:
    """Read one or more unprocessed sections. Call EXACTLY ONCE per iteration.

    Args:
        section_names: Section names to read (list or comma-separated string).
                      If None, reads the first unprocessed section.
    """
    deps = ctx.deps
    iteration_state = _get_iteration_state(ctx)
    iteration_input = deps.iteration_input

    # Enforce single-call rule
    if iteration_state.read_section_called:
        return "错误：本轮已调用过 read_section()，不能重复调用。"

    iteration_state.read_section_called = True

    available_sections = iteration_input.available_sections
    processed = iteration_input.processed_sections
    unprocessed = [s for s in available_sections if s not in processed]

    if not unprocessed:
        return "提示：所有章节都已处理，没有可读取内容。"

    # Normalize and resolve requested section names
    normalized_section_names = normalize_section_names(section_names)
    requested_sections = normalized_section_names or [unprocessed[0]]

    resolved_sections: list[str] = []
    for section in requested_sections:
        if section in available_sections and section not in resolved_sections:
            resolved_sections.append(section)
            continue

        matched = deps.document.get_closest_section_name(section, threshold=0.7)
        if matched and matched not in resolved_sections:
            resolved_sections.append(matched)

    # Filter to only unprocessed sections
    next_sections = [s for s in resolved_sections if s not in processed]
    if not next_sections:
        return f"提示：请求章节 {resolved_sections} 已处理。当前未处理章节：{unprocessed}"

    # Fetch content
    section_content = get_section_content(deps.document, next_sections)
    if not section_content.strip():
        return f"警告：在章节 {next_sections} 中未找到内容，请尝试其他章节。"

    iteration_state.sections_read = next_sections
    iteration_state.section_content = section_content

    if deps.show_progress:
        print(f"\n[Iteration] Read sections: {', '.join(next_sections)}")

    return f"章节内容（读取自：{', '.join(next_sections)}）：\n\n{section_content}\n\n现在请调用 add_memory() 记录提取到的发现。"


@react_iteration_agent.tool
async def add_memory(ctx: RunContext[ReActIterationDeps], memory: str) -> str:
    """Record discovered memory from the current read. Call EXACTLY ONCE per iteration.

    Args:
        memory: Extracted findings and key information from newly read sections.
    """
    iteration_state = _get_iteration_state(ctx)

    # Enforce single-call rule
    if iteration_state.add_memory_called:
        return "错误：本轮已调用过 add_memory()，不能重复调用。"

    iteration_state.add_memory_called = True

    fragment = memory.strip()
    if not fragment:
        return "警告：记忆文本为空。"

    iteration_state.memory_text = fragment

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

    async def analyze_one_iteration(
        self,
        document: Document,
        iteration_input: ReActIterationInput,
        current_loop: int = 1,
        max_loops: int = 8,
        accumulated_memory: str = "",
        show_progress: bool = True,
    ) -> tuple[ReActIterationOutput, ReActIterationState]:
        """Perform a single ReAct iteration: read once, add memory once, return thoughts.

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

        deps = ReActIterationDeps(
            document=document,
            task=iteration_input.task,
            iteration_input=iteration_input,
            current_loop=current_loop,
            max_loops=max_loops,
            accumulated_memory=accumulated_memory,
            show_progress=show_progress,
        )
        iteration_state = ReActIterationState()

        try:
            user_message = (
                "最终迭代——不要调用 read_section()。先调用 get_all_memory()，然后返回结构化最终报告，并设置 should_continue=False。"
                if deps.current_loop >= deps.max_loops
                else "读取最具策略价值的未处理章节，将贡献与主张提炼为记忆，然后返回你的思考与阅读计划。"
            )
            result = await self.agent.run(
                user_message,
                deps=deps,
                model=self.model,
                metadata={"iteration_state": iteration_state},
            )

            output = result.output if isinstance(result.output, ReActIterationOutput) else None
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.error(f"Iteration failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            output = None

        # If agent fails to produce valid output, construct fallback
        if output is None:
            output = ReActIterationOutput(
                thoughts="Iteration failed: could not complete analysis.",
                should_continue=True,
            )

        # Ensure convergence at hard loop limit even if model still asks to continue.
        if current_loop >= max_loops and output.should_continue:
            self.logger.info("Final loop reached with should_continue=True; forcing should_continue=False")
            output.should_continue = False
            if not output.report.strip() and accumulated_memory.strip():
                output.report = accumulated_memory.strip()

        return output, iteration_state

    async def analyze_document(
        self,
        document: Document,
        task: str,
        max_loops: int = 8,
        show_progress: bool = True,
    ) -> ReActIterationOutput:
        """Main analysis method: iterate up to max_loops times, accumulating memory.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            max_loops: Maximum number of iterations to perform
            show_progress: Whether to print progress information

        Returns:
            Final ReAct iteration output with should_continue=False and final report populated
        """
        self.logger.debug(f"Starting ReAct multi-iteration analysis (max_loops={max_loops})")

        available_sections = document.get_section_names()
        processed_sections: list[str] = []
        accumulated_memory = ""
        current_thoughts = ""
        last_iteration_output: ReActIterationOutput | None = None

        # Initialize with first iteration input
        iteration_input = ReActIterationInput(
            task=task,
            previous_thoughts="",
            processed_sections=[],
            available_sections=available_sections,
        )

        # Run iterations
        for loop_num in range(1, max_loops + 1):
            self.logger.debug(f"Starting iteration {loop_num}/{max_loops}")

            # Run one iteration
            iteration_output, iteration_state = await self.analyze_one_iteration(
                document=document,
                iteration_input=iteration_input,
                current_loop=loop_num,
                max_loops=max_loops,
                accumulated_memory=accumulated_memory,
                show_progress=show_progress,
            )
            last_iteration_output = iteration_output

            # Update tracking
            for section in iteration_state.sections_read:
                if section not in processed_sections:
                    processed_sections.append(section)

            if iteration_state.memory_text:
                accumulated_memory = (
                    f"{accumulated_memory}\n\n{iteration_state.memory_text}" if accumulated_memory else iteration_state.memory_text
                )
            current_thoughts = iteration_output.thoughts

            remaining_sections = [section for section in available_sections if section not in processed_sections]

            if show_progress:
                console.print(Markdown(current_thoughts))

            # Prepare next iteration input
            iteration_input = ReActIterationInput(
                task=task,
                previous_thoughts=current_thoughts,
                processed_sections=processed_sections,
                available_sections=available_sections,
            )

            # Check stopping conditions
            if not iteration_output.should_continue:
                self.logger.info(f"Agent decided to stop after iteration {loop_num}")
                if show_progress:
                    print("Agent completed analysis")
                break

            if not remaining_sections:
                self.logger.info(f"All sections processed after iteration {loop_num}")
                if show_progress:
                    print("All available sections have been read")
                break

        self.logger.info(f"Multi-iteration analysis completed after {loop_num} iterations")

        final_thoughts = current_thoughts or "Analysis complete."
        final_report = (
            last_iteration_output.report.strip()
            if last_iteration_output and last_iteration_output.report.strip()
            else accumulated_memory.strip()
        )
        if not final_report:
            final_report = "No memory content was generated during analysis."

        final_output = ReActIterationOutput(
            thoughts=final_thoughts,
            should_continue=False,
            report=final_report,
        )

        return final_output

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier})"
