"""ReAct agent for intelligent document analysis with single-iteration loops."""

import asyncio
import json
import traceback
from rich.console import Console
from rich.markdown import Markdown
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import RunContext

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

    previous_context = f"Previous thoughts:\n{iteration_input.previous_thoughts}" if iteration_input.previous_thoughts else ""
    processed_sections_str = (
        f"Already processed: {', '.join(iteration_input.processed_sections)}"
        if iteration_input.processed_sections
        else "No sections processed yet."
    )
    unprocessed = [s for s in iteration_input.available_sections if s not in iteration_input.processed_sections]
    unprocessed_str = f"Remaining unprocessed sections: {', '.join(unprocessed)}" if unprocessed else "All sections have been read."

    # ── FINAL ITERATION: synthesis only, no new reading ──────────────────────
    if is_final_iteration:
        return (
            f"TASK: {deps.task}\n\n"
            f"=== FINAL ITERATION ({deps.current_loop}/{deps.max_loops}) — SYNTHESIS ONLY ===\n\n"
            f"{processed_sections_str}\n\n"
            f"{previous_context}\n\n"
            f"STRICT RULES FOR THIS ITERATION:\n"
            f"  ✗ DO NOT call read_section() — all reading is finished.\n"
            f"  ✓ Step 1: Call get_all_memory() to retrieve every finding accumulated so far.\n"
            f"  ✓ Step 2: Synthesize those findings into a final structured report (see format below).\n"
            f"  ✓ Step 3: Return ReActIterationOutput with should_continue=False and report populated.\n\n"
            f"=== FINAL REPORT FORMAT ===\n"
            f"Write a concise, contribution-focused report. Do NOT produce a section-by-section recap.\n"
            f"Structure it as follows:\n\n"
            f"1. **Core Research Question & Thesis**\n"
            f"   What gap does this paper address? What is the central claim?\n\n"
            f"2. **Key Contributions** (the most important section)\n"
            f"   List 3–5 concrete, specific contributions the paper makes.\n"
            f"   Use precise language — name methods, datasets, metrics, improvements.\n\n"
            f"3. **Methodology (conceptual)**\n"
            f"   Describe the approach at an architectural/algorithmic level, not implementation detail.\n\n"
            f"4. **Main Results & Implications**\n"
            f"   What did the experiments show? What do the numbers mean?\n\n"
            f"5. **Limitations & Open Questions**\n"
            f"   What does the paper acknowledge as unresolved or out of scope?\n\n"
            f"Tone: precise, academic, analytical. Avoid filler phrases like 'the paper discusses...' — state claims directly."
        )

    # ── NORMAL ITERATION ──────────────────────────────────────────────────────
    planning_block = (
        f"=== FIRST ITERATION: PLAN YOUR READING STRATEGY ===\n"
        f"Before reading anything, survey all available sections and decide:\n"
        f"  • Which sections most directly reveal the paper's CLAIMS and CONTRIBUTIONS?\n"
        f"    (Abstract, Introduction, Conclusion, and any 'Contributions' subsection are highest priority.)\n"
        f"  • Which sections contain experimental evidence you will need?\n"
        f"  • Which sections are low-value for the task (e.g., appendices, acknowledgements)?\n"
        f"Start with the highest-signal sections. You may batch multiple related sections in one read_section() call.\n"
        f"Record your reading plan in the 'thoughts' field of your output.\n\n"
        if is_first_iteration
        else ""
    )

    return (
        f"TASK: {deps.task}\n\n"
        f"=== ITERATION {deps.current_loop}/{deps.max_loops} "
        f"(loops remaining after this: {remaining_loops}) ===\n\n"
        f"{planning_block}"
        f"{processed_sections_str}\n"
        f"{unprocessed_str}\n\n"
        f"{previous_context}\n\n"
        f"=== RULES FOR THIS ITERATION ===\n"
        f"1. Call read_section() EXACTLY ONCE.\n"
        f"   • Choose sections STRATEGICALLY — not just the next one in sequence.\n"
        f"   • Prefer sections that directly answer: What does this paper CLAIM? What is NEW?\n"
        f"   • You may batch multiple thematically related sections in one call.\n"
        f"   • Remaining loops: {remaining_loops}. If only 1 loop remains after this,\n"
        f"     focus on the highest-value unread sections; skip low-priority ones.\n\n"
        f"2. Call add_memory() EXACTLY ONCE.\n"
        f"   • Record CONTRIBUTIONS, CLAIMS, and KEY FINDINGS — NOT a content summary.\n"
        f"   • Format your memory as bullet points:\n"
        f"     - [CLAIM] <what the paper asserts>\n"
        f"     - [CONTRIBUTION] <what is novel>\n"
        f"     - [RESULT] <key empirical finding with numbers if available>\n"
        f"     - [METHOD] <core technique, only if architecturally significant>\n"
        f"   • Omit anything that is background, motivation, or boilerplate.\n\n"
        f"3. Return ReActIterationOutput immediately — no further tool calls.\n"
        f"   • thoughts: Your reading rationale and what you plan to read next (and why).\n"
        f"   • should_continue: \n"
        f"     - Set True if high-value unread sections remain.\n"
        f"     - Set False ONLY WHEN you are ready to write the final report YOURSELF in this same output.\n"
        f"       WARNING: Setting should_continue=False means THIS IS YOUR LAST CHANCE to produce a report.\n"
        f"       If you set should_continue=False, you MUST also populate the 'report' field with the full\n"
        f"       structured analysis — there is NO automatic synthesis step after this.\n"
        f"       Do NOT set should_continue=False if report will be empty.\n\n"
        f"   • report: Leave EMPTY (synthesis happens only in the final iteration).\n\n"
        f"=== TOOLS ALLOWED THIS ITERATION ===\n"
        f"  ✓ read_section(section_names)  — call ONCE\n"
        f"  ✓ add_memory(memory)           — call ONCE\n"
        f"  ✗ get_all_memory()             — FORBIDDEN in normal iterations; reserved for final only\n"
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
        return "ERROR: read_section already called in this iteration. Cannot call it again."

    iteration_state.read_section_called = True

    available_sections = iteration_input.available_sections
    processed = iteration_input.processed_sections
    unprocessed = [s for s in available_sections if s not in processed]

    if not unprocessed:
        return "INFO: All sections already processed. No content to read."

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
        return f"INFO: Requested sections {resolved_sections} are already processed. Unprocessed available: {unprocessed}"

    # Fetch content
    section_content = get_section_content(deps.document, next_sections)
    if not section_content.strip():
        return f"WARNING: No content found for sections {next_sections}. Try different sections."

    iteration_state.sections_read = next_sections
    iteration_state.section_content = section_content

    if deps.show_progress:
        print(f"\n[Iteration] Read sections: {', '.join(next_sections)}")

    return (
        f"SECTION_CONTENT (read from: {', '.join(next_sections)}):\n\n{section_content}\n\nNow call add_memory() with extracted findings."
    )


@react_iteration_agent.tool
async def add_memory(ctx: RunContext[ReActIterationDeps], memory: str) -> str:
    """Record discovered memory from the current read. Call EXACTLY ONCE per iteration.

    Args:
        memory: Extracted findings and key information from newly read sections.
    """
    deps = ctx.deps
    iteration_state = _get_iteration_state(ctx)

    # Enforce single-call rule
    if iteration_state.add_memory_called:
        return "ERROR: add_memory already called in this iteration. Cannot call it again."

    iteration_state.add_memory_called = True

    fragment = memory.strip()
    if not fragment:
        return "WARNING: Empty memory text."

    iteration_state.memory_text = fragment

    return "MEMORY RECORDED"


@react_iteration_agent.tool
async def get_all_memory(ctx: RunContext[ReActIterationDeps]) -> str:
    """Read the complete accumulated memory from all previous iterations.

    Use this ONLY on the final iteration when preparing the complete structured analysis.
    Returns the full synthesized memory built up across all iterations.
    """
    deps = ctx.deps

    if not deps.accumulated_memory.strip():
        return "INFO: No accumulated memory yet from previous iterations."

    return (
        f"FULL_ACCUMULATED_MEMORY:\n\n"
        f"{deps.accumulated_memory}\n\n"
        f"Use this complete memory to guide final reasoning and final report generation."
    )


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
                "FINAL ITERATION — DO NOT call read_section(). "
                "Call get_all_memory() first, then return a structured final report with should_continue=False."
                if deps.current_loop >= deps.max_loops
                else "Read the most strategically valuable unprocessed sections, extract contributions and claims into memory, then return your thoughts and reading plan."
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
                print(f"[Thoughts] {iteration_output.thoughts}...")

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
