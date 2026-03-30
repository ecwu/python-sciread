"""ReAct agent for intelligent document analysis."""

import asyncio
import json
import traceback
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ..document import Document
from ..document.document_renderers import get_sections_content
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.react_models import AnalysisReport

logger = get_logger(__name__)


@dataclass
class ReActDeps:
    """Immutable dependencies for ReActAgent tool-driven analysis."""

    document: Document
    task: str
    max_loops: int = 8
    show_progress: bool = True


@dataclass
class ReActState:
    """Mutable runtime state managed on the Python side."""

    current_sections: list[str] = field(default_factory=list)
    processed_sections: list[str] = field(default_factory=list)
    current_report: str = ""
    loop_count: int = 0


def _get_state(ctx: RunContext[ReActDeps]) -> ReActState:
    """Retrieve per-run mutable state from agent metadata."""
    metadata = ctx.metadata or {}
    state = metadata.get("react_state")
    if not isinstance(state, ReActState):
        raise RuntimeError("ReAct runtime state is missing from run metadata.")
    return state


# ReAct agent implementation


def load_and_process_document(
    file_path: str | Path, to_markdown: bool = True
) -> Document:
    """Load and process a document using markdown conversion and natural section splitting.

    Args:
        file_path: Path to the PDF file
        to_markdown: Whether to convert PDF to markdown using Mineru API

    Returns:
        Document instance with processed chunks using natural markdown sections
    """
    logger.info(f"Loading document from {file_path} (to_markdown={to_markdown})")

    # Create document with markdown conversion and auto-splitting
    # Document.from_file() automatically loads and splits the document when auto_split=True
    document = Document.from_file(file_path, to_markdown=to_markdown, auto_split=True)

    logger.info(
        f"Document processed into {len(document.chunks)} chunks with natural markdown sections"
    )
    logger.info(f"Available sections: {document.get_section_names()}")

    return document


def get_initial_sections(document: Document) -> list[str]:
    """Get the initial sections to start analysis (abstract and introduction).

    Args:
        document: Processed document instance

    Returns:
        List of section names to start with
    """
    available_sections = document.get_section_names()
    initial_sections = []

    try:
        # Use unified section matching for better results
        # Look for abstract first
        abstract_match = document.get_closest_section_name("abstract", threshold=0.7)
        if abstract_match:
            initial_sections.append(abstract_match)

        # Look for introduction next
        intro_match = document.get_closest_section_name("introduction", threshold=0.7)
        if intro_match and intro_match not in initial_sections:
            initial_sections.append(intro_match)

        # If no matches found, use first section
        if not initial_sections and available_sections:
            initial_sections = [available_sections[0]]

    except Exception as e:
        logger.warning(f"Unified section matching failed, using fallback approach: {e}")

        # Fallback to original approach
        for section in available_sections:
            if "abstract" in section.lower():
                initial_sections.append(section)
                break

        for section in available_sections:
            if "introduction" in section.lower() and section not in initial_sections:
                initial_sections.append(section)
                break

        if not initial_sections and available_sections:
            initial_sections = [available_sections[0]]

    logger.info(f"Initial sections selected: {initial_sections}")
    return initial_sections


def format_status_summary(stage: str, current_loop: int, max_loops: int) -> str:
    """Format status summary combining stage, loop count, and remaining loops.

    Args:
        stage: Current analysis stage description
        current_loop: Current iteration number (1-based)
        max_loops: Maximum number of loops allowed

    Returns:
        Formatted status summary string
    """
    return f"{stage} (loop {current_loop} of {max_loops})"


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
    logger.debug(
        f"Retrieved content for sections {section_names}: {len(combined_content)} characters"
    )

    return combined_content


def normalize_section_names(section_names: list[str] | str | None) -> list[str] | None:
    """Normalize tool input to a list of section names.

    Some models send a JSON-encoded list as a string. This parser keeps the
    tool callable instead of failing at schema validation time.
    """
    if section_names is None:
        return None

    if isinstance(section_names, list):
        return [
            name for name in section_names if isinstance(name, str) and name.strip()
        ]

    if isinstance(section_names, str):
        raw = section_names.strip()
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [
                    name for name in parsed if isinstance(name, str) and name.strip()
                ]
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
) -> AnalysisReport:
    """Analyze a document using the ReAct agent.

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
    logger.info(f"Starting ReAct analysis for file: {document_file}")
    logger.info(f"Task: {task[:100]}...")
    logger.info(
        f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}"
    )

    # Check if file exists
    if not Path(document_file).exists():
        raise FileNotFoundError(f"Document file not found: {document_file}")

    # Load and process the document
    document = load_and_process_document(document_file, to_markdown=to_markdown)

    # Create and run the ReAct agent
    agent = ReActAgent(model=model, max_loops=max_loops)
    result = await agent.analyze_document(document, task, show_progress=show_progress)

    logger.debug("ReAct analysis completed successfully!")
    return result


def analyze_document_with_react_sync(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> AnalysisReport:
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


react_agent = Agent(
    deps_type=ReActDeps,
    output_type=AnalysisReport,
)


@react_agent.system_prompt
async def react_system_prompt(ctx: RunContext[ReActDeps]) -> str:
    """Generate system prompt for tool-driven ReAct analysis."""
    deps = ctx.deps
    state = _get_state(ctx)
    current_report = state.current_report or "[No report yet]"
    processed_sections = (
        ", ".join(state.processed_sections) if state.processed_sections else "None"
    )

    return (
        "You are an expert academic research analyst using a native ReAct tool-calling workflow.\n\n"
        f"Task: {deps.task}\n"
        f"Available sections: {', '.join(deps.document.get_section_names())}\n"
        f"Already processed sections: {processed_sections}\n"
        f"Current report:\n{current_report}\n\n"
        f"Max read_section calls: {deps.max_loops}\n\n"
        "Context rules:\n"
        "- In each turn, you can only rely on the current full report and the newly read section content.\n"
        "- Previously read section content will not remain available unless you write it into the report now.\n"
        "- If information is not incorporated into the report in the current turn, it will be lost in later turns.\n\n"
        "You MUST use tools to analyze the paper:\n"
        "1) Call read_section(section_names) to fetch one or more sections.\n"
        "   - section_names should be a JSON array of strings.\n"
        '   - Preferred: {"section_names": ["1 introduction", "2 method"]}\n'
        '   - Also tolerated: {"section_names": "[\\"1 introduction\\", \\"2 method\\"]"}\n'
        "2) Immediately after each read, call update_report(report_fragment) with only the new report fragment derived from the just-read sections.\n"
        "3) Repeat until the report clearly covers research questions, methodology, results, and contributions.\n"
        "4) Then return the final structured analysis.\n\n"
        "Rules:\n"
        "- Do not invent content that was not read via tools.\n"
        "- Prefer unprocessed sections first.\n"
        "- update_report appends to the cumulative report; do not rewrite the entire report each turn.\n"
        "- Each report_fragment should contain only new findings from the latest read sections, merged with minimal overlap.\n"
        "- After reading a section, do not call read_section again until the report has been updated.\n"
        "- Keep report updates non-redundant and evidence-driven.\n"
        "- The final answer must populate: summary, research_questions, methodology, key_findings, contributions, limitations, sections_covered, and final_report.\n"
    )


@react_agent.tool
async def read_section(
    ctx: RunContext[ReActDeps], section_names: list[str] | str | None = None
) -> str:
    """Read one or more document sections and update progress state.

    section_names is expected as a list, but stringified JSON lists are
    accepted for model compatibility.
    """
    deps = ctx.deps
    state = _get_state(ctx)

    if state.loop_count >= deps.max_loops:
        return "READ_LIMIT_REACHED: You have reached max read attempts. Finish using existing evidence and return final report."

    available_sections = deps.document.get_section_names()
    unprocessed_sections = [
        s for s in available_sections if s not in state.processed_sections
    ]
    normalized_section_names = normalize_section_names(section_names)
    requested_sections = (
        normalized_section_names or state.current_sections or unprocessed_sections[:1]
    )

    resolved_sections: list[str] = []
    for section in requested_sections:
        if section in available_sections and section not in resolved_sections:
            resolved_sections.append(section)
            continue

        matched = deps.document.get_closest_section_name(section, threshold=0.7)
        if matched and matched not in resolved_sections:
            resolved_sections.append(matched)

    next_sections = [s for s in resolved_sections if s not in state.processed_sections]
    if not next_sections:
        remaining = [s for s in available_sections if s not in state.processed_sections]
        return (
            "NO_NEW_SECTIONS: Requested sections are already processed or not found. "
            f"Remaining unprocessed sections: {remaining if remaining else 'None'}."
        )

    state.loop_count += 1
    state.current_sections = next_sections

    section_content = get_section_content(deps.document, next_sections)
    if not section_content.strip():
        return "SECTION_CONTENT_EMPTY: No content found for selected sections. Select different sections."

    for section in next_sections:
        if section not in state.processed_sections:
            state.processed_sections.append(section)

    remaining = [s for s in available_sections if s not in state.processed_sections]

    if deps.show_progress:
        print(f"\n--- Loop {state.loop_count}/{deps.max_loops} ---")
        print(f"Sections analyzed: {', '.join(next_sections)}")
        print(f"Remaining sections: {', '.join(remaining) if remaining else 'None'}")
        print("-" * 50)

    return (
        f"READ_RESULT\n"
        f"Loop: {state.loop_count}/{deps.max_loops}\n"
        f"Read sections: {', '.join(next_sections)}\n"
        f"Remaining sections: {', '.join(remaining) if remaining else 'None'}\n\n"
        f"{section_content}\n\n"
        "NEXT_ACTION_REQUIRED: Call update_report with only the newly written report fragment before reading more sections."
    )


@react_agent.tool
async def update_report(ctx: RunContext[ReActDeps], report_fragment: str) -> str:
    """Append a newly written report fragment to the cumulative report."""
    deps = ctx.deps
    state = _get_state(ctx)

    fragment = report_fragment.strip()
    if not fragment:
        return "REPORT_NOT_UPDATED: Empty report provided."

    if state.current_report.strip():
        state.current_report = f"{state.current_report}\n\n{fragment}"
    else:
        state.current_report = fragment

    if deps.show_progress:
        print(
            f"Report fragment appended: +{len(fragment)} chars, total {len(state.current_report)} chars"
        )

    return (
        f"REPORT_APPENDED: +{len(fragment)} chars, total {len(state.current_report)} chars, "
        f"{len(state.processed_sections)} sections covered."
    )


class ReActAgent:
    """ReAct agent for intelligent document analysis with iterative section exploration.

    This agent implements the Reasoning and Acting pattern to analyze documents
    by iteratively reading sections, making decisions about what to read next,
    and building a comprehensive report using native message history.
    """

    def __init__(self, model: str = "deepseek-chat", max_loops: int = 8):
        """Initialize the ReAct agent.

        Args:
            model: Model identifier for the LLM provider
            max_loops: Maximum number of analysis iterations
        """
        self.logger = get_logger(__name__)
        self.max_loops = max_loops
        self.model = get_model(model)
        self.model_identifier = model
        self.agent = react_agent

        self.logger.info(
            f"Initialized ReActAgent with model: {model} (max_loops={max_loops})"
        )

    async def analyze_document(
        self, document: Document, task: str, show_progress: bool = True
    ) -> AnalysisReport:
        """Main analysis method using a single native tool-calling run.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            show_progress: Whether to print reasoning at each step

        Returns:
            Structured analysis report
        """
        self.logger.info(f"Starting ReAct analysis for task: {task[:100]}...")

        deps = ReActDeps(
            document=document,
            task=task,
            max_loops=self.max_loops,
            show_progress=show_progress,
        )
        state = ReActState(current_sections=get_initial_sections(document))

        try:
            self.logger.debug("Running ReAct agent in a single tool-calling session")
            result = await self.agent.run(
                "Start analysis. Use tools to read sections, append a report fragment after each read, then return the final consolidated report.",
                deps=deps,
                model=self.model,
                metadata={"react_state": state},
            )
            report = (
                result.output if isinstance(result.output, AnalysisReport) else None
            )
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            report = None

        if report is None:
            report = AnalysisReport(
                summary="",
                research_questions=[],
                methodology="",
                key_findings=[],
                contributions=[],
                limitations=None,
                sections_covered=list(state.processed_sections),
                final_report=state.current_report,
            )
        elif not report.final_report.strip() and state.current_report.strip():
            report.final_report = state.current_report
            if not report.sections_covered:
                report.sections_covered = list(state.processed_sections)

        self.logger.info(
            f"ReAct analysis completed after {state.loop_count} read loops"
        )

        # Log and print the final report
        if report.final_report:
            self.logger.info(f"Report length: {len(report.final_report)} characters")

            # Print the final report to console
            print("\n" + "=" * 80)
            print("FINAL ANALYSIS REPORT")
            print("=" * 80)
            print(report.final_report)
            print("=" * 80)
        else:
            self.logger.warning("No final report generated")
            print("\nWarning: No analysis report was generated")

        return report

    def analyze_document_sync(
        self, document: Document, task: str, show_progress: bool = True
    ) -> AnalysisReport:
        """Synchronous wrapper for top-level sync callers such as the CLI."""
        return asyncio.run(
            self.analyze_document(
                document=document, task=task, show_progress=show_progress
            )
        )

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier}, max_loops={self.max_loops})"
