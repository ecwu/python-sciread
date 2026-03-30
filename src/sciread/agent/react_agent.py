"""ReAct agent for intelligent document analysis.

This module implements a ReAct (Reasoning and Acting) agent for intelligent
iterative document analysis using pydantic-ai framework.
"""

import asyncio
import traceback
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Thread

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ..document import Document
from ..document.document_renderers import get_sections_content
from ..llm_provider import get_model
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ReActDeps:
    """Dependencies for ReActAgent tool-driven analysis."""

    document: Document
    task: str
    max_loops: int = 8
    show_progress: bool = True
    current_sections: list[str] = field(default_factory=list)
    processed_sections: list[str] = field(default_factory=list)
    current_report: str = ""
    loop_count: int = 0


# ReAct agent implementation


def load_and_process_document(file_path: str | Path, to_markdown: bool = True) -> Document:
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

    logger.info(f"Document processed into {len(document.chunks)} chunks with natural markdown sections")
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
    logger.debug(f"Retrieved content for sections {section_names}: {len(combined_content)} characters")

    return combined_content


def analyze_document_with_react(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True,
) -> str:
    """Analyze a document using the ReAct agent.

    Args:
        document_file: Path to the document file (PDF or TXT)
        task: Analysis task or question about the document
        model: Model identifier for the LLM provider
        max_loops: Maximum number of analysis iterations
        to_markdown: Whether to convert PDF to markdown using Mineru API
        show_progress: Whether to show progress during analysis

    Returns:
        Comprehensive analysis report generated by the ReAct agent

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the analysis fails
    """
    logger.info(f"Starting ReAct analysis for file: {document_file}")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}")

    # Check if file exists
    if not Path(document_file).exists():
        raise FileNotFoundError(f"Document file not found: {document_file}")

    # Load and process the document
    document = load_and_process_document(document_file, to_markdown=to_markdown)

    # Create and run the ReAct agent
    agent = ReActAgent(model=model, max_loops=max_loops)
    result = agent.analyze_document(document, task, show_progress=show_progress)

    logger.debug("ReAct analysis completed successfully!")
    return result


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

        # Create a native tool-calling ReAct agent.
        self.agent = Agent(
            model=self.model,
            deps_type=ReActDeps,
            output_type=str,
        )

        # Use tool-calling to read sections and accumulate a report until complete.
        @self.agent.system_prompt
        async def react_system_prompt(ctx: RunContext[ReActDeps]) -> str:
            """Generate system prompt for tool-driven ReAct analysis."""
            deps = ctx.deps

            return (
                "You are an expert academic research analyst using a native ReAct tool-calling workflow.\n\n"
                f"Task: {deps.task}\n"
                f"Available sections: {', '.join(deps.document.get_section_names())}\n"
                f"Max read_section calls: {deps.max_loops}\n\n"
                "You MUST use tools to analyze the paper:\n"
                "1) Call read_section(section_names) to fetch one or more sections.\n"
                "2) After each read, extract key findings and call append_to_report(report_fragment).\n"
                "3) Repeat until the report clearly covers research questions, methodology, results, and contributions.\n"
                "4) Then return the final consolidated report as plain text.\n\n"
                "Rules:\n"
                "- Do not invent content that was not read via tools.\n"
                "- Prefer unprocessed sections first.\n"
                "- Keep report updates non-redundant and evidence-driven.\n"
            )

        @self.agent.tool
        async def read_section(ctx: RunContext[ReActDeps], section_names: list[str] | None = None) -> str:
            """Read one or more document sections and update progress state."""
            deps = ctx.deps

            if deps.loop_count >= deps.max_loops:
                return "READ_LIMIT_REACHED: You have reached max read attempts. Finish using existing evidence and return final report."

            available_sections = deps.document.get_section_names()
            unprocessed_sections = [s for s in available_sections if s not in deps.processed_sections]
            requested_sections = section_names or deps.current_sections or unprocessed_sections[:1]

            resolved_sections: list[str] = []
            for section in requested_sections:
                if section in available_sections and section not in resolved_sections:
                    resolved_sections.append(section)
                    continue

                matched = deps.document.get_closest_section_name(section, threshold=0.7)
                if matched and matched not in resolved_sections:
                    resolved_sections.append(matched)

            next_sections = [s for s in resolved_sections if s not in deps.processed_sections]
            if not next_sections:
                remaining = [s for s in available_sections if s not in deps.processed_sections]
                return (
                    "NO_NEW_SECTIONS: Requested sections are already processed or not found. "
                    f"Remaining unprocessed sections: {remaining if remaining else 'None'}."
                )

            deps.loop_count += 1
            deps.current_sections = next_sections

            section_content = get_section_content(deps.document, next_sections)
            if not section_content.strip():
                return "SECTION_CONTENT_EMPTY: No content found for selected sections. Select different sections."

            for section in next_sections:
                if section not in deps.processed_sections:
                    deps.processed_sections.append(section)

            remaining = [s for s in available_sections if s not in deps.processed_sections]

            if deps.show_progress:
                print(f"\n--- Loop {deps.loop_count}/{deps.max_loops} ---")
                print(f"Sections analyzed: {', '.join(next_sections)}")
                print(f"Remaining sections: {', '.join(remaining) if remaining else 'None'}")
                print("-" * 50)

            return (
                f"READ_RESULT\n"
                f"Loop: {deps.loop_count}/{deps.max_loops}\n"
                f"Read sections: {', '.join(next_sections)}\n"
                f"Remaining sections: {', '.join(remaining) if remaining else 'None'}\n\n"
                f"{section_content}"
            )

        @self.agent.tool
        async def append_to_report(ctx: RunContext[ReActDeps], report_fragment: str) -> str:
            """Append new analysis content to the cumulative report."""
            deps = ctx.deps

            fragment = report_fragment.strip()
            if not fragment:
                return "REPORT_NOT_UPDATED: Empty fragment provided."

            if deps.current_report:
                deps.current_report += "\n\n"
            deps.current_report += fragment

            if deps.show_progress:
                print(f"Report updated: {len(deps.current_report)} characters")

            return f"REPORT_UPDATED: Current length={len(deps.current_report)} chars; processed_sections={len(deps.processed_sections)}."

        self.logger.info(f"Initialized ReActAgent with model: {model} (max_loops={max_loops})")

    def _run_agent_sync(self, prompt: str, deps: ReActDeps):
        """Run the async agent from sync code without using deprecated loop APIs."""

        async def _runner():
            return await self.agent.run(prompt, deps=deps)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        # If a loop is already running in this thread, execute in a dedicated thread.
        outcome: dict[str, object] = {}

        def _target() -> None:
            try:
                outcome["result"] = asyncio.run(_runner())
            except Exception as e:  # pragma: no cover - rare path
                outcome["error"] = e

        thread = Thread(target=_target, daemon=True)
        thread.start()
        thread.join()

        if "error" in outcome:
            raise outcome["error"]  # type: ignore[misc]

        return outcome.get("result")

    def analyze_document(self, document: Document, task: str, show_progress: bool = True) -> str:
        """Main analysis method using a single native tool-calling run.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            show_progress: Whether to print reasoning at each step

        Returns:
            Comprehensive analysis report
        """
        self.logger.info(f"Starting ReAct analysis for task: {task[:100]}...")

        deps = ReActDeps(
            document=document,
            task=task,
            max_loops=self.max_loops,
            show_progress=show_progress,
            current_sections=get_initial_sections(document),
        )

        try:
            self.logger.debug("Running ReAct agent in a single tool-calling session")
            result = self._run_agent_sync(
                "Start analysis. Use tools to read sections and append report fragments, then return the final consolidated report.",
                deps=deps,
            )
            final_report = result.output.strip() if isinstance(result.output, str) else ""
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            final_report = ""

        if not final_report:
            final_report = deps.current_report

        self.logger.info(f"ReAct analysis completed after {deps.loop_count} read loops")

        # Log and print the final report
        if final_report:
            self.logger.info(f"Report length: {len(final_report)} characters")

            # Print the final report to console
            print("\n" + "=" * 80)
            print("FINAL ANALYSIS REPORT")
            print("=" * 80)
            print(final_report)
            print("=" * 80)
        else:
            self.logger.warning("No final report generated")
            print("\nWarning: No analysis report was generated")

        return final_report

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier}, max_loops={self.max_loops})"
