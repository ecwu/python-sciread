"""ReAct agent for intelligent document analysis.

This module implements a ReAct (Reasoning and Acting) agent for intelligent
iterative document analysis using pydantic-ai framework.
"""

import traceback
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List

from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext

from ..document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.react_models import ReActAgentOutput
from .prompts.react import format_agent_prompt

logger = get_logger(__name__)


@dataclass
class ReActDeps:
    """Dependencies for ReActAgent iterative analysis."""

    document: Document
    task: str
    max_loops: int = 8
    show_progress: bool = True
    current_sections: List[str] = field(default_factory=list)
    processed_sections: List[str] = field(default_factory=list)
    current_report: str = ""
    loop_count: int = 0


@dataclass
class ReActState:
    """State management for ReAct analysis using message history."""

    current_sections: List[str] = field(default_factory=list)
    processed_sections: List[str] = field(default_factory=list)
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

    # Look for abstract first
    for section in available_sections:
        if "abstract" in section.lower():
            initial_sections.append(section)
            break

    # Look for introduction next
    for section in available_sections:
        if "introduction" in section.lower() and section not in initial_sections:
            initial_sections.append(section)
            break

    # If no abstract/introduction found, use first section
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

    sections_chunks = document.get_sections_by_name(section_names)
    if not sections_chunks:
        logger.warning(f"No chunks found for sections: {section_names}")
        return ""

    # Sort chunks by position to maintain reading order
    sections_chunks.sort(key=lambda chunk: chunk.position)

    # Combine content with section headers
    content_parts = []
    for chunk in sections_chunks:
        section_name = chunk.chunk_name if chunk.chunk_name != "unknown" else "unknown"
        content_parts.append(f"=== {section_name.upper()} ===\n{chunk.content}")

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

        # Create the pydantic-ai agent with dependencies and structured output
        self.agent = Agent(
            model=self.model,
            deps_type=ReActDeps,
            output_type=ReActAgentOutput,
        )

        # Add context-aware system prompt with better error handling
        @self.agent.system_prompt
        async def react_system_prompt(ctx: RunContext[ReActDeps]) -> str:
            """Generate system prompt with current analysis state."""
            deps = ctx.deps

            # Format status summary
            status = f"Analyzing sections (loop {deps.loop_count + 1} of {deps.max_loops})"

            # Get content for current sections
            section_content = ""
            if deps.current_sections:
                section_content = get_section_content(deps.document, deps.current_sections)
                if not section_content.strip():
                    raise ModelRetry(
                        f"No content found for sections: {deps.current_sections}. "
                        "Please select different sections or provide more specific guidance."
                    )

            # Format the agent prompt with all necessary information
            return format_agent_prompt(
                task=deps.task,
                available_sections=deps.document.get_section_names(),
                status=status,
                section_content=section_content,
                current_report=deps.current_report,
                processed_sections=deps.processed_sections.copy(),
            )

        self.logger.info(f"Initialized ReActAgent with model: {model} (max_loops={max_loops})")

    def analyze_document(self, document: Document, task: str, show_progress: bool = True) -> str:
        """Main analysis method that orchestrates the ReAct loop using native message history.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            show_progress: Whether to print reasoning at each step

        Returns:
            Comprehensive analysis report
        """
        self.logger.info(f"Starting ReAct analysis for task: {task[:100]}...")

        # Initialize analysis state
        state = ReActState()
        state.current_sections = get_initial_sections(document)
        message_history = []

        # Main ReAct loop with message history
        while state.loop_count < self.max_loops:
            state.loop_count += 1

            self.logger.info(f"Loop {state.loop_count}/{self.max_loops}: Analyzing sections: {state.current_sections}")

            try:
                # Create dependencies for this iteration
                deps = ReActDeps(
                    document=document,
                    task=task,
                    max_loops=self.max_loops,
                    show_progress=show_progress,
                    current_sections=state.current_sections,
                    processed_sections=state.processed_sections,
                    current_report=state.current_report,
                    loop_count=state.loop_count,
                )

                # Run the agent with message history for context persistence
                self.logger.debug("Running agent with message history")
                result = self.agent.run_sync("Execute analysis iteration", deps=deps, message_history=message_history)
                agent_output = result.output

                self.logger.debug(f"Agent response: should_stop={agent_output.should_stop}, next_sections={agent_output.next_sections}")

                # Print reasoning for this iteration if show_progress is enabled
                if show_progress:
                    print(f"\n--- Loop {state.loop_count}/{self.max_loops} ---")
                    print(f"Sections analyzed: {', '.join(state.current_sections)}")
                    print(f"Reasoning: {agent_output.reasoning}")
                    if agent_output.should_stop:
                        print("Decision: STOP - Analysis complete")
                    else:
                        print(f"Next sections to read: {', '.join(agent_output.next_sections) if agent_output.next_sections else 'None'}")
                    print("-" * 50)

                # Update state
                if agent_output.report_section.strip():
                    if state.current_report:
                        state.current_report += "\n\n"
                    state.current_report += agent_output.report_section

                # Mark sections as processed
                for section in state.current_sections:
                    if section not in state.processed_sections:
                        state.processed_sections.append(section)

                # Update message history with this iteration
                message_history.extend(result.new_messages())

                # Check if agent wants to stop
                if agent_output.should_stop:
                    self.logger.info(f"Agent chose to stop after loop {state.loop_count}: {agent_output.reasoning}")
                    break

                # Determine next sections
                next_sections = [s for s in agent_output.next_sections if s not in state.processed_sections]

                if not next_sections:
                    self.logger.info("No new sections to analyze (all selected sections already processed)")
                    break

                state.current_sections = next_sections

            except Exception as e:
                self.logger.error(f"Agent execution failed in loop {state.loop_count}: {e}")
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                break

        self.logger.info(f"ReAct analysis completed after {state.loop_count} loops")

        # Log and print the final report
        if state.current_report:
            self.logger.info(f"Report length: {len(state.current_report)} characters")

            # Print the final report to console
            print("\n" + "=" * 80)
            print("FINAL ANALYSIS REPORT")
            print("=" * 80)
            print(state.current_report)
            print("=" * 80)
        else:
            self.logger.warning("No final report generated")
            print("\nWarning: No analysis report was generated")

        return state.current_report

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier}, max_loops={self.max_loops})"
