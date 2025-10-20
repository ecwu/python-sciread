"""ReAct agent for intelligent document analysis.

This module implements a ReAct (Reasoning and Acting) agent for intelligent
iterative document analysis using pydantic-ai framework.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from ..document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .prompts.react import SYSTEM_PROMPT
from .prompts.react import format_agent_prompt

logger = get_logger(__name__)


# Pydantic models for ReAct agent input and output


class ReActAgentInput(BaseModel):
    """Input model for ReAct agent iterations."""

    task_prompt: str = Field(description="The original analysis task or question about the document")
    available_sections: List[str] = Field(description="List of all available section names in the document")
    status_summary: str = Field(
        description="Summary of current stage, loop count, and remaining loops (e.g., 'Initial analysis (loop 1 of 8)')"
    )
    section_content: str = Field(description="Content of the sections to analyze in this iteration (empty for initial step)")
    current_report: str = Field(description="The cumulative report built so far from previous iterations")
    processed_sections: List[str] = Field(description="List of sections that have already been processed")


class ReActAgentOutput(BaseModel):
    """Output model for ReAct agent iterations."""

    should_stop: bool = Field(description="Whether to stop the analysis process (True) or continue (False)")
    report_section: str = Field(description="New content generated for the current section content")
    next_sections: List[str] = Field(description="List of section names to analyze in the next iteration (empty if should_stop is True)")
    reasoning: str = Field(description="Explanation of why the agent made these choices (stop decision and section selection)")


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


def get_initial_sections(document: Document) -> List[str]:
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


def get_section_content(document: Document, section_names: List[str]) -> str:
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
    logger.info(f"Retrieved content for sections {section_names}: {len(combined_content)} characters")

    return combined_content


def analyze_document_with_react(
    document_file: str, task: str, model: str = "deepseek-chat", max_loops: int = 8, to_markdown: bool = True, show_progress: bool = True
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

    logger.info("ReAct analysis completed successfully!")
    return result


class ReActAgent:
    """ReAct agent for intelligent document analysis with iterative section exploration.

    This agent implements the Reasoning and Acting pattern to analyze documents
    by iteratively reading sections, making decisions about what to read next,
    and building a comprehensive report.
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

        # State management
        self.current_report = ""
        self.processed_sections: List[str] = []
        self.loop_count = 0

        # Create the pydantic-ai agent
        self.agent = self._create_agent()

        self.logger.info(f"Initialized ReActAgent with model: {model} (max_loops={max_loops})")

    def _format_agent_prompt(
        self,
        task: str,
        available_sections: List[str],
        status: str,
        section_content: str,
        current_report: str,
        processed_sections: List[str],
    ) -> str:
        """Format the agent prompt with all necessary information.

        Args:
            task: The original analysis task
            available_sections: List of all available section names
            status: Current status summary
            section_content: Content of sections to analyze
            current_report: Current cumulative report
            processed_sections: List of already processed sections

        Returns:
            Formatted prompt string for the agent
        """
        return format_agent_prompt(
            task=task,
            available_sections=available_sections,
            status=status,
            section_content=section_content,
            current_report=current_report,
            processed_sections=processed_sections,
        )

    def _create_agent(self) -> Agent[str, ReActAgentOutput]:
        """Create and configure the pydantic-ai agent for ReAct analysis."""
        return Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            output_type=ReActAgentOutput,
        )

    def analyze_document(self, document: Document, task: str, show_progress: bool = True) -> str:
        """Main analysis method that orchestrates the ReAct loop.

        Args:
            document: Processed document with natural markdown sections
            task: Analysis task or question about the document
            show_progress: Whether to print reasoning at each step

        Returns:
            Comprehensive analysis report
        """
        self.logger.info(f"Starting ReAct analysis for task: {task[:100]}...")

        # Reset state
        self.current_report = ""
        self.processed_sections = []
        self.loop_count = 0

        # Get initial sections (abstract + introduction)
        current_sections = get_initial_sections(document)

        # Main ReAct loop
        while self.loop_count < self.max_loops:
            self.loop_count += 1

            # Get content for current sections
            section_content = get_section_content(document, current_sections)

            # Format status summary
            status = format_status_summary("Analyzing sections", self.loop_count, self.max_loops)

            # Prepare input for the agent as a formatted string
            input_prompt = self._format_agent_prompt(
                task=task,
                available_sections=document.get_section_names(),
                status=status,
                section_content=section_content,
                current_report=self.current_report,
                processed_sections=self.processed_sections.copy(),
            )

            self.logger.info(f"Loop {self.loop_count}/{self.max_loops}: Analyzing sections: {current_sections}")

            # Run the agent
            try:
                self.logger.debug("Running agent with string prompt")
                result = self.agent.run_sync(input_prompt)
                self.logger.debug(f"Agent result type: {type(result)}")
                # Access the structured output from AgentRunResult
                agent_output = result.output

                self.logger.debug(f"Agent response: should_stop={agent_output.should_stop}, next_sections={agent_output.next_sections}")
            except Exception as e:
                self.logger.error(f"Agent execution failed in loop {self.loop_count}: {e}")
                self.logger.error(f"Exception type: {type(e)}")
                import traceback

                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                break

            # Print reasoning for this iteration if show_progress is enabled
            if show_progress:
                print(f"\n--- Loop {self.loop_count}/{self.max_loops} ---")
                print(f"Sections analyzed: {', '.join(current_sections)}")
                print(f"Reasoning: {agent_output.reasoning}")
                if agent_output.should_stop:
                    print("Decision: STOP - Analysis complete")
                else:
                    print(f"Next sections to read: {', '.join(agent_output.next_sections) if agent_output.next_sections else 'None'}")
                print("-" * 50)

            # Update state
            self._update_state(agent_output, current_sections)

            # Check if agent wants to stop
            if agent_output.should_stop:
                self.logger.info(f"Agent chose to stop after loop {self.loop_count}: {agent_output.reasoning}")
                break

            # Determine next sections
            next_sections = agent_output.next_sections

            # Filter out already processed sections
            next_sections = [s for s in next_sections if s not in self.processed_sections]

            if not next_sections:
                self.logger.info("No new sections to analyze (all selected sections already processed)")
                break

            current_sections = next_sections

        self.logger.info(f"ReAct analysis completed after {self.loop_count} loops")

        # Log and print the final report
        if self.current_report:
            self.logger.info("Final analysis report generated")
            self.logger.info(f"Report length: {len(self.current_report)} characters")

            # Print the final report to console
            print("\n" + "=" * 80)
            print("FINAL ANALYSIS REPORT")
            print("=" * 80)
            print(self.current_report)
            print("=" * 80)
        else:
            self.logger.warning("No final report generated")
            print("\nWarning: No analysis report was generated")

        return self.current_report

    def _update_state(self, agent_output: ReActAgentOutput, current_sections: List[str]) -> None:
        """Update agent state after each iteration.

        Args:
            agent_output: Output from the agent
            current_sections: Sections that were just analyzed
        """
        # Add new content to the report
        if agent_output.report_section.strip():
            if self.current_report:
                self.current_report += "\n\n"
            self.current_report += agent_output.report_section

        # Mark sections as processed
        for section in current_sections:
            if section not in self.processed_sections:
                self.processed_sections.append(section)

        self.logger.debug(f"State updated: processed sections={self.processed_sections}, report length={len(self.current_report)}")

    def __repr__(self) -> str:
        """String representation of the ReActAgent."""
        return f"ReActAgent(model={self.model_identifier}, max_loops={self.max_loops})"
