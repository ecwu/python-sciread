"""ReAct agent for intelligent document analysis."""

from pathlib import Path
from typing import List

from pydantic_ai import Agent

from ..document import Document
from ..document.loaders import PdfLoader
from ..document.external_clients import MineruClient
from ..llm_provider import get_model
from ..logging_config import get_logger
from .react_models import ReActAgentOutput

logger = get_logger(__name__)


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
        if 'abstract' in section.lower():
            initial_sections.append(section)
            break

    # Look for introduction next
    for section in available_sections:
        if 'introduction' in section.lower() and section not in initial_sections:
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
        logger.warning(f"No content found for sections: {section_names}")
        return ""

    # Sort by position to maintain document order
    sections_chunks.sort(key=lambda chunk: chunk.position)

    # Combine content with section headers
    content_parts = []
    for chunk in sections_chunks:
        section_name = chunk.chunk_name if chunk.chunk_name != "unknown" else "unknown"
        content_parts.append(f"=== {section_name.upper()} ===\n{chunk.content}")

    combined_content = "\n\n".join(content_parts)
    logger.info(f"Retrieved content for sections {section_names}: {len(combined_content)} characters")

    return combined_content


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

    def _format_agent_prompt(self, task: str, available_sections: List[str], status: str,
                            section_content: str, current_report: str, processed_sections: List[str]) -> str:
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
        prompt = f"""ANALYSIS TASK: {task}

CURRENT STATUS: {status}

AVAILABLE SECTIONS: {', '.join(available_sections)}

ALREADY PROCESSED SECTIONS: {', '.join(processed_sections) if processed_sections else 'None'}

CURRENT REPORT BUILT SO FAR:
{current_report if current_report else '[No previous analysis yet - this is the first iteration]'}

=== SECTIONS TO ANALYZE IN THIS ITERATION ===
{section_content if section_content else '[No section content provided]'}

=== YOUR ANALYSIS TASK ===
Based on the sections provided above and your existing analysis, please:

1. Analyze the current section content thoroughly
2. Update your understanding of the research based on this new information
3. Decide whether you should continue reading more sections or stop

Please provide your response as a structured analysis with the following components:
- Should you stop analysis? (true/false)
- New report content to add based on these sections
- Which sections to read next (if continuing)
- Your reasoning for these decisions

Focus on understanding: research questions, methodology, key findings, and contributions."""

        return prompt

    def _create_agent(self) -> Agent[str, ReActAgentOutput]:
        """Create and configure the pydantic-ai agent for ReAct analysis."""

        system_prompt = """You are an expert academic research analyst using the ReAct (Reasoning and Acting) pattern to analyze academic papers intelligently.

Your primary goal is to understand academic papers by focusing on the essential research elements: research questions, methodology, results, and contributions. You should analyze document sections strategically to build a comprehensive understanding.

CORE ANALYSIS FRAMEWORK:
For standard academic analysis, focus on these key areas:
1. **Research Questions & Objectives**: What problem is being addressed? What are the specific research questions or hypotheses?
2. **Methodology & Approach**: How did the researchers conduct their study? What methods, data, and procedures were used?
3. **Key Findings & Results**: What did the research discover? What are the main results and evidence?
4. **Contributions & Significance**: Why does this research matter? What are the main contributions to the field?

CORE PRINCIPLES:
1. Start by understanding what content you've been given and what has already been reported
2. Analyze the current section content thoroughly in the context of understanding the research
3. Make strategic decisions about which sections to read next based on:
   - Information gaps in your current understanding of the research
   - Logical flow of academic papers (abstract → intro → methods → results → discussion)
   - Section names that indicate important content (methods, results, discussion, etc.)
   - Relevance to understanding the complete research story

BEHAVIORAL GUIDELINES:
- Build on the existing analysis rather than repeating content
- Focus on sections that will help complete your understanding of the research
- Follow the natural progression of academic research when selecting sections
- Avoid selecting sections that have already been processed
- Stop analysis when you have a complete picture of the research questions, methods, results, and contributions

REPORT WRITING:
- Write in a professional academic tone
- Focus exclusively on the paper's content and findings
- Structure your analysis logically around the key research elements
- Add new insights that build upon previous sections

SECTION SELECTION STRATEGY:
- After abstract/introduction, typically prioritize: methods → results → discussion → conclusion
- Look for sections that will fill gaps in your understanding of the research
- Consider what information is needed to complete the analysis framework
- Be strategic - you have limited iterations, so choose the most informative sections

STOPPING CRITERIA:
- Stop when you can clearly articulate: the research questions, methodology, key results, and contributions
- Stop when you have sufficient information from the most relevant sections
- Continue reading only if there are clearly important gaps in understanding the research

Remember: You are building a comprehensive understanding of the research piece by piece. Each iteration should add meaningful new information to complete the research analysis."""

        return Agent(
            model=self.model,
            system_prompt=system_prompt,
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
            status = format_status_summary(
                "Analyzing sections", self.loop_count, self.max_loops
            )

            # Prepare input for the agent as a formatted string
            input_prompt = self._format_agent_prompt(
                task=task,
                available_sections=document.get_section_names(),
                status=status,
                section_content=section_content,
                current_report=self.current_report,
                processed_sections=self.processed_sections.copy()
            )

            self.logger.info(f"Loop {self.loop_count}/{self.max_loops}: Analyzing sections: {current_sections}")

            # Run the agent
            try:
                self.logger.debug(f"Running agent with string prompt")
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
        return self.current_report

    def _update_state(self, result: ReActAgentOutput, sections: List[str]):
        """Update internal state after each iteration.

        Args:
            result: Agent output from current iteration
            sections: Sections that were analyzed in this iteration
        """
        # Add new report content
        if result.report_section.strip():
            # Add section separator if this isn't the first content
            if self.current_report.strip():
                self.current_report += "\n\n"
            self.current_report += result.report_section

        # Mark sections as processed
        self.processed_sections.extend(sections)
        self.processed_sections = list(set(self.processed_sections))  # Remove duplicates

        self.logger.debug(f"Report length: {len(self.current_report)} chars, Processed sections: {self.processed_sections}")

    def get_agent_info(self) -> dict:
        """Get information about the agent configuration and state.

        Returns:
            Dictionary with agent configuration and current state
        """
        return {
            "model": self.model_identifier,
            "max_loops": self.max_loops,
            "current_loop": self.loop_count,
            "processed_sections": self.processed_sections.copy(),
            "report_length": len(self.current_report),
        }


def analyze_document_with_react(
    file_path: str | Path,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = True,
    show_progress: bool = True
) -> str:
    """Main entry point for ReAct document analysis.

    Args:
        file_path: Path to the PDF file to analyze
        task: Analysis task or question about the document
        model: Model identifier for the LLM provider
        max_loops: Maximum number of analysis iterations
        to_markdown: Whether to convert PDF to markdown using Mineru API
        show_progress: Whether to print reasoning at each step (default: True)

    Returns:
        Comprehensive analysis report generated by the ReAct agent

    Example:
        >>> report = analyze_document_with_react(
        ...     "paper.pdf",
        ...     "What are the main contributions and methodology of this paper?",
        ...     max_loops=6,
        ...     show_progress=True
        ... )
        >>> # During analysis, reasoning will be printed for each step
        >>> # Final complete report is returned as a string
        >>> print(report)
    """
    logger.info(f"Starting ReAct analysis for document: {file_path}")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}")

    try:
        # Load and process document
        document = load_and_process_document(file_path, to_markdown=to_markdown)

        # Create and run ReAct agent
        agent = ReActAgent(model=model, max_loops=max_loops)
        report = agent.analyze_document(document, task, show_progress=show_progress)

        # Print final report header if progress was shown
        if show_progress:
            print("\n" + "=" * 60)
            print("FINAL ANALYSIS REPORT")
            print("=" * 60)
            print(report)
            print("=" * 60)

        # Log completion information
        agent_info = agent.get_agent_info()
        logger.info(f"ReAct analysis completed successfully:")
        logger.info(f"  - Loops used: {agent_info['current_loop']}/{agent_info['max_loops']}")
        logger.info(f"  - Sections processed: {len(agent_info['processed_sections'])}")
        logger.info(f"  - Report length: {agent_info['report_length']} characters")

        return report

    except Exception as e:
        logger.error(f"ReAct analysis failed: {e}")
        raise