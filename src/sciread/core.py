import asyncio
from pathlib import Path
from typing import Optional

from .agent import SimpleAgent, remove_references, CoordinateAgent, analyze_document_with_react
from .document import Document, DocumentFactory
from .logging_config import get_logger

logger = get_logger(__name__)


def compute(args):
    """Compute function that returns the longest string from input arguments."""
    logger.debug(f"Computing result for {len(args)} arguments: {args}")

    if not args:
        logger.warning("No arguments provided to compute function")
        return ""

    result = max(args, key=len)
    logger.debug(f"Compute result: {result}")
    return result


async def main(document_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Main function that processes a document file using the document agent.

    Args:
        document_file_path: Path to the document file to process (PDF or TXT)
        model: Model identifier for the LLM provider (default: "deepseek/deepseek-chat")

    Returns:
        Analysis result as a string

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the analysis fails
    """
    logger.info(f"Starting main function with document file: {document_file_path}")

    # Check if file exists
    if not Path(document_file_path).exists():
        raise FileNotFoundError(f"Document file not found: {document_file_path}")

    # Create an agent
    agent = SimpleAgent(model, max_retries=3, timeout=300.0)

    # Load the document file using the document loading system
    # Use to_markdown=False for agent mode to keep traditional text extraction
    doc = Document.from_file(document_file_path, to_markdown=False, auto_split=True)

    # Document is automatically loaded and split with the new API
    logger.info(f"Document loaded successfully: {len(doc.text)} characters")
    logger.info(f"Document split into {len(doc.chunks)} chunks")

    # Check if document was loaded successfully
    if not doc.text.strip():
        error_msg = "Failed to load document: no text content extracted"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Test the reference removal function
    cleaned_text = remove_references(doc.text)
    logger.info(f"Text after reference removal: {len(cleaned_text)} characters")

    # Define the task prompt (same as in test_agent.py)
    task_prompt = """Your task is to write a detailed report using the Feynman technique to explain a given paper. When creating this report, you should approach it as if you were the author of the paper.
Here are some important constraints:
- Title and basic info first: Start the report with the paper title, authors, and publication details.
- Content Format: When writing the content of the report, use multiple levels of titles and subtitles to organize the information clearly, using markdown formats such as bold & italic text, code block, table, and lists. This will make the report easier to read and understand.
- Serious Work Requirement: This report is a serious piece of work, so avoid using emojis when writing the report.
- Content Don'ts: Do not start the content with the paper title, it already displayed in the page title; Do not mention/introduce the faynman technique in the content.
"""

    logger.info("Starting document analysis...")
    try:
        result = await agent.analyze(
            document=doc,
            task_prompt=task_prompt,
            remove_references=True,
            clean_text=True,
        )

        logger.info("Analysis completed successfully!")
        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def run_main(document_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Run the main function synchronously.

    This is a wrapper around the async main function for use in synchronous contexts.

    Args:
        document_file_path: Path to the document file to process (PDF or TXT)
        model: Model identifier for the LLM provider

    Returns:
        Analysis result as a string
    """
    return asyncio.run(main(document_file_path, model))


async def comprehensive_analysis(
    pdf_file_path: str, model: str = "deepseek/deepseek-chat"
):
    """Comprehensive document analysis using the multi-agent CoordinateAgent system.

    This function uses the CoordinateAgent with multiple expert sub-agents to provide
    a detailed analysis of academic papers, including metadata extraction,
    methodology analysis, experiments evaluation, and future directions.

    Args:
        pdf_file_path: Path to the PDF file to process
        model: Model identifier for the LLM provider (default: "deepseek/deepseek-chat")

    Returns:
        ComprehensiveAnalysisResult object containing all sub-agent analyses
        and a synthesized final report

    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: If the analysis fails
    """
    logger.info(
        f"Starting comprehensive analysis with CoordinateAgent for file: {pdf_file_path}"
    )

    # Check if file exists
    if not Path(pdf_file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

    # Create the multi-agent system
    logger.debug(f"Creating CoordinateAgent with model: {model}")
    coordinate_agent = CoordinateAgent(model)

    # Load the PDF file with to_markdown=True for CoordinateAgent
    logger.debug(f"Loading document from file: {pdf_file_path}")
    doc = Document.from_file(pdf_file_path, to_markdown=True, auto_split=True)
    logger.debug(f"Document created from PDF: {pdf_file_path}")
    logger.debug(f"PDF loaded successfully: {len(doc.text)} characters")
    logger.debug(f"Document split into {len(doc.chunks)} chunks")

    # Extract and log section information
    section_names = doc.get_section_names()
    logger.debug(f"Discovered {len(section_names)} sections: {section_names}")

    # Display section information to user
    if section_names:
        print("\n📋 Document Structure Analysis")
        print(f"Found {len(section_names)} main sections:")
        for i, section_name in enumerate(section_names, 1):
            section_chunks = doc.get_sections_by_name([section_name])
            section_word_count = sum(
                len(chunk.content.split()) for chunk in section_chunks
            )
            print(
                f"  {i}. {section_name.title()} ({len(section_chunks)} chunks, ~{section_word_count} words)"
            )
        print()

        # Log section chunk distribution
        section_distribution = {}
        for section_name in section_names:
            section_chunks = doc.get_sections_by_name([section_name])
            section_distribution[section_name] = len(section_chunks)
        logger.info(f"Section distribution: {section_distribution}")
    else:
        print("\n📋 Document Structure Analysis")
        print("No named sections found - document will be analyzed as continuous text")
        print()
        logger.info(
            "No named sections found - document will be analyzed as continuous text"
        )

    # Check if document was loaded successfully
    if not doc.text.strip():
        raise ValueError("Failed to load PDF: no text content extracted")

    # Run comprehensive analysis
    logger.info("Starting comprehensive document analysis with CoordinateAgent...")
    logger.debug(f"Analyzing document with {len(doc.chunks)} chunks using {len(section_names)} sections")
    try:
        result = await coordinate_agent.analyze(doc)

        logger.info("Comprehensive analysis completed successfully!")
        logger.info(f"Total execution time: {result.total_execution_time:.2f} seconds")
        logger.info(
            f"Agents executed: {result.execution_summary['total_agents_executed']}"
        )
        logger.info(
            f"Successful agents: {result.execution_summary['successful_agents']}"
        )
        logger.debug(f"Final report length: {len(result.final_report)} characters")

        # Log section analysis summary if available
        if hasattr(result, "analysis_plan") and result.analysis_plan:
            plan = result.analysis_plan
            logger.info("Section-based analysis summary:")
            if plan.previous_methods_sections:
                logger.info(
                    f"  Previous methods sections: {plan.previous_methods_sections}"
                )
            if plan.research_questions_sections:
                logger.info(
                    f"  Research questions sections: {plan.research_questions_sections}"
                )
            if plan.methodology_sections:
                logger.info(f"  Methodology sections: {plan.methodology_sections}")
            if plan.experiments_sections:
                logger.info(f"  Experiments sections: {plan.experiments_sections}")
            if plan.future_directions_sections:
                logger.info(
                    f"  Future directions sections: {plan.future_directions_sections}"
                )

        return result

    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise


def run_comprehensive_analysis(
    pdf_file_path: str,
    model: str = "deepseek/deepseek-chat",
):
    """Run comprehensive analysis synchronously.

    This is a wrapper around the async comprehensive_analysis function for use in synchronous contexts.

    Args:
        pdf_file_path: Path to the PDF file to process
        model: Model identifier for the LLM provider

    Returns:
        ComprehensiveAnalysisResult object containing all sub-agent analyses
        and a synthesized final report
    """
    result = asyncio.run(comprehensive_analysis(pdf_file_path, model))
    return result






def run_react_analysis(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    show_progress: bool = True
):
    """Run ReAct analysis on a document.

    Args:
        document_file: Path to the document file to analyze (PDF or TXT)
        task: Analysis task or question about the document
        model: Model identifier for the LLM provider
        max_loops: Maximum number of analysis iterations
        show_progress: Whether to show progress during analysis

    Returns:
        Comprehensive analysis report generated by the ReAct agent

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the analysis fails
    """
    logger.info(f"Starting ReAct analysis with file: {document_file}")
    logger.info(f"Task: {task[:100]}...")
    logger.info(f"Configuration: model={model}, max_loops={max_loops}, show_progress={show_progress}")

    try:
        result = analyze_document_with_react(
            document_file,
            task,
            model=model,
            max_loops=max_loops,
            to_markdown=True,
            show_progress=show_progress
        )

        logger.info("ReAct analysis completed successfully!")
        return result

    except Exception as e:
        logger.error(f"ReAct analysis failed: {e}")
        raise
