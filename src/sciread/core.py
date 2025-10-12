import asyncio
from pathlib import Path
from typing import Optional

from .agent import create_agent, remove_references_section, ToolAgent
from .document import Document
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
    agent = create_agent(model)

    # Load the document file using the document loading system
    # Use to_markdown=False for agent mode to keep traditional text extraction
    doc = Document.from_file(document_file_path, to_markdown=False)

    # Load the document content
    load_result = doc.load()
    if not load_result.success:
        error_msg = f"Failed to load document: {load_result.errors}"
        logger.error(error_msg)
        # Also log any warnings that might provide context
        if load_result.warnings:
            logger.warning(f"Document loading warnings: {load_result.warnings}")
        raise ValueError(error_msg)

    logger.info(f"Document loaded successfully: {len(doc.text)} characters")
    logger.info(f"Document loaded using: {load_result.extraction_info.get('extraction_method', 'unknown')}")

    # Log any warnings from the loading process
    if load_result.warnings:
        for warning in load_result.warnings:
            logger.warning(f"Document loading warning: {warning}")

    # Test the reference removal function
    cleaned_text = remove_references_section(doc.text)
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
        result = await agent.analyze_document(
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


async def comprehensive_analysis(pdf_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Comprehensive document analysis using the multi-agent ToolAgent system.

    This function uses the ToolAgent with multiple expert sub-agents to provide
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
    logger.info(f"Starting comprehensive analysis with ToolAgent for file: {pdf_file_path}")

    # Check if file exists
    if not Path(pdf_file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

    # Create the multi-agent system
    tool_agent = ToolAgent(model)

    # Load the PDF file with to_markdown=True for ToolAgent
    doc = Document.from_file(pdf_file_path, to_markdown=True)
    logger.info(f"Document created from PDF: {pdf_file_path}")

    # Load and split the document
    load_result = doc.load()
    if not load_result.success:
        raise ValueError(f"Failed to load PDF: {load_result.errors}")

    logger.info(f"PDF loaded successfully: {len(doc.text)} characters")

    # Split document into chunks
    chunks = doc.split()
    logger.info(f"Document split into {len(chunks)} chunks")

    # Run comprehensive analysis
    logger.info("Starting comprehensive document analysis with ToolAgent...")
    try:
        result = await tool_agent.analyze_document(doc)

        logger.info("Comprehensive analysis completed successfully!")
        logger.info(f"Total execution time: {result.total_execution_time:.2f} seconds")
        logger.info(f"Agents executed: {result.execution_summary['total_agents_executed']}")
        logger.info(f"Successful agents: {result.execution_summary['successful_agents']}")

        return result

    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise


def run_comprehensive_analysis(pdf_file_path: str, model: str = "deepseek/deepseek-chat", debug_output: Optional[str] = None):
    """Run comprehensive analysis synchronously.

    This is a wrapper around the async comprehensive_analysis function for use in synchronous contexts.

    Args:
        pdf_file_path: Path to the PDF file to process
        model: Model identifier for the LLM provider
        debug_output: Optional path to save interaction log for debugging

    Returns:
        ComprehensiveAnalysisResult object containing all sub-agent analyses
        and a synthesized final report
    """
    result = asyncio.run(comprehensive_analysis(pdf_file_path, model))

    # Save debug output if requested
    if debug_output:
        # Create a new ToolAgent to access the interaction log
        tool_agent = ToolAgent(model)
        tool_agent.interaction_log = result.interaction_log if hasattr(result, "interaction_log") else []
        tool_agent.save_interaction_log(debug_output)
        logger.info(f"Debug interaction log saved to: {debug_output}")

    return result


async def comprehensive_analysis_with_debug(pdf_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Comprehensive analysis with ToolAgent that captures all interactions.

    Args:
        pdf_file_path: Path to the PDF file to process
        model: Model identifier for the LLM provider

    Returns:
        ComprehensiveAnalysisResult with interaction log attached
    """
    logger.info(f"Starting comprehensive analysis with ToolAgent for file: {pdf_file_path}")

    # Check if file exists
    if not Path(pdf_file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

    # Create the multi-agent system
    tool_agent = ToolAgent(model)

    # Load the PDF file with to_markdown=True for ToolAgent
    doc = Document.from_file(pdf_file_path, to_markdown=True)
    logger.info(f"Document created from PDF: {pdf_file_path}")

    # Load and split the document
    load_result = doc.load()
    if not load_result.success:
        raise ValueError(f"Failed to load PDF: {load_result.errors}")

    logger.info(f"PDF loaded successfully: {len(doc.text)} characters")

    # Split document into chunks
    chunks = doc.split()
    logger.info(f"Document split into {len(chunks)} chunks")

    # Run comprehensive analysis
    logger.info("Starting comprehensive document analysis with ToolAgent...")
    try:
        result = await tool_agent.analyze_document(doc)

        # Attach interaction log to result
        result.interaction_log = tool_agent.get_interaction_log()

        logger.info("Comprehensive analysis completed successfully!")
        logger.info(f"Total execution time: {result.total_execution_time:.2f} seconds")
        logger.info(f"Agents executed: {result.execution_summary['total_agents_executed']}")
        logger.info(f"Successful agents: {result.execution_summary['successful_agents']}")
        logger.info(f"Total interactions logged: {len(result.interaction_log)}")

        return result

    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise


def run_comprehensive_analysis_with_debug(pdf_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Run comprehensive analysis with debug logging synchronously.

    Args:
        pdf_file_path: Path to the PDF file to process
        model: Model identifier for the LLM provider

    Returns:
        ComprehensiveAnalysisResult with interaction log attached
    """
    return asyncio.run(comprehensive_analysis_with_debug(pdf_file_path, model))
