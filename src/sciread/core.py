import asyncio
from pathlib import Path

from .agent import create_agent, remove_references_section
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


async def main(txt_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Main function that processes a txt file using the document agent.

    Args:
        txt_file_path: Path to the txt file to process
        model: Model identifier for the LLM provider (default: "deepseek/deepseek-chat")

    Returns:
        Analysis result as a string

    Raises:
        FileNotFoundError: If the txt file is not found
        Exception: If the analysis fails
    """
    logger.info(f"Starting main function with txt file: {txt_file_path}")

    # Check if file exists
    if not Path(txt_file_path).exists():
        raise FileNotFoundError(f"Txt file not found: {txt_file_path}")

    # Create an agent
    agent = create_agent(model)

    # Load the txt file
    sample_text = Path(txt_file_path).read_text(encoding="latin-1")
    doc = Document.from_text(sample_text)
    logger.info(f"Document created: {len(doc.text)} characters")

    # Test the reference removal function
    cleaned_text = remove_references_section(sample_text)
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


def run_main(txt_file_path: str, model: str = "deepseek/deepseek-chat"):
    """Run the main function synchronously.

    This is a wrapper around the async main function for use in synchronous contexts.

    Args:
        txt_file_path: Path to the txt file to process
        model: Model identifier for the LLM provider

    Returns:
        Analysis result as a string
    """
    return asyncio.run(main(txt_file_path, model))
