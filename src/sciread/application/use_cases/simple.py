"""Simple-agent application use case."""

from __future__ import annotations

from ...agent.shared import remove_references
from ...agent.simple import DEFAULT_TASK_PROMPT
from ...agent.simple import SimpleAgent
from ...platform.logging import get_logger
from .common import load_document

logger = get_logger(__name__)


async def run_simple_analysis(document_file_path: str, model: str = "deepseek/deepseek-chat") -> str:
    """Run the single-agent paper analysis workflow."""
    logger.debug(f"Starting simple analysis with document file: {document_file_path}")

    agent = SimpleAgent(model, max_retries=3, timeout=300.0)
    document = load_document(document_file_path, to_markdown=False)

    logger.debug(f"Document loaded successfully: {len(document.text)} characters")
    logger.debug(f"Document split into {len(document.chunks)} chunks")

    if not document.text.strip():
        raise ValueError("Failed to load document: no text content extracted")

    cleaned_text = remove_references(document.text)
    logger.debug(f"Text after reference removal: {len(cleaned_text)} characters")

    try:
        return await agent.analyze(
            document=document,
            task_prompt=DEFAULT_TASK_PROMPT,
            remove_references=True,
            clean_text=True,
        )
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        raise
