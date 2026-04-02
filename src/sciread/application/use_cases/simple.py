"""Simple-agent application use case."""

from __future__ import annotations

from ...agent.simple import DEFAULT_TASK_PROMPT
from ...agent.simple import analyze_file_with_simple
from ...platform.logging import get_logger

logger = get_logger(__name__)


async def run_simple_analysis(document_file_path: str, model: str = "deepseek/deepseek-chat") -> str:
    """Run the single-agent paper analysis workflow."""
    logger.debug(f"Starting simple analysis with document file: {document_file_path}")

    try:
        return await analyze_file_with_simple(
            file_path=document_file_path,
            task_prompt=DEFAULT_TASK_PROMPT,
            model=model,
            remove_references=True,
            clean_text=True,
        )
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        raise
