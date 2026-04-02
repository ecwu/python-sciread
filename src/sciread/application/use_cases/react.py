"""ReAct-agent application use case."""

from __future__ import annotations

from ...agent.react import analyze_file_with_react
from ...platform.logging import get_logger

logger = get_logger(__name__)


async def run_react_analysis(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 5,
    show_progress: bool = True,
):
    """Run ReAct analysis on a document."""
    logger.debug(f"Starting ReAct analysis with file: {document_file}")
    logger.debug(f"Task: {task[:100]}...")
    logger.debug(f"Configuration: model={model}, max_loops={max_loops}, show_progress={show_progress}")

    try:
        return await analyze_file_with_react(
            document_file,
            task,
            model=model,
            max_loops=max_loops,
            to_markdown=True,
            show_progress=show_progress,
        )
    except Exception as exc:
        logger.error(f"ReAct analysis failed: {exc}")
        raise
