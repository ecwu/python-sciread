"""Search-react application use case."""

from __future__ import annotations

from ...agent.search_react import analyze_file_with_search_react
from ...platform.logging import get_logger

logger = get_logger(__name__)


async def run_search_react_analysis(
    document_file: str,
    task: str,
    *,
    model: str = "deepseek-chat",
    max_loops: int = 5,
    show_progress: bool = True,
    retriever: str = "hybrid",
    compare: list[str] | None = None,
    top_k: int = 5,
    neighbor_window: int = 1,
):
    """Run search-react analysis on a document."""
    logger.debug(f"Starting search-react analysis with file: {document_file}")
    logger.debug(
        "Configuration: "
        f"model={model}, max_loops={max_loops}, show_progress={show_progress}, "
        f"retriever={retriever}, compare={compare}, top_k={top_k}, neighbor_window={neighbor_window}"
    )

    try:
        return await analyze_file_with_search_react(
            document_file,
            task,
            model=model,
            max_loops=max_loops,
            to_markdown=True,
            show_progress=show_progress,
            retriever=retriever,
            compare=compare,
            top_k=top_k,
            neighbor_window=neighbor_window,
        )
    except Exception as exc:
        logger.error(f"Search-react analysis failed: {exc}")
        raise
