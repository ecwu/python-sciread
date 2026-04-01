"""Discussion-agent application use case."""

from __future__ import annotations

from ...agent.discussion import DiscussionAgent
from ...agent.discussion import DiscussionResult
from ...platform.logging import get_logger
from .common import load_document

logger = get_logger(__name__)


async def run_discussion_analysis(document_file_path: str, model: str = "deepseek-chat") -> tuple[dict[str, object], DiscussionResult]:
    """Run the discussion-based multi-agent workflow."""
    logger.debug(f"Creating DiscussionAgent with model: {model}")
    discussion_agent = DiscussionAgent(model)

    document = load_document(document_file_path, to_markdown=True)
    logger.debug(f"Document loaded successfully: {len(document.text)} characters")
    logger.debug(f"Document split into {len(document.chunks)} chunks")

    section_names = document.get_section_names()
    logger.debug(f"Discovered {len(section_names)} sections: {section_names}")

    overview = {
        "document_title": document.metadata.title or "Untitled",
        "total_content_chars": len(document.text),
        "chunk_count": len(document.chunks),
        "section_names": section_names,
    }

    if not document.text.strip():
        raise ValueError("Failed to load document: no text content extracted")

    try:
        result = await discussion_agent.analyze_document(document)
        return overview, result
    except Exception as exc:
        logger.error(f"Discussion-based analysis failed: {exc}")
        raise
