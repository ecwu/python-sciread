"""Coordinate-agent application use case."""

from __future__ import annotations

from ...agent.coordinate import ComprehensiveAnalysisResult
from ...agent.coordinate import CoordinateAgent
from ...platform.logging import get_logger
from .common import load_document

logger = get_logger(__name__)


async def run_coordinate_analysis(document_file_path: str, model: str = "deepseek/deepseek-chat") -> ComprehensiveAnalysisResult:
    """Run the coordinate-agent workflow on a PDF document."""
    logger.debug(f"Creating CoordinateAgent with model: {model}")
    coordinate_agent = CoordinateAgent(model)

    document = load_document(document_file_path, to_markdown=True)
    logger.debug(f"Document loaded successfully: {len(document.text)} characters")
    logger.debug(f"Document split into {len(document.chunks)} chunks")

    if not document.text.strip():
        raise ValueError("Failed to load PDF: no text content extracted")

    section_names = document.get_section_names()
    logger.debug(f"Discovered {len(section_names)} sections: {section_names}")

    try:
        result = await coordinate_agent.analyze(document)
        logger.debug(f"Total execution time: {result.total_execution_time:.2f} seconds")
        logger.debug(f"Agents executed: {result.execution_summary['total_agents_executed']}")
        logger.debug(f"Successful agents: {result.execution_summary['successful_agents']}")
        logger.debug(f"Final report length: {len(result.final_report)} characters")
        return result
    except Exception as exc:
        logger.error(f"Comprehensive analysis failed: {exc}")
        raise
