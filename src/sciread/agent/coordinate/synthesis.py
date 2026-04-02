"""Synthesis helpers for the coordinate agent."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent

from ...document import Document
from .models import AnalysisPlan
from .models import ComprehensiveAnalysisResult
from .planner import extract_abstract
from .prompts import SYNTHESIS_SYSTEM_PROMPT
from .prompts import build_report_synthesis_prompt
from .runtime import ANALYSIS_TASKS
from .runtime import INTERNAL_SECTIONS_KEY


def validate_pdf_document(document: Document) -> None:
    """Validate that the coordinate agent received a PDF document."""
    if not document.source_path or document.source_path.suffix.lower() != ".pdf":
        raise ValueError(
            "CoordinateAgent only supports PDF files. "
            f"Got: {document.source_path.suffix if document.source_path else 'unknown file type'}. "
            "Please provide a PDF document."
        )


def build_execution_summary(sub_agent_results: dict[str, Any]) -> dict[str, Any]:
    """Build a compact execution summary from expert results."""
    agent_results = get_agent_results(sub_agent_results)

    return {
        "total_agents_executed": len(agent_results),
        "successful_agents": sum(1 for result in agent_results.values() if result.get("success", False)),
        "failed_agents": sum(1 for result in agent_results.values() if not result.get("success", False)),
        "agent_results": {name: {"success": data.get("success", False)} for name, data in agent_results.items()},
    }


def get_agent_results(sub_agent_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return only actual agent results, excluding internal bookkeeping."""
    return {name: data for name, data in sub_agent_results.items() if name != INTERNAL_SECTIONS_KEY and isinstance(data, dict)}


def get_successful_result(sub_agent_results: dict[str, Any], analysis_type: str) -> Any:
    """Return a successful expert result or None."""
    result_data = sub_agent_results.get(analysis_type, {})
    if result_data.get("success"):
        return result_data.get("result")
    return None


def build_comprehensive_result(
    analysis_plan: AnalysisPlan,
    sub_agent_results: dict[str, Any],
    final_report: str,
    total_execution_time: float,
) -> ComprehensiveAnalysisResult:
    """Assemble the final coordinate-agent response model."""
    result_fields = {task.output_field: get_successful_result(sub_agent_results, task.analysis_type) for task in ANALYSIS_TASKS}

    return ComprehensiveAnalysisResult(
        analysis_plan=analysis_plan,
        **result_fields,
        execution_summary=build_execution_summary(sub_agent_results),
        final_report=final_report,
        total_execution_time=total_execution_time,
        sections_analyzed=sub_agent_results.get(INTERNAL_SECTIONS_KEY, {}),
    )


async def synthesize_report(
    analysis_plan: AnalysisPlan,
    sub_agent_results: dict[str, Any],
    document: Document,
    model,
    max_retries: int,
    timeout: float,
    logger,
) -> str:
    """Synthesize the final report from successful expert outputs."""
    if not isinstance(sub_agent_results, dict):
        raise TypeError(f"Expected dict for sub_agent_results, got {type(sub_agent_results)}")

    agent_results = get_agent_results(sub_agent_results)
    paper_title = document.metadata.title or "Unknown Paper"
    metadata_result = get_successful_result(sub_agent_results, "metadata")
    if paper_title == "Unknown Paper" and metadata_result is not None:
        if getattr(metadata_result, "title", None):
            paper_title = metadata_result.title

    successful_analyses = [name for name, result in agent_results.items() if result.get("success", False)]
    if not successful_analyses:
        logger.error("No successful analyses available for synthesis")
        return "No successful analyses available for report synthesis. Please try again with different settings."

    synthesis_agent = Agent(model=model, deps_type=dict, system_prompt=SYNTHESIS_SYSTEM_PROMPT, retries=max_retries)

    prompt = build_report_synthesis_prompt(
        paper_title=paper_title,
        source_path=str(document.source_path) if document.source_path else "text document",
        abstract=extract_abstract(document, logger),
        sub_agent_results=agent_results,
    )

    try:
        result = await asyncio.wait_for(
            synthesis_agent.run(prompt, deps={"successful_analyses": successful_analyses}),
            timeout=timeout,
        )
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        output_preview = str(result.output)[:200] + "..." if len(str(result.output)) > 200 else str(result.output)
        logger.debug(f"[synthesis_agent] Prompt ({len(prompt)} chars): {prompt_preview}")
        logger.debug(f"[synthesis_agent] Output ({len(str(result.output))} chars): {output_preview}")
        return result.output
    except TimeoutError:
        logger.error(f"Report synthesis timed out after {timeout} seconds")
        return f"Report synthesis timed out after {timeout} seconds. Please try again or use a shorter document."
    except Exception as exc:
        logger.error(f"Report synthesis failed: {exc}")
        return f"Report synthesis failed: {exc}. Please try again."
