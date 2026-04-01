"""Synthesis helpers for the coordinate agent."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ...document_structure import Document
from .models import AnalysisPlan
from .models import ComprehensiveAnalysisResult
from .planner import extract_abstract
from .prompts import SYNTHESIS_SYSTEM_PROMPT
from .prompts import build_report_synthesis_prompt


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
    return {
        "total_agents_executed": len(sub_agent_results) - 1,
        "successful_agents": len(
            [result for name, result in sub_agent_results.items() if name != "_sections_analyzed" and result.get("success", False)]
        ),
        "failed_agents": len(
            [result for name, result in sub_agent_results.items() if name != "_sections_analyzed" and not result.get("success", False)]
        ),
        "agent_results": {
            name: {"success": data.get("success", False)}
            for name, data in sub_agent_results.items()
            if name != "_sections_analyzed"
        },
    }


def build_comprehensive_result(
    analysis_plan: AnalysisPlan,
    sub_agent_results: dict[str, Any],
    final_report: str,
    total_execution_time: float,
) -> ComprehensiveAnalysisResult:
    """Assemble the final coordinate-agent response model."""
    return ComprehensiveAnalysisResult(
        analysis_plan=analysis_plan,
        metadata_result=sub_agent_results.get("metadata", {}).get("result") if sub_agent_results.get("metadata", {}).get("success") else None,
        previous_methods_result=(
            sub_agent_results.get("previous_methods", {}).get("result") if sub_agent_results.get("previous_methods", {}).get("success") else None
        ),
        research_questions_result=(
            sub_agent_results.get("research_questions", {}).get("result")
            if sub_agent_results.get("research_questions", {}).get("success")
            else None
        ),
        methodology_result=sub_agent_results.get("methodology", {}).get("result") if sub_agent_results.get("methodology", {}).get("success") else None,
        experiment_result=sub_agent_results.get("experiments", {}).get("result") if sub_agent_results.get("experiments", {}).get("success") else None,
        future_directions_result=(
            sub_agent_results.get("future_directions", {}).get("result")
            if sub_agent_results.get("future_directions", {}).get("success")
            else None
        ),
        execution_summary=build_execution_summary(sub_agent_results),
        final_report=final_report,
        total_execution_time=total_execution_time,
        sections_analyzed=sub_agent_results.get("_sections_analyzed", {}),
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

    paper_title = document.metadata.title or "Unknown Paper"
    if paper_title == "Unknown Paper" and sub_agent_results.get("metadata", {}).get("success"):
        metadata_result = sub_agent_results["metadata"]["result"]
        if getattr(metadata_result, "title", None):
            paper_title = metadata_result.title

    successful_analyses = [
        name for name, result in sub_agent_results.items() if name != "_sections_analyzed" and result.get("success", False)
    ]
    if not successful_analyses:
        logger.error("No successful analyses available for synthesis")
        return "No successful analyses available for report synthesis. Please try again with different settings."

    synthesis_agent = Agent(model=model, deps_type=dict, system_prompt=SYNTHESIS_SYSTEM_PROMPT, retries=max_retries)

    @synthesis_agent.system_prompt
    async def synthesis_system_prompt(ctx: RunContext[dict]) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    prompt = build_report_synthesis_prompt(
        paper_title=paper_title,
        source_path=str(document.source_path) if document.source_path else "text document",
        abstract=extract_abstract(document, logger),
        sub_agent_results=sub_agent_results,
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
