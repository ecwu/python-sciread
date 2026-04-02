"""Planning helpers for the coordinate agent."""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext

from ...document.document_renderers import choose_best_section_match
from ...document.document_renderers import format_section_choices
from ...document.document_renderers import get_section_length_map
from ...document.document_renderers import is_likely_heading_only
from ...document_structure import Document
from .models import AnalysisPlan
from .prompts import CONTROLLER_INSTRUCTIONS
from .prompts import build_analysis_planning_prompt
from .runtime import EXPERT_SECTION_PREFERENCES
from .runtime import CoordinateDeps
from .runtime import default_analysis_plan


def select_sections_for_expert(document: Document, analysis_type: str, planned_sections: list[str] | None) -> list[str]:
    """Choose the best-matching sections for a given expert."""
    available_sections = document.get_section_names()
    section_lengths = get_section_length_map(document, available_sections)

    if planned_sections:
        targets = planned_sections
    elif analysis_type in EXPERT_SECTION_PREFERENCES:
        targets = EXPERT_SECTION_PREFERENCES[analysis_type]
    else:
        targets = ["abstract", "introduction", "methodology", "results", "conclusion"]

    matched: list[str] = []
    for target in targets:
        match = document.get_closest_section_name(target, threshold=0.7)
        if match and is_likely_heading_only(section_lengths.get(match, 0)):
            preferred_match = choose_best_section_match(target, available_sections, section_lengths)
            if preferred_match:
                match = preferred_match

        if not match:
            match = choose_best_section_match(target, available_sections, section_lengths)

        if match and match not in matched:
            matched.append(match)

    if not matched:
        matched = [section for section in available_sections if not is_likely_heading_only(section_lengths.get(section, 0))][:3]
    if not matched:
        matched = available_sections[:3]

    return matched


def build_expert_content(
    document: Document,
    analysis_type: str,
    sections_to_analyze: list[str] | None,
    max_tokens: int | None,
) -> str:
    """Assemble expert-oriented document context."""
    section_names = select_sections_for_expert(document, analysis_type, sections_to_analyze)
    return document.get_for_llm(
        section_names=section_names,
        max_tokens=max_tokens,
        include_headers=True,
        clean_text=True,
        max_chars_per_section=2500,
    )


def extract_abstract(document: Document, logger) -> str:
    """Extract abstract-like content for controller planning."""
    logger.debug("Extracting abstract from document using semantic sections")

    section_names = document.get_section_names()
    for section_name in section_names:
        section_lower = section_name.lower()
        if any(keyword in section_lower for keyword in ["abstract", "summary", "overview"]):
            logger.debug(f"Found potential abstract section: '{section_name}'")
            abstract_chunks = document.get_sections_by_name([section_name])
            if abstract_chunks:
                abstract_content = " ".join(chunk.content for chunk in abstract_chunks)
                logger.info(f"Abstract extracted from semantic sections: {len(abstract_content)} characters")
                return abstract_content

    if len(document.chunks) >= 2:
        fallback_text = " ".join(chunk.content for chunk in document.chunks[:2])[:2000]
        logger.warning(f"No abstract section found, using first 2 chunks ({len(fallback_text)} chars) as fallback for analysis planning")
        return fallback_text

    if document.chunks:
        fallback_text = document.chunks[0].content[:2000]
        logger.warning(f"No abstract section found, using first chunk ({len(fallback_text)} chars) as fallback for analysis planning")
        return fallback_text

    text = document.get_full_text() if document.chunks else document.text
    if text:
        logger.warning("No abstract section found, using first 2000 chars of document as fallback for analysis planning")
        return text[:2000]

    logger.error("No abstract found and no text available in document")
    return ""


def create_planning_agent(model, max_retries: int, logger) -> Agent[CoordinateDeps, AnalysisPlan]:
    """Create the controller agent used for planning."""
    planning_agent = Agent(
        model=model,
        deps_type=CoordinateDeps,
        output_type=AnalysisPlan,
        retries=max_retries,
    )

    @planning_agent.system_prompt
    async def planning_system_prompt(ctx: RunContext[CoordinateDeps]) -> str:
        deps = ctx.deps
        abstract = extract_abstract(deps.document, logger)
        section_names = deps.document.get_section_names()

        if not abstract and not section_names:
            raise ModelRetry(
                "Document appears to have no extractable content or sections. "
                "Please ensure the document is properly processed and contains readable text."
            )

        section_lengths = get_section_length_map(deps.document, section_names)
        available_sections_text = format_section_choices(section_names, section_lengths)
        planning_prompt = build_analysis_planning_prompt(abstract, available_sections_text)
        return f"{CONTROLLER_INSTRUCTIONS}\n\n{planning_prompt}"

    return planning_agent


async def plan_analysis(
    document: Document,
    section_names: list[str],
    planning_agent: Agent[CoordinateDeps, AnalysisPlan],
    timeout: float,
    model_identifier: str,
    logger,
) -> AnalysisPlan:
    """Generate a controller plan for coordinate analysis."""
    abstract = extract_abstract(document, logger)

    if not abstract.strip():
        logger.warning("Empty abstract provided, using default comprehensive analysis plan")
        return default_analysis_plan("No abstract available, using comprehensive analysis plan with all sections")

    section_lengths = get_section_length_map(document, section_names)
    available_sections_text = format_section_choices(section_names, section_lengths)
    prompt = build_analysis_planning_prompt(abstract, available_sections_text)

    try:
        logger.debug(f"Calling controller agent with prompt length: {len(prompt)} chars")
        logger.debug(f"Controller agent model: {model_identifier}")

        result = await asyncio.wait_for(
            planning_agent.run("请生成分析计划", deps=CoordinateDeps(document=document)),
            timeout=timeout,
        )

        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        output_preview = str(result.output)[:200] + "..." if len(str(result.output)) > 200 else str(result.output)
        logger.debug(f"[controller_agent] Prompt ({len(prompt)} chars): {prompt_preview}")
        logger.debug(f"[controller_agent] Output ({len(str(result.output))} chars): {output_preview}")

        if not result.output.reasoning.strip():
            logger.warning("Controller agent returned plan with empty reasoning")

        return result.output
    except TimeoutError:
        logger.error(f"Analysis planning timed out after {timeout} seconds")
        return default_analysis_plan(f"Planning timed out after {timeout} seconds, using default comprehensive analysis with all sections")
    except Exception as exc:
        logger.error(f"Analysis planning failed: {exc}")
        return default_analysis_plan(f"Planning failed ({exc}), using default comprehensive analysis with all sections")
