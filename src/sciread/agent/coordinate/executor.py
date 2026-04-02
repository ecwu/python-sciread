"""Execution helpers for coordinate expert agents."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext

from ...document_structure import Document
from .models import AnalysisPlan
from .planner import build_expert_content
from .runtime import ANALYSIS_TASKS
from .runtime import EXPERT_AGENT_CONFIG
from .runtime import INTERNAL_SECTIONS_KEY
from .runtime import ExpertAgentDeps


def create_expert_agent(model, max_retries: int, analysis_type: str, logger) -> Agent[ExpertAgentDeps, BaseModel]:
    """Create an expert agent for one analysis type."""
    if analysis_type not in EXPERT_AGENT_CONFIG:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    config = EXPERT_AGENT_CONFIG[analysis_type]
    agent = Agent(
        model=model,
        deps_type=ExpertAgentDeps,
        output_type=config.output_type,
        system_prompt=config.system_prompt,
        retries=max_retries,
    )

    @agent.system_prompt
    async def expert_system_prompt(ctx: RunContext[ExpertAgentDeps]) -> str:
        deps = ctx.deps

        try:
            content = build_expert_content(
                document=deps.document,
                analysis_type=deps.analysis_type,
                sections_to_analyze=deps.sections_to_analyze or None,
                max_tokens=6000,
            )
            if not content.strip():
                raise ModelRetry(f"No content found for {deps.analysis_type} analysis. Please check document sections and try again.")
            return config.prompt_builder(content)
        except Exception as exc:
            logger.warning(f"Expert content assembly failed for {deps.analysis_type}: {exc}")
            raise ModelRetry(f"Unable to assemble content for {deps.analysis_type}.") from exc

    return agent


def deduplicate_sections(sections: list[str]) -> list[str]:
    """Remove duplicate section names while preserving order."""
    return list(dict.fromkeys(sections))


async def safe_execute_agent(coro, agent_name: str, logger):
    """Safely execute an agent run and unwrap its output."""
    try:
        result = await coro
        return result.output
    except Exception as exc:
        logger.error(f"Agent {agent_name} execution failed: {exc}")
        raise


def resolve_sections_to_analyze(
    document: Document,
    analysis_plan: AnalysisPlan,
    task,
    logger,
) -> list[str]:
    """Resolve the section names passed to a given expert."""
    if task.sections_field is None:
        return [f"First {min(3, len(document.chunks))} chunks"]

    selected_sections = deduplicate_sections(getattr(analysis_plan, task.sections_field))
    if selected_sections:
        logger.debug(f"{task.analysis_type} agent using specific sections: {selected_sections}")
        return selected_sections

    logger.warning(f"{task.analysis_type} agent: No sections specified, using 'All sections' fallback.")
    return ["All sections"]


async def execute_sub_agents(
    document: Document,
    analysis_plan: AnalysisPlan,
    model,
    max_retries: int,
    logger,
) -> dict[str, Any]:
    """Execute enabled expert agents in parallel."""
    start_time = asyncio.get_event_loop().time()

    results: dict[str, Any] = {}
    tasks: list[tuple[str, Awaitable[BaseModel]]] = []
    sections_analyzed: dict[str, list[str]] = {}

    for task in ANALYSIS_TASKS:
        if not getattr(analysis_plan, task.plan_field):
            continue

        sections_to_analyze = resolve_sections_to_analyze(document, analysis_plan, task, logger)
        sections_analyzed[task.analysis_type] = sections_to_analyze

        agent = create_expert_agent(model, max_retries, task.analysis_type, logger)
        deps = ExpertAgentDeps(
            document=document,
            analysis_type=task.analysis_type,
            sections_to_analyze=sections_to_analyze,
        )
        agent_task = asyncio.create_task(safe_execute_agent(agent.run("请执行专家分析", deps=deps), task.analysis_type, logger))
        tasks.append((task.analysis_type, agent_task))

    if not tasks:
        logger.warning("No agents selected for execution")
        return results

    completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    for (agent_name, _), result in zip(tasks, completed_tasks, strict=True):
        if isinstance(result, Exception):
            logger.error(f"Agent {agent_name} failed: {result}")
            results[agent_name] = {"error": str(result), "success": False}
        else:
            logger.debug(f"Agent {agent_name} completed successfully")
            results[agent_name] = {"result": result, "success": True}

    execution_time = asyncio.get_event_loop().time() - start_time
    logger.debug(f"Sub-agent execution completed in {execution_time:.2f} seconds")

    results[INTERNAL_SECTIONS_KEY] = sections_analyzed
    return results
