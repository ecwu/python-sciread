"""Execution helpers for coordinate expert agents."""

from __future__ import annotations

import asyncio
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
from .runtime import ExpertAgentDeps


def create_expert_agent(model, max_retries: int, analysis_type: str, logger) -> Agent[ExpertAgentDeps, BaseModel]:
    """Create an expert agent for one analysis type."""
    if analysis_type not in EXPERT_AGENT_CONFIG:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    config = EXPERT_AGENT_CONFIG[analysis_type]
    agent = Agent(
        model=model,
        deps_type=ExpertAgentDeps,
        output_type=config["output_type"],
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
            return config["prompt_builder"](content)
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
    tasks = []
    sections_analyzed: dict[str, list[str]] = {}

    for plan_field, agent_key, result_key, sections_field in ANALYSIS_TASKS:
        if not getattr(analysis_plan, plan_field):
            continue

        if agent_key == "metadata":
            sections_analyzed[result_key] = [f"First {min(3, len(document.chunks))} chunks"]
        else:
            sections = getattr(analysis_plan, sections_field)
            if sections:
                sections_analyzed[result_key] = deduplicate_sections(sections)
                logger.debug(f"{agent_key} agent using specific sections: {sections_analyzed[result_key]}")
            else:
                sections_analyzed[result_key] = ["All sections"]
                logger.warning(f"{agent_key} agent: No sections specified, using 'All sections' fallback.")

        agent = create_expert_agent(model, max_retries, agent_key, logger)
        deps = ExpertAgentDeps(
            document=document,
            analysis_type=agent_key,
            sections_to_analyze=sections_analyzed[result_key],
            analysis_plan=analysis_plan,
        )
        task = asyncio.create_task(safe_execute_agent(agent.run("请执行专家分析", deps=deps), agent_key, logger))
        tasks.append((result_key, task))

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

    results["_sections_analyzed"] = sections_analyzed
    return results
