"""Coordinate-agent orchestration for comprehensive paper analysis."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ...document_structure import Document
from ...llm_provider import get_model
from ...platform.logging import get_logger
from .executor import execute_sub_agents
from .models import AnalysisPlan
from .models import ComprehensiveAnalysisResult
from .planner import create_planning_agent
from .planner import extract_abstract
from .planner import plan_analysis
from .synthesis import build_comprehensive_result
from .synthesis import synthesize_report
from .synthesis import validate_pdf_document


class CoordinateAgent:
    """Coordinate expert agents for comprehensive academic paper analysis."""

    def __init__(
        self,
        model: str | OpenAIChatModel | AnthropicModel,
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        self.planning_agent = create_planning_agent(self.model, self.max_retries, self.logger)
        self.logger.debug("CoordinateAgent controller initialized successfully")

    def extract_abstract(self, document: Document) -> str:
        """Extract abstract-like planning context from the document."""
        return extract_abstract(document, self.logger)

    async def plan_analysis(self, document: Document, section_names: list[str]) -> AnalysisPlan:
        """Generate the analysis plan for expert execution."""
        return await plan_analysis(
            document=document,
            section_names=section_names,
            planning_agent=self.planning_agent,
            timeout=self.timeout,
            model_identifier=self.model_identifier,
            logger=self.logger,
        )

    async def execute_sub_agents(
        self,
        document: Document,
        analysis_plan: AnalysisPlan,
    ) -> dict[str, Any]:
        """Execute enabled expert agents."""
        return await execute_sub_agents(
            document=document,
            analysis_plan=analysis_plan,
            model=self.model,
            max_retries=self.max_retries,
            logger=self.logger,
        )

    async def synthesize_report(
        self,
        analysis_plan: AnalysisPlan,
        sub_agent_results: dict[str, Any],
        document: Document,
    ) -> str:
        """Synthesize the final comprehensive report."""
        return await synthesize_report(
            analysis_plan=analysis_plan,
            sub_agent_results=sub_agent_results,
            document=document,
            model=self.model,
            max_retries=self.max_retries,
            timeout=self.timeout,
            logger=self.logger,
        )

    async def analyze(
        self,
        document: Document,
        custom_plan: AnalysisPlan | None = None,
        **_: Any,
    ) -> ComprehensiveAnalysisResult:
        """Run the full coordinate-agent workflow on a PDF document."""
        validate_pdf_document(document)

        self.logger.info(f"Starting comprehensive document analysis: {document.source_path}")
        start_time = asyncio.get_event_loop().time()

        try:
            section_names = document.get_section_names()
            self.logger.debug(f"Found {len(section_names)} sections: {section_names}")

            analysis_plan = custom_plan or await self.plan_analysis(document, section_names)
            sub_agent_results = await self.execute_sub_agents(document, analysis_plan)
            final_report = await self.synthesize_report(analysis_plan, sub_agent_results, document)

            total_execution_time = asyncio.get_event_loop().time() - start_time
            return build_comprehensive_result(
                analysis_plan=analysis_plan,
                sub_agent_results=sub_agent_results,
                final_report=final_report,
                total_execution_time=total_execution_time,
            )
        except Exception as exc:
            self.logger.error(f"Comprehensive document analysis failed: {exc}")
            raise

    def __repr__(self) -> str:
        return f"CoordinateAgent(model={self.model_identifier})"
