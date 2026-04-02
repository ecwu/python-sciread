"""Shared runtime configuration for the coordinate agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pydantic import BaseModel

from .models import AnalysisPlan
from .models import ExperimentResult
from .models import FutureDirectionsResult
from .models import MetadataExtractionResult
from .models import MethodologyResult
from .models import PreviousMethodsResult
from .models import ResearchQuestionsResult
from .prompts import EXPERIMENTS_SYSTEM_PROMPT
from .prompts import FUTURE_DIRECTIONS_SYSTEM_PROMPT
from .prompts import METADATA_EXTRACTION_SYSTEM_PROMPT
from .prompts import METHODOLOGY_SYSTEM_PROMPT
from .prompts import PREVIOUS_METHODS_SYSTEM_PROMPT
from .prompts import RESEARCH_QUESTIONS_SYSTEM_PROMPT
from .prompts import build_experiments_analysis_prompt
from .prompts import build_future_directions_analysis_prompt
from .prompts import build_metadata_analysis_prompt
from .prompts import build_methodology_analysis_prompt
from .prompts import build_previous_methods_analysis_prompt
from .prompts import build_research_questions_analysis_prompt


@dataclass
class CoordinateDeps:
    """Dependencies for controller-side planning."""

    document: Any


@dataclass
class ExpertAgentDeps:
    """Dependencies for individual expert agents."""

    document: Any
    analysis_type: str
    sections_to_analyze: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpertAgentConfig:
    """Static configuration for one expert agent."""

    system_prompt: str
    output_type: type[BaseModel]
    prompt_builder: Callable[[str], str]


@dataclass(frozen=True)
class AnalysisTask:
    """Metadata for one coordinate-agent task."""

    plan_field: str
    analysis_type: str
    output_field: str
    sections_field: str | None = None


EXPERT_AGENT_CONFIG: dict[str, ExpertAgentConfig] = {
    "metadata": ExpertAgentConfig(
        system_prompt=METADATA_EXTRACTION_SYSTEM_PROMPT,
        output_type=MetadataExtractionResult,
        prompt_builder=build_metadata_analysis_prompt,
    ),
    "previous_methods": ExpertAgentConfig(
        system_prompt=PREVIOUS_METHODS_SYSTEM_PROMPT,
        output_type=PreviousMethodsResult,
        prompt_builder=build_previous_methods_analysis_prompt,
    ),
    "research_questions": ExpertAgentConfig(
        system_prompt=RESEARCH_QUESTIONS_SYSTEM_PROMPT,
        output_type=ResearchQuestionsResult,
        prompt_builder=build_research_questions_analysis_prompt,
    ),
    "methodology": ExpertAgentConfig(
        system_prompt=METHODOLOGY_SYSTEM_PROMPT,
        output_type=MethodologyResult,
        prompt_builder=build_methodology_analysis_prompt,
    ),
    "experiments": ExpertAgentConfig(
        system_prompt=EXPERIMENTS_SYSTEM_PROMPT,
        output_type=ExperimentResult,
        prompt_builder=build_experiments_analysis_prompt,
    ),
    "future_directions": ExpertAgentConfig(
        system_prompt=FUTURE_DIRECTIONS_SYSTEM_PROMPT,
        output_type=FutureDirectionsResult,
        prompt_builder=build_future_directions_analysis_prompt,
    ),
}


EXPERT_SECTION_PREFERENCES = {
    "metadata": ["abstract", "introduction", "title"],
    "methodology": [
        "methodology",
        "method",
        "approach",
        "design",
        "technical approach",
    ],
    "experiments": [
        "experiments",
        "experimental setup",
        "evaluation",
        "study design",
        "case study",
    ],
    "evaluation": ["results", "evaluation", "findings", "outcomes", "performance"],
    "contributions": ["introduction", "contributions", "novelty", "innovation"],
    "limitations": ["limitations", "discussion", "conclusion", "future work"],
}


ANALYSIS_TASKS: tuple[AnalysisTask, ...] = (
    AnalysisTask(plan_field="analyze_metadata", analysis_type="metadata", output_field="metadata_result"),
    AnalysisTask(
        plan_field="analyze_previous_methods",
        analysis_type="previous_methods",
        output_field="previous_methods_result",
        sections_field="previous_methods_sections",
    ),
    AnalysisTask(
        plan_field="analyze_research_questions",
        analysis_type="research_questions",
        output_field="research_questions_result",
        sections_field="research_questions_sections",
    ),
    AnalysisTask(
        plan_field="analyze_methodology",
        analysis_type="methodology",
        output_field="methodology_result",
        sections_field="methodology_sections",
    ),
    AnalysisTask(
        plan_field="analyze_experiments",
        analysis_type="experiments",
        output_field="experiment_result",
        sections_field="experiments_sections",
    ),
    AnalysisTask(
        plan_field="analyze_future_directions",
        analysis_type="future_directions",
        output_field="future_directions_result",
        sections_field="future_directions_sections",
    ),
)

INTERNAL_SECTIONS_KEY = "_sections_analyzed"


def default_analysis_plan(reasoning: str) -> AnalysisPlan:
    """Build the default all-on fallback plan."""
    return AnalysisPlan(
        analyze_metadata=True,
        analyze_previous_methods=True,
        analyze_research_questions=True,
        analyze_methodology=True,
        analyze_experiments=True,
        analyze_future_directions=True,
        previous_methods_sections=[],
        research_questions_sections=[],
        methodology_sections=[],
        experiments_sections=[],
        future_directions_sections=[],
        reasoning=reasoning,
    )
