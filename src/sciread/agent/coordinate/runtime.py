"""Shared runtime configuration for the coordinate agent."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

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
    custom_plan: AnalysisPlan | None = None
    max_retries: int = 3
    timeout: float = 300.0


@dataclass
class ExpertAgentDeps:
    """Dependencies for individual expert agents."""

    document: Any
    analysis_type: str
    sections_to_analyze: list[str] = field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None


EXPERT_AGENT_CONFIG = {
    "metadata": {
        "system_prompt": METADATA_EXTRACTION_SYSTEM_PROMPT,
        "output_type": MetadataExtractionResult,
        "timeout": 60.0,
        "prompt_builder": build_metadata_analysis_prompt,
    },
    "previous_methods": {
        "system_prompt": PREVIOUS_METHODS_SYSTEM_PROMPT,
        "output_type": PreviousMethodsResult,
        "timeout": 120.0,
        "prompt_builder": build_previous_methods_analysis_prompt,
    },
    "research_questions": {
        "system_prompt": RESEARCH_QUESTIONS_SYSTEM_PROMPT,
        "output_type": ResearchQuestionsResult,
        "timeout": 120.0,
        "prompt_builder": build_research_questions_analysis_prompt,
    },
    "methodology": {
        "system_prompt": METHODOLOGY_SYSTEM_PROMPT,
        "output_type": MethodologyResult,
        "timeout": 120.0,
        "prompt_builder": build_methodology_analysis_prompt,
    },
    "experiments": {
        "system_prompt": EXPERIMENTS_SYSTEM_PROMPT,
        "output_type": ExperimentResult,
        "timeout": 120.0,
        "prompt_builder": build_experiments_analysis_prompt,
    },
    "future_directions": {
        "system_prompt": FUTURE_DIRECTIONS_SYSTEM_PROMPT,
        "output_type": FutureDirectionsResult,
        "timeout": 120.0,
        "prompt_builder": build_future_directions_analysis_prompt,
    },
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


ANALYSIS_TASKS = [
    ("analyze_metadata", "metadata", "metadata", None),
    ("analyze_previous_methods", "previous_methods", "previous_methods_sections", "previous_methods_sections"),
    ("analyze_research_questions", "research_questions", "research_questions_sections", "research_questions_sections"),
    ("analyze_methodology", "methodology", "methodology_sections", "methodology_sections"),
    ("analyze_experiments", "experiments", "experiments_sections", "experiments_sections"),
    ("analyze_future_directions", "future_directions", "future_directions_sections", "future_directions_sections"),
]


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
