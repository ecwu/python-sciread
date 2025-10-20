"""Prompts module for sciread agents.

This module provides centralized access to all prompts used by different
agent types in the sciread system.
"""

# SimpleAgent prompts
from .coordinate import CONTROLLER_INSTRUCTIONS
from .coordinate import EXPERIMENTS_SYSTEM_PROMPT
from .coordinate import FUTURE_DIRECTIONS_SYSTEM_PROMPT

# CoordinateAgent prompts
from .coordinate import METADATA_EXTRACTION_SYSTEM_PROMPT
from .coordinate import METHODOLOGY_SYSTEM_PROMPT
from .coordinate import PREVIOUS_METHODS_SYSTEM_PROMPT
from .coordinate import RESEARCH_QUESTIONS_SYSTEM_PROMPT
from .coordinate import SYNTHESIS_SYSTEM_PROMPT
from .coordinate import build_analysis_planning_prompt
from .coordinate import build_experiments_analysis_prompt
from .coordinate import build_future_directions_analysis_prompt
from .coordinate import build_generic_analysis_prompt
from .coordinate import build_metadata_analysis_prompt
from .coordinate import build_methodology_analysis_prompt
from .coordinate import build_previous_methods_analysis_prompt
from .coordinate import build_report_synthesis_prompt
from .coordinate import build_research_questions_analysis_prompt

# ReActAgent prompts
from .react import SYSTEM_PROMPT
from .react import format_agent_prompt
from .simple import DEFAULT_SYSTEM_PROMPT
from .simple import build_analysis_prompt

__all__ = [
    # SimpleAgent
    "DEFAULT_SYSTEM_PROMPT",
    "build_analysis_prompt",
    # CoordinateAgent
    "METADATA_EXTRACTION_SYSTEM_PROMPT",
    "PREVIOUS_METHODS_SYSTEM_PROMPT",
    "RESEARCH_QUESTIONS_SYSTEM_PROMPT",
    "METHODOLOGY_SYSTEM_PROMPT",
    "EXPERIMENTS_SYSTEM_PROMPT",
    "FUTURE_DIRECTIONS_SYSTEM_PROMPT",
    "CONTROLLER_INSTRUCTIONS",
    "SYNTHESIS_SYSTEM_PROMPT",
    "build_metadata_analysis_prompt",
    "build_previous_methods_analysis_prompt",
    "build_research_questions_analysis_prompt",
    "build_methodology_analysis_prompt",
    "build_experiments_analysis_prompt",
    "build_future_directions_analysis_prompt",
    "build_analysis_planning_prompt",
    "build_report_synthesis_prompt",
    "build_generic_analysis_prompt",
    # ReActAgent
    "SYSTEM_PROMPT",
    "format_agent_prompt",
]
