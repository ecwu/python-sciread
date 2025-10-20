"""Prompts module for sciread agents.

This module provides centralized access to all prompts used by different
agent types in the sciread system.
"""

# SimpleAgent prompts
from .simple import DEFAULT_SYSTEM_PROMPT, build_analysis_prompt

# CoordinateAgent prompts
from .coordinate import (
    METADATA_EXTRACTION_SYSTEM_PROMPT,
    PREVIOUS_METHODS_SYSTEM_PROMPT,
    RESEARCH_QUESTIONS_SYSTEM_PROMPT,
    METHODOLOGY_SYSTEM_PROMPT,
    EXPERIMENTS_SYSTEM_PROMPT,
    FUTURE_DIRECTIONS_SYSTEM_PROMPT,
    CONTROLLER_INSTRUCTIONS,
    SYNTHESIS_SYSTEM_PROMPT,
    build_metadata_analysis_prompt,
    build_previous_methods_analysis_prompt,
    build_research_questions_analysis_prompt,
    build_methodology_analysis_prompt,
    build_experiments_analysis_prompt,
    build_future_directions_analysis_prompt,
    build_analysis_planning_prompt,
    build_report_synthesis_prompt,
    build_generic_analysis_prompt,
)

# ReActAgent prompts
from .react import SYSTEM_PROMPT, format_agent_prompt

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