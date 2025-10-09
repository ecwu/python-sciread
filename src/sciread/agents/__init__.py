"""Agent module for analyzing academic papers with LLM-driven agents.

This module provides three different agent approaches for document analysis:

1. SimpleAgent: Direct full document processing with single LLM calls
2. ToolCallingAgent: Controller-based system with specialized sub-agents for different sections
3. MultiAgentSystem: Collaborative agents for high-level research question analysis

All agents integrate with the existing document module and LLM provider system.
"""

from .base import Agent, AgentResult, AgentConfig
from .factory import AgentOrchestrator, analyze_document, create_agent, get_agent_recommendations
from .multi_agent_system import MultiAgentSystem
from .prompts import (
    get_simple_analysis_prompt,
    get_section_specific_prompt,
    get_research_question_prompts,
)
from .simple_agent import SimpleAgent
from .tool_calling_agent import ToolCallingAgent

__all__ = [
    "Agent",
    "AgentResult",
    "AgentConfig",
    "SimpleAgent",
    "ToolCallingAgent",
    "MultiAgentSystem",
    "get_simple_analysis_prompt",
    "get_section_specific_prompt",
    "get_research_question_prompts",
    "create_agent",
    "analyze_document",
    "get_agent_recommendations",
    "AgentOrchestrator",
]