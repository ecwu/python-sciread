"""Agent module for analyzing academic papers with Pydantic AI agents.

This module provides three different agent approaches for document analysis, all built
using Pydantic AI's Agent class:

1. SimpleAgent: Direct full document processing with single LLM calls using the Feynman technique
2. ToolCallingAgent: Controller-based system with specialized tools for different sections
3. MultiAgentSystem: Collaborative agents for high-level research question analysis

All agents integrate with the existing document module and LLM provider system.
"""

from .factory import AgentOrchestrator
from .factory import analyze_document
from .factory import create_agent_analysis
from .factory import get_agent_recommendations
from .multi_agent_system import analyze_document_with_multi_agent
from .multi_agent_system import analyze_with_agent_details
from .multi_agent_system import coordinator_agent
from .multi_agent_system import create_coordinator_agent
from .multi_agent_system import create_research_question_agent
from .prompts import (
    get_collaborative_agent_prompt,
    get_final_synthesis_prompt,
    get_research_question_prompts,
    get_section_specific_prompt,
    get_simple_analysis_prompt,
    get_synthesis_prompt,
    remove_citations_section,
)
from .schemas import AgentError
from .schemas import ContributionAnalysis
from .schemas import DocumentAnalysisResult
from .schemas import DocumentDeps
from .schemas import DocumentMetadata
from .schemas import ResearchQuestionAnalysis
from .schemas import SectionAnalysis
from .schemas import SimpleAnalysisResult
from .simple_agent import analyze_document_simple
from .simple_agent import create_simple_agent
from .simple_agent import simple_agent
from .simple_agent import simple_agent_claude
from .simple_agent import simple_agent_gpt4
from .tool_calling_agent import analyze_document_with_sections
from .tool_calling_agent import create_tool_calling_agent
from .tool_calling_agent import tool_calling_agent
from .tool_calling_agent import tool_calling_agent_claude
from .tool_calling_agent import tool_calling_agent_gpt4

__all__ = [
    # Core agent creation functions
    "create_simple_agent",
    "create_tool_calling_agent",
    "create_coordinator_agent",
    "create_research_question_agent",

    # Pre-configured agent instances
    "simple_agent",
    "simple_agent_gpt4",
    "simple_agent_claude",
    "tool_calling_agent",
    "tool_calling_agent_gpt4",
    "tool_calling_agent_claude",
    "coordinator_agent",

    # Analysis functions
    "analyze_document_simple",
    "analyze_document_with_sections",
    "analyze_document_with_multi_agent",
    "analyze_with_agent_details",

    # Factory and orchestration
    "analyze_document",
    "create_agent_analysis",
    "get_agent_recommendations",
    "AgentOrchestrator",

    # Prompts
    "get_simple_analysis_prompt",
    "get_section_specific_prompt",
    "get_synthesis_prompt",
    "get_collaborative_agent_prompt",
    "get_final_synthesis_prompt",
    "get_research_question_prompts",
    "remove_citations_section",

    # Schemas and data models
    "DocumentDeps",
    "SimpleAnalysisResult",
    "DocumentAnalysisResult",
    "SectionAnalysis",
    "ResearchQuestionAnalysis",
    "ContributionAnalysis",
    "DocumentMetadata",
    "AgentError",
]