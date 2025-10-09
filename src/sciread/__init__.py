from .core import compute
from .llm_provider import get_model
from .logging_config import get_logger
from .logging_config import setup_logging
from .agents import (
    Agent,
    AgentConfig,
    AgentOrchestrator,
    AgentResult,
    MultiAgentSystem,
    SimpleAgent,
    ToolCallingAgent,
    analyze_document,
    create_agent,
    get_agent_recommendations,
    get_research_question_prompts,
    get_section_specific_prompt,
    get_simple_analysis_prompt,
)

__version__ = "0.0.0"

__all__ = [
    "compute",
    "get_logger",
    "get_model",
    "setup_logging",
    "Agent",
    "AgentConfig",
    "AgentOrchestrator",
    "AgentResult",
    "MultiAgentSystem",
    "SimpleAgent",
    "ToolCallingAgent",
    "analyze_document",
    "create_agent",
    "get_agent_recommendations",
    "get_research_question_prompts",
    "get_section_specific_prompt",
    "get_simple_analysis_prompt",
]
