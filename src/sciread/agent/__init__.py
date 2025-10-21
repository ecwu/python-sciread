"""Agent module for sciread package.

This module provides document analysis agents using pydantic-ai framework.
The agents can process academic papers and generate comprehensive reports
based on document content and custom prompts.

Main Interface:
    SimpleAgent - Simple agent class for basic document analysis
    CoordinateAgent - Multi-agent controller with expert sub-agents
    ReActAgent - Reasoning and Acting agent for iterative analysis
    DiscussionAgent - Multi-agent discussion system with personality-driven analysis

Example Usage:
    from sciread.agent import SimpleAgent, CoordinateAgent, ReActAgent, DiscussionAgent
    from sciread.document import Document

    # Create a simple agent
    agent = SimpleAgent("deepseek/deepseek-chat")

    # Create a multi-agent system
    coordinate_agent = CoordinateAgent("deepseek/deepseek-chat")

    # Create a ReAct agent
    react_agent = ReActAgent("deepseek/deepseek-chat")

    # Create a discussion-based multi-agent system
    discussion_agent = DiscussionAgent("deepseek/deepseek-chat")

    # Process a document (automatically loaded and split)
    doc = Document.from_file("paper.pdf", to_markdown=True)

    # Generate analysis report with simple agent
    result = await agent.analyze(doc, "Summarize this paper")

    # Generate comprehensive analysis with multi-agent system
    comprehensive_result = await coordinate_agent.analyze(doc)

    # Generate iterative analysis with ReAct agent
    react_result = await react_agent.analyze_document(doc, "What are the main contributions?")

    # Generate discussion-based analysis with personality agents
    discussion_result = await discussion_agent.analyze_document(doc)
"""

# New agent imports
from .coordinate_agent import AnalysisPlan
from .coordinate_agent import ComprehensiveAnalysisResult
from .coordinate_agent import CoordinateAgent
from .coordinate_agent import ExperimentResult
from .coordinate_agent import FutureDirectionsResult

# CoordinateAgent result models
from .coordinate_agent import MetadataExtractionResult
from .coordinate_agent import MethodologyResult
from .coordinate_agent import PreviousMethodsResult
from .coordinate_agent import ResearchQuestionsResult

# Discussion system imports
from .discussion_agent import DiscussionAgent
from .consensus_builder import ConsensusBuilder
from .personality_agents import PersonalityAgent, create_personality_agent
from .task_queue import TaskQueueManager
from .models.discussion_models import (
    AgentPersonality, DiscussionState, AgentInsight, Question, Response,
    DiscussionResult, ConsensusPoint, DivergentView
)
from .models.task_models import (
    Task, TaskType, TaskPriority, TaskStatus, TaskResult, TaskQueue
)

# ReActAgent models
from .react_agent import ReActAgent
from .react_agent import ReActAgentInput
from .react_agent import ReActAgentOutput
from .react_agent import analyze_document_with_react
from .react_agent import format_status_summary
from .react_agent import get_initial_sections
from .react_agent import load_and_process_document
from .simple_agent import SimpleAgent
from .simple_agent import SimpleAnalysisResult
from .text_utils import remove_references

__all__ = [
    # Existing agents
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",
    "CoordinateAgent",
    "ExperimentResult",
    "FutureDirectionsResult",
    "MetadataExtractionResult",
    "MethodologyResult",
    "PreviousMethodsResult",
    "ReActAgent",
    "ReActAgentInput",
    "ReActAgentOutput",
    "ResearchQuestionsResult",
    "SimpleAgent",
    "SimpleAnalysisResult",

    # New discussion system
    "DiscussionAgent",
    "ConsensusBuilder",
    "PersonalityAgent",
    "create_personality_agent",
    "TaskQueueManager",

    # Discussion models
    "AgentPersonality",
    "DiscussionState",
    "AgentInsight",
    "Question",
    "Response",
    "DiscussionResult",
    "ConsensusPoint",
    "DivergentView",

    # Task models
    "Task",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "TaskResult",
    "TaskQueue",

    # Utility functions
    "analyze_document_with_react",
    "format_status_summary",
    "get_initial_sections",
    "load_and_process_document",
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"
