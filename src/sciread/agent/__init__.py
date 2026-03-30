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

from .coordinate_agent import CoordinateAgent
from .discussion import AgentInsight
from .discussion import AgentPersonality
from .discussion import ConsensusBuilder
from .discussion import ConsensusPoint
from .discussion import DiscussionAgent
from .discussion import DiscussionResult
from .discussion import DiscussionState
from .discussion import DivergentView
from .discussion import PersonalityAgent
from .discussion import Question
from .discussion import Response
from .discussion import Task
from .discussion import TaskPriority
from .discussion import TaskQueue
from .discussion import TaskQueueManager
from .discussion import TaskResult
from .discussion import TaskStatus
from .discussion import TaskType
from .discussion import create_personality_agent

# CoordinateAgent models - now imported from models folder
from .models.coordinate_models import AnalysisPlan
from .models.coordinate_models import ComprehensiveAnalysisResult
from .models.coordinate_models import ExperimentResult
from .models.coordinate_models import FutureDirectionsResult
from .models.coordinate_models import MetadataExtractionResult
from .models.coordinate_models import MethodologyResult
from .models.coordinate_models import PreviousMethodsResult
from .models.coordinate_models import ResearchQuestionsResult

# ReActAgent models - now imported from models folder
from .models.react_models import ReActAgentInput
from .models.react_models import ReActAgentOutput

# SimpleAgent models - now imported from models folder
from .models.simple_models import SimpleAnalysisResult

# ReActAgent and utility functions
from .react_agent import ReActAgent
from .react_agent import analyze_document_with_react
from .react_agent import format_status_summary
from .react_agent import get_initial_sections
from .react_agent import load_and_process_document
from .simple_agent import SimpleAgent
from .text_utils import remove_references

__all__ = [
    "AgentInsight",
    # Discussion models
    "AgentPersonality",
    # Existing agents
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",
    "ConsensusBuilder",
    "ConsensusPoint",
    "CoordinateAgent",
    # New discussion system
    "DiscussionAgent",
    "DiscussionResult",
    "DiscussionState",
    "DivergentView",
    "ExperimentResult",
    "FutureDirectionsResult",
    "MetadataExtractionResult",
    "MethodologyResult",
    "PersonalityAgent",
    "PreviousMethodsResult",
    "Question",
    "ReActAgent",
    "ReActAgentInput",
    "ReActAgentOutput",
    "ResearchQuestionsResult",
    "Response",
    "SimpleAgent",
    "SimpleAnalysisResult",
    # Task models
    "Task",
    "TaskPriority",
    "TaskQueue",
    "TaskQueueManager",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    # Utility functions
    "analyze_document_with_react",
    "create_personality_agent",
    "format_status_summary",
    "get_initial_sections",
    "load_and_process_document",
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"
