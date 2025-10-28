"""Agent module for sciread package.

This module provides document analysis agents using pydantic-ai framework.
The agents can process academic papers and generate comprehensive reports
based on document content and custom prompts.

Main Interface:
    SimpleAgent - Simple agent class for basic document analysis
    CoordinateAgent - Multi-agent controller with expert sub-agents
    ReActAgent - Reasoning and Acting agent for iterative analysis
    RAGReActAgent - Retrieval-Augmented Generation + ReAct agent for semantic search-based analysis
    DiscussionAgent - Multi-agent discussion system with personality-driven analysis

Example Usage:
    from sciread.agent import SimpleAgent, CoordinateAgent, ReActAgent, RAGReActAgent, DiscussionAgent
    from sciread.document import Document

    # Create a simple agent
    agent = SimpleAgent("deepseek/deepseek-chat")

    # Create a multi-agent system
    coordinate_agent = CoordinateAgent("deepseek/deepseek-chat")

    # Create a ReAct agent
    react_agent = ReActAgent("deepseek/deepseek-chat")

    # Create a RAG ReAct agent
    rag_react_agent = RAGReActAgent("deepseek/deepseek-chat")

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

    # Generate semantic search-based analysis with RAG ReAct agent
    rag_react_result = await rag_react_agent.analyze_document(doc, "What are the main contributions?")

    # Generate discussion-based analysis with personality agents
    discussion_result = await discussion_agent.analyze_document(doc)
"""

# Agent classes
from .consensus_builder import ConsensusBuilder
from .coordinate_agent import CoordinateAgent

# Discussion system imports
from .discussion_agent import DiscussionAgent

# CoordinateAgent models - now imported from models folder
from .models.coordinate_models import AnalysisPlan
from .models.coordinate_models import ComprehensiveAnalysisResult
from .models.coordinate_models import ExperimentResult
from .models.coordinate_models import FutureDirectionsResult
from .models.coordinate_models import MetadataExtractionResult
from .models.coordinate_models import MethodologyResult
from .models.coordinate_models import PreviousMethodsResult
from .models.coordinate_models import ResearchQuestionsResult
from .models.discussion_models import AgentInsight
from .models.discussion_models import AgentPersonality
from .models.discussion_models import ConsensusPoint
from .models.discussion_models import DiscussionResult
from .models.discussion_models import DiscussionState
from .models.discussion_models import DivergentView
from .models.discussion_models import Question
from .models.discussion_models import Response

# ReActAgent models - now imported from models folder
from .models.react_models import ReActAgentInput
from .models.react_models import ReActAgentOutput

# RAG ReActAgent models - now imported from models folder
from .models.rag_react_models import RAGReActAgentInput
from .models.rag_react_models import RAGReActAgentOutput

# SimpleAgent models - now imported from models folder
from .models.simple_models import SimpleAnalysisResult
from .models.task_models import Task
from .models.task_models import TaskPriority
from .models.task_models import TaskQueue
from .models.task_models import TaskResult
from .models.task_models import TaskStatus
from .models.task_models import TaskType
from .personality_agents import PersonalityAgent
from .personality_agents import create_personality_agent

# ReActAgent and utility functions
from .react_agent import ReActAgent
from .react_agent import analyze_document_with_react
from .react_agent import format_status_summary
from .react_agent import get_initial_sections
from .react_agent import load_and_process_document

# RAG ReActAgent and utility functions
from .rag_react_agent import RAGReActAgent
from .rag_react_agent import analyze_document_with_rag_react
from .simple_agent import SimpleAgent
from .task_queue import TaskQueueManager
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
    "RAGReActAgent",
    "RAGReActAgentInput",
    "RAGReActAgentOutput",
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
    "analyze_document_with_rag_react",
    "analyze_document_with_react",
    "format_status_summary",
    "get_initial_sections",
    "load_and_process_document",
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"
