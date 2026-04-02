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
    result = await agent.run_analysis(doc, "Summarize this paper")

    # Generate comprehensive analysis with multi-agent system
    comprehensive_result = await coordinate_agent.analyze(doc)

    # Generate iterative analysis with ReAct agent
    react_result = await react_agent.run_analysis(doc, "What are the main contributions?")

    # Generate discussion-based analysis with personality agents
    discussion_result = await discussion_agent.analyze_document(doc)
"""

from .coordinate import AnalysisPlan
from .coordinate import ComprehensiveAnalysisResult
from .coordinate import CoordinateAgent
from .coordinate import ExperimentResult
from .coordinate import FutureDirectionsResult
from .coordinate import MetadataExtractionResult
from .coordinate import MethodologyResult
from .coordinate import PreviousMethodsResult
from .coordinate import ResearchQuestionsResult
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
from .react import ReActAgent
from .react import ReActIterationOutput
from .react import analyze_file_with_react
from .react import analyze_file_with_react_sync
from .react import load_and_process_document
from .shared import remove_references
from .simple import SimpleAgent
from .simple import SimpleAnalysisResult
from .simple import analyze_file_with_simple
from .simple import analyze_file_with_simple_sync
from .simple import load_document_for_simple_analysis

__all__ = [
    "AgentInsight",
    "AgentPersonality",
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",
    "ConsensusBuilder",
    "ConsensusPoint",
    "CoordinateAgent",
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
    "ReActIterationOutput",
    "ResearchQuestionsResult",
    "Response",
    "SimpleAgent",
    "SimpleAnalysisResult",
    "Task",
    "TaskPriority",
    "TaskQueue",
    "TaskQueueManager",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "analyze_file_with_react",
    "analyze_file_with_react_sync",
    "analyze_file_with_simple",
    "analyze_file_with_simple_sync",
    "create_personality_agent",
    "load_and_process_document",
    "load_document_for_simple_analysis",
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"
