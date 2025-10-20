"""Agent module for sciread package.

This module provides document analysis agents using pydantic-ai framework.
The agents can process academic papers and generate comprehensive reports
based on document content and custom prompts.

Main Interface:
    SimpleAgent - Simple agent class for basic document analysis
    CoordinateAgent - Multi-agent controller with expert sub-agents
    ReActAgent - Reasoning and Acting agent for iterative analysis

Example Usage:
    from sciread.agent import SimpleAgent, CoordinateAgent, ReActAgent
    from sciread.document import Document

    # Create a simple agent
    agent = SimpleAgent("deepseek/deepseek-chat")

    # Create a multi-agent system
    coordinate_agent = CoordinateAgent("deepseek/deepseek-chat")

    # Create a ReAct agent
    react_agent = ReActAgent("deepseek/deepseek-chat")

    # Process a document (automatically loaded and split)
    doc = Document.from_file("paper.pdf", to_markdown=True)

    # Generate analysis report with simple agent
    result = await agent.analyze(doc, "Summarize this paper")

    # Generate comprehensive analysis with multi-agent system
    comprehensive_result = await coordinate_agent.analyze(doc)

    # Generate iterative analysis with ReAct agent
    react_result = await react_agent.analyze_document(doc, "What are the main contributions?")
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
from .react_agent import ReActAgent

# ReActAgent models
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
    "analyze_document_with_react",
    "format_status_summary",
    "get_initial_sections",
    "load_and_process_document",
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"
