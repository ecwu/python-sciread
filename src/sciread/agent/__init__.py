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
from .simple_agent import SimpleAgent, SimpleAnalysisResult
from .coordinate_agent import CoordinateAgent
from .react_agent import ReActAgent, analyze_document_with_react, load_and_process_document, get_initial_sections, format_status_summary
from .text_utils import remove_references

# CoordinateAgent result models
from .coordinate_agent import (
    MetadataExtractionResult,
    PreviousMethodsResult,
    ResearchQuestionsResult,
    MethodologyResult,
    ExperimentResult,
    FutureDirectionsResult,
    AnalysisPlan,
    ComprehensiveAnalysisResult,
)

# ReActAgent models
from .react_agent import ReActAgentInput, ReActAgentOutput

__all__ = [
    # Agent classes
    "SimpleAgent",
    "CoordinateAgent",
    "ReActAgent",

    # Result classes
    "SimpleAnalysisResult",

    # CoordinateAgent result models
    "MetadataExtractionResult",
    "PreviousMethodsResult",
    "ResearchQuestionsResult",
    "MethodologyResult",
    "ExperimentResult",
    "FutureDirectionsResult",
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",

    # ReActAgent functions and models
    "analyze_document_with_react",
    "load_and_process_document",
    "get_initial_sections",
    "format_status_summary",
    "ReActAgentInput",
    "ReActAgentOutput",

    # Text utilities
    "remove_references",
]

# Version of the agent module
__version__ = "2.0.0"