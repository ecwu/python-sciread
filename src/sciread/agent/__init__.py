"""Agent module for sciread package.

This module provides document analysis agents using pydantic-ai framework.
The agents can process academic papers and generate comprehensive reports
based on document content and custom prompts.

Main Interface:
    DocumentAgent - Main agent class for document analysis
    ToolAgent - Multi-agent controller with expert sub-agents
    create_agent - Factory function to create configured agents

Example Usage:
    from sciread.agent import create_agent, ToolAgent
    from sciread.document import Document

    # Create a simple agent
    agent = create_agent("deepseek/deepseek-chat")

    # Create a multi-agent system
    tool_agent = ToolAgent("deepseek/deepseek-chat")

    # Process a document (automatically loaded and split)
    doc = Document.from_file("paper.pdf", to_markdown=True)

    # Generate analysis report with simple agent
    result = await agent.analyze_document(doc, "Summarize this paper")

    # Generate comprehensive analysis with multi-agent system
    comprehensive_result = await tool_agent.analyze_document(doc)
"""

from .document_agent import DocumentAgent
from .document_agent import DocumentAnalysisResult
from .react_agent import ReActAgent
from .react_agent import analyze_document_with_react
from .react_agent import load_and_process_document
from .react_agent import get_initial_sections
from .react_agent import format_status_summary
from .react_models import ReActAgentInput
from .react_models import ReActAgentOutput
from .text_processor import remove_references_section
from .factory import create_agent
from .tool_agent import ToolAgent
from .tool_agent import MetadataExtractionResult
from .tool_agent import PreviousMethodsResult
from .tool_agent import ResearchQuestionsResult
from .tool_agent import MethodologyResult
from .tool_agent import ExperimentResult
from .tool_agent import FutureDirectionsResult
from .tool_agent import AnalysisPlan
from .tool_agent import ComprehensiveAnalysisResult

__all__ = [
    "DocumentAgent",
    "DocumentAnalysisResult",
    "ReActAgent",
    "analyze_document_with_react",
    "load_and_process_document",
    "get_initial_sections",
    "format_status_summary",
    "ReActAgentInput",
    "ReActAgentOutput",
    "ToolAgent",
    "MetadataExtractionResult",
    "PreviousMethodsResult",
    "ResearchQuestionsResult",
    "MethodologyResult",
    "ExperimentResult",
    "FutureDirectionsResult",
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",
    "create_agent",
    "remove_references_section",
]

# Version of the agent module
__version__ = "1.0.0"