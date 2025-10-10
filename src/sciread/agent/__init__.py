"""Agent module for sciread package.

This module provides document analysis agents using pydantic-ai framework.
The agents can process academic papers and generate comprehensive reports
based on document content and custom prompts.

Main Interface:
    DocumentAgent - Main agent class for document analysis
    create_agent - Factory function to create configured agents

Example Usage:
    from sciread.agent import create_agent
    from sciread.document import Document

    # Create an agent with a specific model
    agent = create_agent("deepseek/deepseek-chat")

    # Process a document
    doc = Document.from_file("paper.pdf")
    doc.load()

    # Generate analysis report
    result = await agent.analyze_document(doc, "Summarize this paper")
"""

from .document_agent import DocumentAgent
from .document_agent import DocumentAnalysisResult
from .text_processor import remove_references_section
from .factory import create_agent

__all__ = [
    "DocumentAgent",
    "DocumentAnalysisResult",
    "create_agent",
    "remove_references_section",
]

# Version of the agent module
__version__ = "1.0.0"