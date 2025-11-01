"""RAG ReActAgent input and output models for structured data.

This module contains Pydantic models used by the RAGReActAgent for
structured input/output during the iterative RAG-based analysis process.
"""

from pydantic import BaseModel
from pydantic import Field


class RAGReActAgentInput(BaseModel):
    """Input model for RAG ReAct agent iterations."""

    task_prompt: str = Field(description="The original analysis task or question about the document")
    status_summary: str = Field(
        description="Summary of current stage, loop count, and remaining loops (e.g., 'Initial analysis (loop 1 of 8)')"
    )
    retrieved_content: str = Field(description="Content retrieved from semantic search for this iteration")
    search_query: str = Field(description="The search query used to retrieve the content")
    search_results_summary: str = Field(description="Summary of what search results were found")
    current_report: str = Field(description="The cumulative report built so far from previous iterations")
    previous_queries: list[str] = Field(description="List of search queries already used in previous iterations")


class RAGReActAgentOutput(BaseModel):
    """Output model for RAG ReAct agent iterations."""

    should_stop: bool = Field(description="Whether to stop the analysis process (True) or continue (False)")
    report_section: str = Field(
        description="New content to add to the report. Should be well-structured with clear section headings (e.g., '## Introduction', '## Methodology'). Can be empty if the retrieved content is not substantial enough to add meaningful analysis."
    )
    next_search_query: str = Field(description="Search query for the next iteration (empty if should_stop is True)")
    reasoning: str = Field(description="Explanation of why the agent made these choices (stop decision, content decision, query selection)")
    search_strategy: str = Field(description="Brief description of what information the next search is targeting")
