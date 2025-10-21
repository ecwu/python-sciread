"""ReActAgent input and output models for structured data.

This module contains Pydantic models used by the ReActAgent for
structured input/output during the iterative analysis process.
"""

from pydantic import BaseModel
from pydantic import Field


class ReActAgentInput(BaseModel):
    """Input model for ReAct agent iterations."""

    task_prompt: str = Field(description="The original analysis task or question about the document")
    available_sections: list[str] = Field(description="List of all available section names in the document")
    status_summary: str = Field(
        description="Summary of current stage, loop count, and remaining loops (e.g., 'Initial analysis (loop 1 of 8)')"
    )
    section_content: str = Field(description="Content of the sections to analyze in this iteration (empty for initial step)")
    current_report: str = Field(description="The cumulative report built so far from previous iterations")
    processed_sections: list[str] = Field(description="List of sections that have already been processed")


class ReActAgentOutput(BaseModel):
    """Output model for ReAct agent iterations."""

    should_stop: bool = Field(description="Whether to stop the analysis process (True) or continue (False)")
    report_section: str = Field(description="New content generated for the current section content")
    next_sections: list[str] = Field(description="List of section names to analyze in the next iteration (empty if should_stop is True)")
    reasoning: str = Field(description="Explanation of why the agent made these choices (stop decision and section selection)")
