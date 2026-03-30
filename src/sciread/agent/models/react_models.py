"""ReActAgent result models for structured outputs."""

from pydantic import BaseModel
from pydantic import Field


class AnalysisReport(BaseModel):
    """Structured final output for ReAct-based document analysis."""

    summary: str = Field(description="High-level summary of the paper and its main purpose")
    research_questions: list[str] = Field(default_factory=list, description="Primary research questions or objectives")
    methodology: str = Field(description="Summary of the methodology and technical approach")
    key_findings: list[str] = Field(default_factory=list, description="Main findings and empirical or theoretical results")
    contributions: list[str] = Field(default_factory=list, description="Main contributions and significance of the work")
    limitations: str | None = Field(default=None, description="Key limitations, caveats, or unresolved issues")
    sections_covered: list[str] = Field(default_factory=list, description="Document sections incorporated into the analysis")
    final_report: str = Field(description="Complete human-readable report synthesizing all known information")
