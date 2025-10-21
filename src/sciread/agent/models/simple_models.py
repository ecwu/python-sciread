"""SimpleAgent result models for structured outputs.

This module contains Pydantic models used by the SimpleAgent for
structured output from document analysis.
"""

from pydantic import BaseModel


class SimpleAnalysisResult(BaseModel):
    """Result of simple document analysis."""

    report: str
