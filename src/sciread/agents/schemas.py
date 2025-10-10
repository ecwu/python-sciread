"""Pydantic models for agent outputs and dependencies."""

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class DocumentMetadata(BaseModel):
    """Metadata about the document being analyzed."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    section_types: List[str] = Field(default_factory=list)
    processing_time: float = 0.0


class SectionAnalysis(BaseModel):
    """Analysis result for a specific document section."""
    section_type: str
    content_summary: str
    key_insights: List[str] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0


class ResearchQuestionAnalysis(BaseModel):
    """Analysis of research questions from the paper."""
    primary_question: str
    subsidiary_questions: List[str] = Field(default_factory=list)
    research_gap: str
    problem_context: str
    scope_and_limitations: str


class ContributionAnalysis(BaseModel):
    """Analysis of paper contributions."""
    primary_contribution: str
    technical_contributions: List[str] = Field(default_factory=list)
    theoretical_contributions: List[str] = Field(default_factory=list)
    state_of_the_art_advancement: str
    potential_impact: str


class DocumentAnalysisResult(BaseModel):
    """Comprehensive document analysis result."""
    summary: str
    key_contributions: List[str] = Field(default_factory=list)
    methodology_overview: str
    main_findings: List[str] = Field(default_factory=list)
    implications: str
    future_directions: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    sections_analyzed: List[str] = Field(default_factory=list)
    metadata: DocumentMetadata
    execution_time: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimpleAnalysisResult(BaseModel):
    """Result from simple document analysis."""
    content: str
    question_answered: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    model_used: str
    token_count: int = 0


class AgentError(BaseModel):
    """Error information from agent execution."""
    error_type: str
    error_message: str
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentDeps(BaseModel):
    """Dependency injection for document analysis."""
    document_text: str
    document_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    available_sections: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_identifier: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: Optional[int] = None

    @model_validator(mode='before')
    @classmethod
    def calculate_available_sections(cls, data):
        """Calculate available_sections from document chunks."""
        if isinstance(data, dict):
            document_chunks = data.get('document_chunks', [])
            available_sections = list(set(
                chunk.get('chunk_type', 'unknown')
                for chunk in document_chunks
                if isinstance(chunk, dict) and chunk.get('chunk_type')
            ))
            data['available_sections'] = available_sections
        return data

    def get_section_chunks(self, section_type: str) -> List[Dict[str, Any]]:
        """Get chunks of a specific section type."""
        return [
            chunk for chunk in self.document_chunks
            if chunk.get('chunk_type') == section_type
        ]

    def has_section(self, section_type: str) -> bool:
        """Check if document has a specific section."""
        return section_type in self.available_sections