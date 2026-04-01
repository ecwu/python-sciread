"""CoordinateAgent result models for structured outputs.

This module contains Pydantic models used by CoordinateAgent system
for structured output from various expert sub-agents.
"""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class MetadataExtractionResult(BaseModel):
    """Result of metadata extraction from academic paper."""

    title: str | None = Field(None, description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of authors")
    affiliations: list[str] = Field(
        default_factory=list,
        description="Author affiliations (company, university, or lab)",
    )
    venue: str | None = Field(None, description="Publication venue (journal, conference, or arxiv)")
    year: int | None = Field(None, description="Publication year")
    confidence: float = Field(1.0, description="Confidence in extracted metadata")


class PreviousMethodsResult(BaseModel):
    """Result of previous work and methods analysis."""

    related_work: list[str] = Field(default_factory=list, description="Key related work papers and approaches")
    key_methods: list[str] = Field(default_factory=list, description="Important methodologies from prior work")
    limitations: list[str] = Field(default_factory=list, description="Limitations of existing approaches")
    research_gaps: list[str] = Field(default_factory=list, description="Identified research gaps")
    novelty_aspects: list[str] = Field(default_factory=list, description="Novel aspects compared to prior work")
    confidence: float = Field(1.0, description="Confidence in analysis")


class ResearchQuestionsResult(BaseModel):
    """Result of research questions and contributions analysis."""

    main_questions: list[str] = Field(default_factory=list, description="Primary research questions")
    hypotheses: list[str] = Field(default_factory=list, description="Research hypotheses")
    contributions: list[str] = Field(default_factory=list, description="Main contributions")
    research_significance: str = Field("", description="Significance of the research")
    target_audience: list[str] = Field(default_factory=list, description="Target audience for this work")
    confidence: float = Field(1.0, description="Confidence in analysis")


class MethodologyResult(BaseModel):
    """Result of methodology and technical approach analysis."""

    approach: str = Field("", description="Overall methodological approach")
    techniques: list[str] = Field(default_factory=list, description="Specific techniques used")
    assumptions: list[str] = Field(default_factory=list, description="Key assumptions made")
    data_sources: list[str] = Field(default_factory=list, description="Data sources or datasets used")
    evaluation_metrics: list[str] = Field(default_factory=list, description="Metrics used for evaluation")
    limitations: list[str] = Field(default_factory=list, description="Methodological limitations")
    reproducibility_notes: str = Field("", description="Notes on reproducibility")
    confidence: float = Field(1.0, description="Confidence in analysis")


class ExperimentResult(BaseModel):
    """Result of experiments and results analysis."""

    setup: str = Field("", description="Experimental setup description")
    datasets: list[str] = Field(default_factory=list, description="Datasets used in experiments")
    baselines: list[str] = Field(default_factory=list, description="Baseline methods compared against")
    results: list[str] = Field(default_factory=list, description="Key experimental results")
    quantitative_results: dict[str, float] = Field(default_factory=dict, description="Quantitative metrics")
    qualitative_findings: list[str] = Field(default_factory=list, description="Qualitative findings")
    statistical_significance: list[str] = Field(default_factory=list, description="Statistical significance observations")
    error_analysis: list[str] = Field(default_factory=list, description="Error analysis and failure cases")
    confidence: float = Field(1.0, description="Confidence in analysis")


class FutureDirectionsResult(BaseModel):
    """Result of future work and implications analysis."""

    future_work: list[str] = Field(default_factory=list, description="Suggested future research directions")
    limitations: list[str] = Field(default_factory=list, description="Current limitations of the work")
    practical_implications: list[str] = Field(default_factory=list, description="Practical applications and implications")
    theoretical_implications: list[str] = Field(default_factory=list, description="Theoretical contributions")
    open_questions: list[str] = Field(default_factory=list, description="Open questions raised by the work")
    societal_impact: list[str] = Field(default_factory=list, description="Societal impact considerations")
    confidence: float = Field(1.0, description="Confidence in analysis")


class AnalysisPlan(BaseModel):
    """Plan for which sub-agents to use for analysis."""

    analyze_metadata: bool = Field(description="Whether to analyze metadata")
    analyze_previous_methods: bool = Field(description="Whether to analyze previous methods")
    analyze_research_questions: bool = Field(description="Whether to analyze research questions")
    analyze_methodology: bool = Field(description="Whether to analyze methodology")
    analyze_experiments: bool = Field(description="Whether to analyze experiments")
    analyze_future_directions: bool = Field(description="Whether to analyze future directions")

    # Section selection fields for all agents EXCEPT metadata
    # Metadata uses hardcoded first 3 chunks approach
    previous_methods_sections: list[str] = Field(
        description="Sections for previous work analysis. MUST be a list of specific section names from the available sections, NOT 'All sections'."
    )
    research_questions_sections: list[str] = Field(
        description="Sections for research questions. MUST be a list of specific section names from the available sections, NOT 'All sections'."
    )
    methodology_sections: list[str] = Field(
        description="Sections for methodology analysis. MUST be a list of specific section names from the available sections, NOT 'All sections'."
    )
    experiments_sections: list[str] = Field(
        description="Sections for experiments. MUST be a list of specific section names from the available sections, NOT 'All sections'."
    )
    future_directions_sections: list[str] = Field(
        description="Sections for future directions. MUST be a list of specific section names from the available sections, NOT 'All sections'."
    )

    reasoning: str = Field(
        description="Detailed reasoning behind the analysis plan, including WHY specific sections were selected for each analysis type."
    )
    estimated_relevance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Relevance scores (0.0 to 1.0) for each analysis type indicating how valuable that analysis would be for this paper.",
    )


class ComprehensiveAnalysisResult(BaseModel):
    """Comprehensive result containing all sub-agent analyses."""

    analysis_plan: AnalysisPlan
    metadata_result: MetadataExtractionResult | None = None
    previous_methods_result: PreviousMethodsResult | None = None
    research_questions_result: ResearchQuestionsResult | None = None
    methodology_result: MethodologyResult | None = None
    experiment_result: ExperimentResult | None = None
    future_directions_result: FutureDirectionsResult | None = None

    execution_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of execution details")
    final_report: str = Field("", description="Synthesized final report")
    total_execution_time: float = Field(0.0, description="Total execution time in seconds")
    sections_analyzed: dict[str, list[str]] = Field(default_factory=dict, description="Which sections were analyzed by each agent")
