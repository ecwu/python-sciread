"""Multi-agent document analysis system with controller and expert sub-agents.

This module implements a comprehensive document analysis system using a controller
agent that coordinates multiple expert sub-agents for detailed academic paper analysis.
Each sub-agent specializes in extracting specific types of information from papers.

The system uses programmatic agent hand-off where the controller agent decides
which sub-agents to invoke based on abstract analysis, then the application
code orchestrates the execution and result synthesis.
"""

import asyncio
from typing import Any
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .text_processor import clean_academic_text
from .text_processor import extract_document_metadata
from .text_processor import remove_references_section

# Pydantic models for structured results


class MetadataExtractionResult(BaseModel):
    """Result of metadata extraction from academic paper."""

    title: Optional[str] = Field(None, description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of authors")
    affiliations: list[str] = Field(default_factory=list, description="Author affiliations (company, university, or lab)")
    venue: Optional[str] = Field(None, description="Publication venue (journal, conference, or arxiv)")
    year: Optional[int] = Field(None, description="Publication year")
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

    analyze_metadata: bool = Field(True, description="Whether to analyze metadata")
    analyze_previous_methods: bool = Field(True, description="Whether to analyze previous methods")
    analyze_research_questions: bool = Field(True, description="Whether to analyze research questions")
    analyze_methodology: bool = Field(True, description="Whether to analyze methodology")
    analyze_experiments: bool = Field(True, description="Whether to analyze experiments")
    analyze_future_directions: bool = Field(True, description="Whether to analyze future directions")
    reasoning: str = Field("", description="Reasoning behind the analysis plan")
    estimated_relevance_scores: dict[str, float] = Field(default_factory=dict, description="Relevance scores for each analysis type")


class ComprehensiveAnalysisResult(BaseModel):
    """Comprehensive result containing all sub-agent analyses."""

    analysis_plan: AnalysisPlan
    metadata_result: Optional[MetadataExtractionResult] = None
    previous_methods_result: Optional[PreviousMethodsResult] = None
    research_questions_result: Optional[ResearchQuestionsResult] = None
    methodology_result: Optional[MethodologyResult] = None
    experiment_result: Optional[ExperimentResult] = None
    future_directions_result: Optional[FutureDirectionsResult] = None

    execution_summary: dict[str, Any] = Field(default_factory=dict, description="Summary of execution details")
    final_report: str = Field("", description="Synthesized final report")
    total_execution_time: float = Field(0.0, description="Total execution time in seconds")
    interaction_log: list[dict[str, Any]] = Field(default_factory=list, description="Interaction log with prompts and outputs")


# Expert sub-agent classes


class MetadataExtractorAgent:
    """Expert agent for extracting bibliographic metadata from academic papers."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 60.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the metadata extractor agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        # System prompt for metadata extraction
        system_prompt = """You are an expert bibliographic analyst specializing in extracting structured metadata from academic papers. Your task is to carefully read academic documents and extract precise bibliographic information.

Key responsibilities:
1. Extract the exact title of the paper
2. Identify all authors and their affiliations (company, university, or lab)
3. Determine the publication venue (journal name, conference name, or arxiv)
4. Extract publication year (if available)

Guidelines:
- Be precise and accurate in information extraction
- If information is not clearly present, mark it as None rather than guessing
- Author names should be extracted exactly as they appear
- Extract affiliations as complete institutional names (e.g., "OpenAI", "Stanford University", "Google Research")
- For venue: extract journal name, conference name, or identify as "arXiv" if it's an arXiv preprint
- Only extract venue information if it can be clearly obtained from the text
- Publication year should only be extracted if explicitly mentioned
- Confidence should reflect how certain you are about the extracted information
- Pay attention to formatting variations (e.g., different citation styles)

Always provide structured, accurate metadata that could be used for academic citation purposes."""

        # Create the pydantic-ai agent
        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=MetadataExtractionResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "metadata_extractor",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"MetadataExtractor interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> MetadataExtractionResult:
        """Extract metadata from the document."""
        self.logger.info("Starting metadata extraction")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        if not text or not text.strip():
            raise ValueError("Document has no text content to analyze")

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Extract the following key bibliographic metadata from this academic paper:

1. Title: The exact paper title
2. Authors: Complete list of authors as they appear
3. Affiliations: Author affiliations (company, university, or lab)
4. Venue: Publication venue (journal name, conference name, or arxiv)
5. Year: Publication year (only if explicitly mentioned)

Document text:
{text[:10000]}  # Limit to first 10k chars for metadata extraction

Focus on extracting accurate information for these five fields. Only include venue and year if they can be clearly identified from the text. For affiliations, extract the complete institutional names. For venue, be specific about journal name, conference name, or identify as arXiv preprint if applicable."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Metadata extraction completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Metadata extraction timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Metadata extraction timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class PreviousMethodsAgent:
    """Expert agent for analyzing previous work and methodologies."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 120.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the previous methods agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        system_prompt = """You are an expert research analyst specializing in understanding the context of academic papers within their research field. Your task is to analyze how a paper relates to existing work and identify its unique contributions.

Key responsibilities:
1. Identify key related work and prior approaches mentioned
2. Extract important methodologies from previous research
3. Analyze limitations of existing approaches
4. Identify research gaps the current work addresses
5. Highlight novel aspects compared to prior work

Guidelines:
- Focus on the context and background sections
- Look for explicit mentions of related work
- Identify limitations mentioned by the authors
- Pay attention to claims about novelty
- Consider both methodological and theoretical contributions
- Be thorough in identifying the research landscape

Provide a comprehensive analysis of how this work fits into the broader research context."""

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=PreviousMethodsResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "previous_methods",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"PreviousMethods interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> PreviousMethodsResult:
        """Analyze previous work and methods."""
        self.logger.info("Starting previous methods analysis")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Analyze the research context and previous work related to this academic paper. Focus on:

1. Related work: Identify key papers and approaches mentioned
2. Key methods: Extract important methodologies from prior research
3. Limitations: Analyze limitations of existing approaches identified by authors
4. Research gaps: Identify specific gaps this work addresses
5. Novelty: Highlight what makes this work novel compared to prior approaches

Document text:
{text}

Provide a comprehensive analysis of how this work relates to and builds upon previous research."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Previous methods analysis completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Previous methods analysis timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Previous methods analysis timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Previous methods analysis failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class ResearchQuestionsAgent:
    """Expert agent for analyzing research questions and contributions."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 120.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the research questions agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        system_prompt = """You are an expert research analyst specializing in identifying the core research questions and contributions of academic papers. Your task is to analyze what questions the paper addresses and what it contributes to the field.

Key responsibilities:
1. Identify primary research questions
2. Extract research hypotheses when present
3. Analyze main contributions of the work
4. Assess research significance
5. Identify target audience for the work

Guidelines:
- Look for explicit research questions in introduction
- Identify hypotheses in theoretical work
- Analyze contributions mentioned in abstract and conclusion
- Consider both theoretical and practical contributions
- Assess significance based on claimed impact
- Identify who would benefit from this research

Provide a comprehensive analysis of the research questions and contributions."""

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=ResearchQuestionsResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "research_questions",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"ResearchQuestions interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> ResearchQuestionsResult:
        """Analyze research questions and contributions."""
        self.logger.info("Starting research questions analysis")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Analyze the research questions and contributions of this academic paper. Focus on:

1. Main questions: Identify the primary research questions addressed
2. Hypotheses: Extract research hypotheses if present
3. Contributions: Analyze the main contributions of the work
4. Significance: Assess the significance and impact of the research
5. Target audience: Identify who would benefit from this research

Document text:
{text}

Provide a comprehensive analysis of what research questions this work addresses and its contributions to the field."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Research questions analysis completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Research questions analysis timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Research questions analysis timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Research questions analysis failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class MethodologyAgent:
    """Expert agent for analyzing methodology and technical approach."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 120.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the methodology agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        system_prompt = """You are an expert technical analyst specializing in understanding research methodologies in academic papers. Your task is to analyze the technical approach, methods, and experimental design of research work.

Key responsibilities:
1. Identify the overall methodological approach
2. Extract specific techniques and algorithms used
3. Analyze key assumptions made in the methodology
4. Identify data sources or datasets
5. Extract evaluation metrics and validation methods
6. Identify methodological limitations
7. Assess reproducibility of the approach

Guidelines:
- Focus on methodology, methods, and experimental setup sections
- Identify both theoretical foundations and practical implementations
- Pay attention to assumptions and constraints
- Consider data collection and processing methods
- Analyze how results are evaluated
- Assess limitations and potential issues with the approach

Provide a comprehensive analysis of the technical methodology and approach."""

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=MethodologyResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "methodology",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"Methodology interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> MethodologyResult:
        """Analyze methodology and technical approach."""
        self.logger.info("Starting methodology analysis")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Analyze the methodology and technical approach of this academic paper. Focus on:

1. Overall approach: Describe the main methodological framework
2. Techniques: Identify specific techniques, algorithms, or methods used
3. Assumptions: Extract key assumptions made in the methodology
4. Data sources: Identify datasets or data sources used
5. Evaluation metrics: Extract metrics used for evaluation
6. Limitations: Identify methodological limitations
7. Reproducibility: Assess reproducibility of the approach

Document text:
{text}

Provide a comprehensive analysis of the technical methodology and experimental approach."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Methodology analysis completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Methodology analysis timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Methodology analysis timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Methodology analysis failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class ExperimentsAgent:
    """Expert agent for analyzing experiments and results."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 120.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the experiments agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        system_prompt = """You are an expert experimental analyst specializing in understanding experimental design and results in academic papers. Your task is to analyze how experiments are conducted and what results are obtained.

Key responsibilities:
1. Analyze experimental setup and design
2. Identify datasets and baselines used
3. Extract key experimental results
4. Analyze quantitative metrics and performance
5. Identify qualitative findings and insights
6. Analyze statistical significance
7. Examine error analysis and failure cases

Guidelines:
- Focus on experiments, results, and evaluation sections
- Identify both quantitative and qualitative results
- Pay attention to experimental design choices
- Consider statistical analysis and significance testing
- Analyze comparison with baseline methods
- Look for error analysis and ablation studies
- Extract specific numbers and metrics when available

Provide a comprehensive analysis of the experimental setup, results, and findings."""

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=ExperimentResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "experiments",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"Experiments interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> ExperimentResult:
        """Analyze experiments and results."""
        self.logger.info("Starting experiments analysis")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Analyze the experiments and results of this academic paper. Focus on:

1. Experimental setup: Describe how experiments are designed and conducted
2. Datasets: Identify datasets used in the experiments
3. Baselines: Extract baseline methods compared against
4. Results: Analyze key experimental results and findings
5. Quantitative results: Extract specific numerical results and metrics
6. Qualitative findings: Identify qualitative insights and observations
7. Statistical significance: Analyze statistical significance of results
8. Error analysis: Examine error analysis and failure cases

Document text:
{text}

Provide a comprehensive analysis of the experimental setup, results, and findings."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Experiments analysis completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Experiments analysis timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Experiments analysis timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Experiments analysis failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class FutureDirectionsAgent:
    """Expert agent for analyzing future work and implications."""

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 120.0,
        interaction_log: Optional[list] = None,
    ):
        """Initialize the future directions agent."""
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        # Interaction log storage (use provided log or create local one)
        self.interaction_log = interaction_log if interaction_log is not None else []

        system_prompt = """You are an expert research analyst specializing in understanding the broader impact and future directions of academic research. Your task is to analyze the implications, limitations, and future work suggested by academic papers.

Key responsibilities:
1. Identify suggested future research directions
2. Analyze current limitations of the work
3. Extract practical applications and implications
4. Identify theoretical contributions and impact
5. Extract open questions raised by the work
6. Analyze societal impact considerations

Guidelines:
- Focus on conclusion, discussion, and future work sections
- Look for explicit mentions of limitations
- Identify suggested improvements or extensions
- Consider both theoretical and practical implications
- Analyze impact on the research field and society
- Look for open questions and challenges

Provide a comprehensive analysis of the implications, limitations, and future directions."""

        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=FutureDirectionsResult,
            retries=self.max_retries,
        )

    def _log_interaction(self, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction for this agent."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": "future_directions",
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"FutureDirections interaction: Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    async def analyze(
        self,
        document: Document,
        remove_references: bool = True,
        clean_text: bool = True,
    ) -> FutureDirectionsResult:
        """Analyze future directions and implications."""
        self.logger.info("Starting future directions analysis")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Process text
        if remove_references:
            text = remove_references_section(text)

        if clean_text:
            text = clean_academic_text(text)

        # Build prompt
        prompt = f"""Analyze the future directions and implications of this academic paper. Focus on:

1. Future work: Identify suggested future research directions
2. Limitations: Analyze current limitations of the work
3. Practical implications: Extract practical applications and implications
4. Theoretical implications: Identify theoretical contributions and impact
5. Open questions: Extract open questions raised by the work
6. Societal impact: Analyze societal impact considerations

Document text:
{text}

Provide a comprehensive analysis of the implications, limitations, and future directions of this research."""

        try:
            result = await asyncio.wait_for(
                self.agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Future directions analysis completed successfully")
            self._log_interaction(prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Future directions analysis timed out after {self.timeout} seconds")
            self._log_interaction(prompt, "", f"TimeoutError: {self.timeout} seconds")
            raise TimeoutError(f"Future directions analysis timed out after {self.timeout} seconds") from None

        except Exception as e:
            self.logger.error(f"Future directions analysis failed: {e}")
            self._log_interaction(prompt, "", str(e))
            raise


class ToolAgent:
    """Controller agent for coordinating expert sub-agents in academic paper analysis.

    This controller agent uses programmatic agent hand-off to coordinate multiple
    expert sub-agents for comprehensive academic paper analysis. It analyzes
    the abstract to determine which sub-agents to invoke, orchestrates their
    execution, and synthesizes the results into a comprehensive report.
    """

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        """Initialize the ToolAgent controller.

        Args:
            model: Model identifier for the LLM provider
            max_retries: Maximum number of retries for failed requests
            timeout: Default timeout in seconds for controller operations
        """
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Interaction logging storage (always available)
        self.interaction_log = []

        # Initialize model
        if isinstance(model, str):
            self.model = get_model(model)
            self.model_identifier = model
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")

        self.logger.info(f"Initialized ToolAgent controller with model: {self.model_identifier}")

        # Initialize expert sub-agents
        self.metadata_agent = MetadataExtractorAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=60.0,
            interaction_log=self.interaction_log,
        )
        self.previous_methods_agent = PreviousMethodsAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=120.0,
            interaction_log=self.interaction_log,
        )
        self.research_questions_agent = ResearchQuestionsAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=120.0,
            interaction_log=self.interaction_log,
        )
        self.methodology_agent = MethodologyAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=120.0,
            interaction_log=self.interaction_log,
        )
        self.experiments_agent = ExperimentsAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=120.0,
            interaction_log=self.interaction_log,
        )
        self.future_directions_agent = FutureDirectionsAgent(
            model=model,
            max_retries=self.max_retries,
            timeout=120.0,
            interaction_log=self.interaction_log,
        )

        # Controller agent for planning and synthesis
        system_prompt = """You are an expert academic research coordinator specializing in analyzing academic papers and determining the most effective analysis strategy. Your role is to understand the paper's content and domain to create an optimal analysis plan.

Key responsibilities:
1. Analyze the abstract to understand the paper's domain and type
2. Determine which expert analyses would be most valuable
3. Plan the sequence and priority of different analyses
4. Synthesize results from multiple expert analyses into a coherent report

Analysis types available:
- Metadata extraction: Bibliographic information and paper identification
- Previous methods analysis: Context, related work, and novelty assessment
- Research questions analysis: Core questions, contributions, and significance
- Methodology analysis: Technical approach, methods, and design choices
- Experiments analysis: Experimental setup, results, and validation
- Future directions analysis: Limitations, implications, and future work

Guidelines for planning:
- Consider the paper's domain (e.g., theoretical CS, empirical study, survey)
- Assess what information is likely to be present based on abstract content
- Prioritize analyses that will provide the most valuable insights
- Consider which analyses are most relevant for the paper type
- Plan for comprehensive but focused analysis

Provide clear reasoning for your analysis plan and relevance assessments."""

        self.controller_agent = Agent(
            model=self.model,
            system_prompt=system_prompt,
            output_type=AnalysisPlan,
            retries=self.max_retries,
        )

        self.logger.info("ToolAgent controller initialized successfully")

    def _log_interaction(self, agent_name: str, prompt: str, output: str, error: Optional[str] = None):
        """Log interaction information for agent prompts and outputs."""
        import datetime

        interaction_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent": agent_name,
            "prompt": prompt,
            "output": output,
            "error": error,
            "prompt_length": len(prompt),
            "output_length": len(output) if output else 0,
        }

        self.interaction_log.append(interaction_entry)
        self.logger.debug(
            f"Interaction log: {agent_name} - Prompt: {interaction_entry['prompt_length']} chars, Output: {interaction_entry['output_length']} chars"
        )

    def save_interaction_log(self, file_path: str):
        """Save interaction log to a JSON file."""
        import json
        from pathlib import Path

        try:
            Path(file_path).write_text(json.dumps(self.interaction_log, indent=2), encoding="utf-8")
            self.logger.info(f"Interaction log saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save interaction log: {e}")

    def get_interaction_log(self) -> list[dict[str, Any]]:
        """Get the complete interaction log."""
        return self.interaction_log.copy()

    def get_interaction_summary(self) -> dict[str, Any]:
        """Get a summary of interaction information."""
        total_interactions = len(self.interaction_log)
        total_prompt_chars = sum(entry["prompt_length"] for entry in self.interaction_log)
        total_output_chars = sum(entry["output_length"] for entry in self.interaction_log)
        agents_used = list(set(entry["agent"] for entry in self.interaction_log))
        errors = sum(1 for entry in self.interaction_log if entry["error"])

        return {
            "total_interactions": total_interactions,
            "total_prompt_characters": total_prompt_chars,
            "total_output_characters": total_output_chars,
            "agents_used": agents_used,
            "errors_count": errors,
            "entries": self.interaction_log,
        }

    def get_agent_interactions(self, agent_name: str) -> list[dict[str, Any]]:
        """Get interactions for a specific agent."""
        return [entry for entry in self.interaction_log if entry["agent"] == agent_name]

    def clear_interaction_log(self):
        """Clear the interaction log."""
        self.interaction_log.clear()
        self.logger.info("Interaction log cleared")

    def extract_abstract(self, document: Document) -> str:
        """Extract abstract from document.

        Args:
            document: Document object to extract abstract from

        Returns:
            Abstract text as string
        """
        self.logger.info("Extracting abstract from document")

        # Try to get abstract from document metadata first
        if document.metadata and document.metadata.title:
            # Use text_processor to extract abstract
            if document.chunks:
                text = document.get_full_text()
            else:
                text = document.text

            metadata = extract_document_metadata(text)
            if metadata.get("abstract"):
                self.logger.info("Abstract extracted from document metadata")
                return metadata["abstract"]

        # If no abstract in metadata, try to find it in the text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        # Look for abstract section
        lines = text.split("\n")
        abstract_lines = []
        abstract_started = False

        for line in lines:
            line_stripped = line.strip().lower()

            if line_stripped.startswith("abstract"):
                abstract_started = True
                continue

            if abstract_started:
                if line_stripped and not line_stripped.startswith(("introduction", "keywords", "1.", "i.", "©")):
                    abstract_lines.append(line.strip())
                elif line_stripped and len(abstract_lines) > 0:
                    # End of abstract
                    break

        if abstract_lines:
            abstract = " ".join(abstract_lines)
            self.logger.info(f"Abstract extracted from text: {len(abstract)} characters")
            return abstract

        self.logger.warning("No abstract found in document")
        return ""

    async def plan_analysis(self, abstract: str) -> AnalysisPlan:
        """Plan which sub-agents to use based on abstract analysis.

        Args:
            abstract: Abstract text to analyze

        Returns:
            AnalysisPlan with selected sub-agents and reasoning
        """
        self.logger.info("Planning analysis based on abstract")

        if not abstract or not abstract.strip():
            self.logger.warning("No abstract provided, using default analysis plan")
            return AnalysisPlan(reasoning="No abstract available, using comprehensive analysis plan")

        prompt = f"""Based on the following abstract, create an optimal analysis plan for this academic paper. Determine which expert analyses would be most valuable and provide relevance scores (0.0-1.0) for each analysis type.

Abstract:
{abstract}

Consider:
1. What type of paper this appears to be (theoretical, empirical, survey, etc.)
2. What domain/field the paper is in
3. What information is likely to be present based on the abstract
4. Which analyses would provide the most valuable insights

Provide a comprehensive analysis plan with clear reasoning."""

        try:
            result = await asyncio.wait_for(
                self.controller_agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Analysis plan created successfully")
            self._log_interaction("controller_agent", prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Analysis planning timed out after {self.timeout} seconds")
            self._log_interaction("controller_agent", prompt, "", f"TimeoutError: {self.timeout} seconds")
            # Return default plan
            return AnalysisPlan(reasoning="Planning timed out, using default comprehensive analysis")

        except Exception as e:
            self.logger.error(f"Analysis planning failed: {e}")
            self._log_interaction("controller_agent", prompt, "", str(e))
            # Return default plan
            return AnalysisPlan(reasoning=f"Planning failed ({e}), using default comprehensive analysis")

    async def execute_sub_agents(
        self,
        document: Document,
        analysis_plan: AnalysisPlan,
    ) -> dict[str, Any]:
        """Execute selected sub-agents based on analysis plan.

        Args:
            document: Document object to analyze
            analysis_plan: Plan determining which agents to execute

        Returns:
            Dictionary containing results from executed sub-agents
        """
        self.logger.info("Executing sub-agents based on analysis plan")
        start_time = asyncio.get_event_loop().time()

        results = {}
        tasks = []

        # Create tasks for selected agents
        if analysis_plan.analyze_metadata:
            task = asyncio.create_task(self._safe_execute_agent(self.metadata_agent.analyze(document), "metadata_extraction"))
            tasks.append(("metadata", task))

        if analysis_plan.analyze_previous_methods:
            task = asyncio.create_task(self._safe_execute_agent(self.previous_methods_agent.analyze(document), "previous_methods"))
            tasks.append(("previous_methods", task))

        if analysis_plan.analyze_research_questions:
            task = asyncio.create_task(self._safe_execute_agent(self.research_questions_agent.analyze(document), "research_questions"))
            tasks.append(("research_questions", task))

        if analysis_plan.analyze_methodology:
            task = asyncio.create_task(self._safe_execute_agent(self.methodology_agent.analyze(document), "methodology"))
            tasks.append(("methodology", task))

        if analysis_plan.analyze_experiments:
            task = asyncio.create_task(self._safe_execute_agent(self.experiments_agent.analyze(document), "experiments"))
            tasks.append(("experiments", task))

        if analysis_plan.analyze_future_directions:
            task = asyncio.create_task(self._safe_execute_agent(self.future_directions_agent.analyze(document), "future_directions"))
            tasks.append(("future_directions", task))

        if not tasks:
            self.logger.warning("No agents selected for execution")
            return results

        # Execute tasks in parallel
        self.logger.info(f"Executing {len(tasks)} sub-agents in parallel")
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        for (agent_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_name} failed: {result}")
                results[agent_name] = {"error": str(result), "success": False}
            else:
                self.logger.info(f"Agent {agent_name} completed successfully")
                results[agent_name] = {"result": result, "success": True}

        execution_time = asyncio.get_event_loop().time() - start_time
        self.logger.info(f"Sub-agent execution completed in {execution_time:.2f} seconds")

        return results

    async def _safe_execute_agent(self, coro, agent_name: str):
        """Safely execute an agent with error handling."""
        try:
            return await coro
        except Exception as e:
            self.logger.error(f"Agent {agent_name} execution failed: {e}")
            raise

    async def synthesize_report(
        self,
        analysis_plan: AnalysisPlan,
        sub_agent_results: dict[str, Any],
        document: Document,
    ) -> str:
        """Synthesize final comprehensive report from sub-agent results.

        Args:
            analysis_plan: The analysis plan that was executed
            sub_agent_results: Results from executed sub-agents
            document: Original document for context

        Returns:
            Comprehensive synthesized report
        """
        self.logger.info("Synthesizing final report from sub-agent results")

        # Extract paper title for structured reporting
        paper_title = "Unknown Paper"
        if document.metadata and document.metadata.title:
            paper_title = document.metadata.title
        else:
            # Try to extract from metadata results
            if sub_agent_results.get("metadata", {}).get("success"):
                metadata_result = sub_agent_results["metadata"]["result"]
                if hasattr(metadata_result, 'title') and metadata_result.title:
                    paper_title = metadata_result.title

        # Build synthesis prompt with structured format requirements
        prompt_parts = [
            "You are an expert academic research analyst tasked with creating a comprehensive, coherent report from multiple expert analyses of an academic paper.",
            "",
            "REPORT STRUCTURE REQUIREMENTS:",
            f"1. Title: Always start with the exact paper title followed by ' - Comprehensive Report'",
            "2. Metadata Section: Include a clear metadata section with article information in list or table format",
            "3. Content Sections: Create well-organized content sections based on the available analyses",
            "4. Focus: Write directly about the paper content without mentioning agents, tools, or analysis processes",
            "",
            "PAPER INFORMATION:",
            f"Paper Title: {paper_title}",
            f"Source: {document.source_path or 'text document'}",
        ]

        abstract = self.extract_abstract(document)
        if abstract:
            prompt_parts.append(f"Abstract: {abstract[:500]}...")

        prompt_parts.extend([
            "",
            "AVAILABLE ANALYSES:",
        ])

        # Add results from successful agents
        for agent_name, result_data in sub_agent_results.items():
            if result_data.get("success", False):
                result = result_data["result"]
                prompt_parts.extend([f"{agent_name.upper()} ANALYSIS:", str(result), ""])
            else:
                prompt_parts.extend([f"{agent_name.upper()} ANALYSIS:", f"Analysis failed: {result_data.get('error', 'Unknown error')}", ""])

        prompt_parts.extend([
            "",
            "SYNTHESIS INSTRUCTIONS:",
            "Create a comprehensive academic paper analysis report with the following structure:",
            "",
            "REPORT FORMAT:",
            f"[Exact Paper Title] - Comprehensive Report",
            "",
            "## Paper Information",
            "- **Title:** [Paper title]",
            "- **Authors:** [List of authors]",
            "- **Affiliations:** [Author affiliations]",
            "- **Venue:** [Journal/Conference/ArXiv]",
            "- **Year:** [Publication year]",
            "",
            "## Main Content Sections",
            "After the metadata section, start with appropriate, natural section titles such as:",
            "## Introduction and Research Context",
            "## Research Questions and Contributions",
            "## Methodology",
            "## Experiments and Results",
            "## Discussion and Analysis",
            "## Limitations and Future Work",
            "## Conclusions and Implications",
            "",
            "Choose and order sections based on what's most relevant to the paper. Use only the sections that have meaningful content.",
            "",
            "WRITING GUIDELINES:",
            "1. Use the exact paper title followed by ' - Comprehensive Report' as the main title",
            "2. Include a comprehensive metadata section with all available bibliographic information",
            "3. Focus exclusively on the paper's content, findings, and contributions",
            "4. DO NOT mention agents, tools, sub-agents, analysis processes, or methodologies used to create the report",
            "5. Write in a professional academic tone suitable for researchers",
            "6. Integrate insights from all available analyses into coherent sections",
            "7. Be comprehensive but maintain readability and logical flow",
            "8. If certain analyses failed, focus on the available information without noting the gaps",
            "9. Use natural, descriptive section titles that readers would expect in an academic paper",
            "",
            "Please provide a thorough, well-structured academic analysis report based on all available information.",
        ])

        prompt = "\n".join(prompt_parts)

        # Create synthesis agent
        synthesis_agent = Agent(
            model=self.model,
            system_prompt="You are an expert academic research analyst specializing in creating comprehensive, well-structured reports from multiple expert analyses.",
            retries=self.max_retries,
        )

        try:
            result = await asyncio.wait_for(
                synthesis_agent.run(prompt),
                timeout=self.timeout,
            )
            self.logger.info("Report synthesis completed successfully")
            self._log_interaction("synthesis_agent", prompt, str(result.output))
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Report synthesis timed out after {self.timeout} seconds")
            self._log_interaction("synthesis_agent", prompt, "", f"TimeoutError: {self.timeout} seconds")
            return f"Report synthesis timed out after {self.timeout} seconds. Please try again or use a shorter document."

        except Exception as e:
            self.logger.error(f"Report synthesis failed: {e}")
            self._log_interaction("synthesis_agent", prompt, "", str(e))
            return f"Report synthesis failed: {e}. Please try again."

    async def analyze_document(
        self,
        document: Document,
        custom_plan: Optional[AnalysisPlan] = None,
        **kwargs: Any,
    ) -> ComprehensiveAnalysisResult:
        """Perform comprehensive document analysis using expert sub-agents.

        Args:
            document: Document object to analyze
            custom_plan: Optional custom analysis plan (if None, will auto-generate)
            **kwargs: Additional arguments (currently unused but kept for compatibility)

        Returns:
            ComprehensiveAnalysisResult with all sub-analyses and final report
        """
        self.logger.info(f"Starting comprehensive document analysis: {document.source_path or 'text document'}")
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Extract abstract
            abstract = self.extract_abstract(document)

            # Step 2: Plan analysis (use custom plan if provided)
            if custom_plan:
                analysis_plan = custom_plan
                self.logger.info("Using custom analysis plan")
            else:
                analysis_plan = await self.plan_analysis(abstract)
                self.logger.info("Generated automatic analysis plan")

            # Step 3: Execute sub-agents
            sub_agent_results = await self.execute_sub_agents(document, analysis_plan)

            # Step 4: Synthesize final report
            final_report = await self.synthesize_report(analysis_plan, sub_agent_results, document)

            # Step 5: Build comprehensive result
            total_execution_time = asyncio.get_event_loop().time() - start_time

            comprehensive_result = ComprehensiveAnalysisResult(
                analysis_plan=analysis_plan,
                metadata_result=sub_agent_results.get("metadata", {}).get("result")
                if sub_agent_results.get("metadata", {}).get("success")
                else None,
                previous_methods_result=sub_agent_results.get("previous_methods", {}).get("result")
                if sub_agent_results.get("previous_methods", {}).get("success")
                else None,
                research_questions_result=sub_agent_results.get("research_questions", {}).get("result")
                if sub_agent_results.get("research_questions", {}).get("success")
                else None,
                methodology_result=sub_agent_results.get("methodology", {}).get("result")
                if sub_agent_results.get("methodology", {}).get("success")
                else None,
                experiment_result=sub_agent_results.get("experiments", {}).get("result")
                if sub_agent_results.get("experiments", {}).get("success")
                else None,
                future_directions_result=sub_agent_results.get("future_directions", {}).get("result")
                if sub_agent_results.get("future_directions", {}).get("success")
                else None,
                execution_summary={
                    "total_agents_executed": len(sub_agent_results),
                    "successful_agents": len([r for r in sub_agent_results.values() if r.get("success", False)]),
                    "failed_agents": len([r for r in sub_agent_results.values() if not r.get("success", False)]),
                    "agent_results": {name: {"success": data.get("success", False)} for name, data in sub_agent_results.items()},
                },
                final_report=final_report,
                total_execution_time=total_execution_time,
                interaction_log=self.get_interaction_log(),
            )

            self.logger.info(f"Comprehensive document analysis completed in {total_execution_time:.2f} seconds")
            return comprehensive_result

        except Exception as e:
            self.logger.error(f"Comprehensive document analysis failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the ToolAgent."""
        return f"ToolAgent(model={self.model_identifier})"
