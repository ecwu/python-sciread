"""Multi-agent document analysis system with controller and expert sub-agents.

This module implements a comprehensive document analysis system using a controller
agent that coordinates multiple expert sub-agents for detailed academic paper analysis.
Each sub-agent specializes in extracting specific types of information from papers.

The system uses programmatic agent hand-off where the controller agent decides
which sub-agents to invoke based on abstract analysis, then the application
code orchestrates the execution and result synthesis.

REFACTORED: Simplified architecture with generic expert agents to reduce code
duplication and improve maintainability.
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..document.models import Chunk
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.coordinate_models import AnalysisPlan
from .models.coordinate_models import ComprehensiveAnalysisResult
from .models.coordinate_models import ExperimentResult
from .models.coordinate_models import FutureDirectionsResult
from .models.coordinate_models import MetadataExtractionResult
from .models.coordinate_models import MethodologyResult
from .models.coordinate_models import PreviousMethodsResult
from .models.coordinate_models import ResearchQuestionsResult
from .prompts.coordinate import CONTROLLER_INSTRUCTIONS
from .prompts.coordinate import EXPERIMENTS_SYSTEM_PROMPT
from .prompts.coordinate import FUTURE_DIRECTIONS_SYSTEM_PROMPT
from .prompts.coordinate import METADATA_EXTRACTION_SYSTEM_PROMPT
from .prompts.coordinate import METHODOLOGY_SYSTEM_PROMPT
from .prompts.coordinate import PREVIOUS_METHODS_SYSTEM_PROMPT
from .prompts.coordinate import RESEARCH_QUESTIONS_SYSTEM_PROMPT
from .prompts.coordinate import SYNTHESIS_SYSTEM_PROMPT
from .prompts.coordinate import build_analysis_planning_prompt
from .prompts.coordinate import build_experiments_analysis_prompt
from .prompts.coordinate import build_future_directions_analysis_prompt
from .prompts.coordinate import build_generic_analysis_prompt
from .prompts.coordinate import build_metadata_analysis_prompt
from .prompts.coordinate import build_methodology_analysis_prompt
from .prompts.coordinate import build_previous_methods_analysis_prompt
from .prompts.coordinate import build_report_synthesis_prompt
from .prompts.coordinate import build_research_questions_analysis_prompt

# Dependency classes for RunContext-based coordination


@dataclass
class CoordinateDeps:
    """Main dependencies for coordinate agent analysis."""

    document: Document
    custom_plan: AnalysisPlan | None = None
    max_retries: int = 3
    timeout: float = 300.0


@dataclass
class ExpertAgentDeps:
    """Dependencies for individual expert agents."""

    document: Document
    analysis_type: str
    sections_to_analyze: list[str] = field(default_factory=list)
    analysis_plan: AnalysisPlan | None = None


@dataclass
class SynthesisDeps:
    """Dependencies for report synthesis."""

    document: Document
    analysis_plan: AnalysisPlan
    sub_agent_results: dict[str, Any]
    paper_title: str = ""


# Expert agent configuration for RunContext-based execution

EXPERT_AGENT_CONFIG = {
    "metadata": {
        "system_prompt": METADATA_EXTRACTION_SYSTEM_PROMPT,
        "output_type": MetadataExtractionResult,
        "timeout": 60.0,
        "prompt_builder": build_metadata_analysis_prompt,
    },
    "previous_methods": {
        "system_prompt": PREVIOUS_METHODS_SYSTEM_PROMPT,
        "output_type": PreviousMethodsResult,
        "timeout": 120.0,
        "prompt_builder": build_previous_methods_analysis_prompt,
    },
    "research_questions": {
        "system_prompt": RESEARCH_QUESTIONS_SYSTEM_PROMPT,
        "output_type": ResearchQuestionsResult,
        "timeout": 120.0,
        "prompt_builder": build_research_questions_analysis_prompt,
    },
    "methodology": {
        "system_prompt": METHODOLOGY_SYSTEM_PROMPT,
        "output_type": MethodologyResult,
        "timeout": 120.0,
        "prompt_builder": build_methodology_analysis_prompt,
    },
    "experiments": {
        "system_prompt": EXPERIMENTS_SYSTEM_PROMPT,
        "output_type": ExperimentResult,
        "timeout": 120.0,
        "prompt_builder": build_experiments_analysis_prompt,
    },
    "future_directions": {
        "system_prompt": FUTURE_DIRECTIONS_SYSTEM_PROMPT,
        "output_type": FutureDirectionsResult,
        "timeout": 120.0,
        "prompt_builder": build_future_directions_analysis_prompt,
    },
}

EXPERT_SECTION_PREFERENCES = {
    "metadata": ["abstract", "introduction", "title"],
    "methodology": ["methodology", "method", "approach", "design", "technical approach"],
    "experiments": ["experiments", "experimental setup", "evaluation", "study design", "case study"],
    "evaluation": ["results", "evaluation", "findings", "outcomes", "performance"],
    "contributions": ["introduction", "contributions", "novelty", "innovation"],
    "limitations": ["limitations", "discussion", "conclusion", "future work"],
}


def _select_sections_for_expert(document: Document, analysis_type: str, planned_sections: list[str] | None) -> list[str]:
    """Choose sections for an expert by matching target patterns to document sections."""
    if planned_sections:
        targets = planned_sections
    elif analysis_type in EXPERT_SECTION_PREFERENCES:
        targets = EXPERT_SECTION_PREFERENCES[analysis_type]
    else:
        targets = ["abstract", "introduction", "methodology", "results", "conclusion"]

    matched: list[str] = []
    for target in targets:
        match = document.get_closest_section_name(target, threshold=0.7)
        if match and match not in matched:
            matched.append(match)

    if not matched:
        matched = document.get_section_names()[:3]

    return matched


def _build_expert_content(document: Document, analysis_type: str, sections_to_analyze: list[str] | None, max_tokens: int | None) -> str:
    """Assemble expert-oriented content using the unified Document helper."""
    section_names = _select_sections_for_expert(document, analysis_type, sections_to_analyze)
    return document.get_for_llm(
        section_names=section_names,
        max_tokens=max_tokens,
        include_headers=True,
        clean_text=True,
        max_chars_per_section=2500,
    )


class CoordinateAgent:
    """Controller agent for coordinating expert sub-agents in academic paper analysis.

    This controller agent uses programmatic agent hand-off to coordinate multiple
    expert sub-agents for comprehensive academic paper analysis. It analyzes
    the abstract to determine which sub-agents to invoke, orchestrates their
    execution, and synthesizes the results into a comprehensive report.

    REFACTORED: Simplified initialization and execution using configuration-driven
    agent dispatch instead of hardcoded agent instances.
    """

    def __init__(
        self,
        model: str | OpenAIChatModel | AnthropicModel,
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        """Initialize the CoordinateAgent controller.

        Args:
            model: Model identifier for the LLM provider
            max_retries: Maximum number of retries for failed requests
            timeout: Default timeout in seconds for controller operations
        """
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

        self.logger.info(f"Initialized CoordinateAgent controller with model: {self.model_identifier}")

        # Create main coordinate agent with dependencies and structured output
        self.coordinate_agent = Agent(
            model=self.model,
            deps_type=CoordinateDeps,
            output_type=AnalysisPlan,
            retries=self.max_retries,
        )

        # Add context-aware planning system prompt with better error handling
        @self.coordinate_agent.system_prompt
        async def planning_system_prompt(ctx: RunContext[CoordinateDeps]) -> str:
            """Generate system prompt for analysis planning."""
            deps = ctx.deps

            # Simple abstract extraction for now
            abstract = ""
            if deps.document.chunks:
                # Look for abstract in first few chunks
                for chunk in deps.document.chunks[:3]:
                    if chunk.content and ("abstract" in chunk.content.lower()[:200]):
                        abstract = chunk.content[:500]
                        break
                if not abstract and deps.document.chunks:
                    abstract = deps.document.chunks[0].content[:500]

            section_names = deps.document.get_section_names()

            # Validate document has content
            if not abstract and not section_names:
                raise ModelRetry(
                    "Document appears to have no extractable content or sections. "
                    "Please ensure the document is properly processed and contains readable text."
                )

            # Build planning prompt and combine with controller instructions
            planning_prompt = build_analysis_planning_prompt(abstract, section_names)
            return f"{CONTROLLER_INSTRUCTIONS}\n\n{planning_prompt}"

        self.logger.debug("CoordinateAgent controller initialized successfully")

    def _create_expert_agent(self, analysis_type: str) -> Agent[ExpertAgentDeps, BaseModel]:
        """Create an expert agent instance using RunContext.

        Args:
            analysis_type: Type of analysis (e.g., "metadata", "methodology")

        Returns:
            Configured Agent instance with RunContext
        """
        if analysis_type not in EXPERT_AGENT_CONFIG:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        config = EXPERT_AGENT_CONFIG[analysis_type]

        # Create expert agent with RunContext support
        agent = Agent(
            model=self.model,
            deps_type=ExpertAgentDeps,
            output_type=config["output_type"],
            retries=self.max_retries,
        )

        # Add context-aware system prompt with better error handling
        @agent.system_prompt
        async def expert_system_prompt(ctx: RunContext[ExpertAgentDeps]) -> str:
            """Generate system prompt for expert analysis."""
            deps = ctx.deps

            try:
                content = _build_expert_content(
                    document=deps.document,
                    analysis_type=deps.analysis_type,
                    sections_to_analyze=deps.sections_to_analyze if deps.sections_to_analyze else None,
                    max_tokens=6000,  # Reasonable limit for expert analysis
                )

                # Validate content is available
                if not content or not content.strip():
                    raise ModelRetry(f"No content found for {deps.analysis_type} analysis. Please check document sections and try again.")

                # Build analysis prompt using the appropriate prompt builder
                if "prompt_builder" in config:
                    return config["prompt_builder"](content)
                else:
                    return build_generic_analysis_prompt(content)

            except Exception as e:
                get_logger(__name__).warning(f"Expert content assembly failed for {deps.analysis_type}: {e}")
                raise ModelRetry(f"Unable to assemble content for {deps.analysis_type}.") from e

        return agent

    def display_analysis_plan(self, analysis_plan: AnalysisPlan, section_names: list[str]) -> str:
        """Create a clear, readable display of the analysis plan.

        Args:
            analysis_plan: The analysis plan to display
            section_names: List of available sections in the document

        Returns:
            Formatted string displaying the analysis plan
        """
        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        plan_display = [
            f"{BOLD}{CYAN}ANALYSIS PLAN{RESET}",
            f"{CYAN}{'=' * 80}{RESET}",
            "",
        ]

        analysis_configs = [
            (
                "analyze_metadata",
                "Metadata Extraction Agent",
                "First 3 chunks (title, authors, abstract)",
            ),
            (
                "analyze_previous_methods",
                "Previous Methods Agent",
                "previous_methods_sections",
            ),
            (
                "analyze_research_questions",
                "Research Questions Agent",
                "research_questions_sections",
            ),
            ("analyze_methodology", "Methodology Agent", "methodology_sections"),
            ("analyze_experiments", "Experiments Agent", "experiments_sections"),
            (
                "analyze_future_directions",
                "Future Directions Agent",
                "future_directions_sections",
            ),
        ]

        for plan_field, agent_name, sections_field in analysis_configs:
            if getattr(analysis_plan, plan_field):
                if sections_field == "First 3 chunks (title, authors, abstract)":
                    sections_display = sections_field
                else:
                    sections = getattr(analysis_plan, sections_field) or ["All sections"]
                    if sections == ["All sections"]:
                        sections_display = f"All sections {YELLOW}(fallback){RESET}"
                    else:
                        sections_display = f"{', '.join(sections)}"

                # Combined display with colors
                plan_display.append(f"{GREEN}[OK] {BOLD}{agent_name}{RESET}{GREEN} -> {BLUE}{sections_display}{RESET}")
                plan_display.append("")

        # Add reasoning section with color
        plan_display.extend(
            [
                f"{MAGENTA}{BOLD}Reasoning:{RESET}",
                f"{MAGENTA}{analysis_plan.reasoning}{RESET}",
                "",
            ]
        )

        # Add section selection quality assessment
        total_sections_selected = sum(
            [
                len(analysis_plan.previous_methods_sections or []),
                len(analysis_plan.research_questions_sections or []),
                len(analysis_plan.methodology_sections or []),
                len(analysis_plan.experiments_sections or []),
                len(analysis_plan.future_directions_sections or []),
            ]
        )

        # Quality assessment with colors
        if total_sections_selected == 0:
            plan_display.extend(
                [
                    f"{YELLOW}WARNING: No specific sections selected. All agents will use 'All sections'.{RESET}",
                    f"{YELLOW}   This may result in longer processing times and less focused analysis.{RESET}",
                    "",
                ]
            )
        elif total_sections_selected > 25:
            plan_display.extend(
                [
                    f"{BLUE}INFO: Many sections selected ({total_sections_selected}). Ensure all are relevant.{RESET}",
                    "",
                ]
            )
        else:
            plan_display.extend(
                [
                    f"{GREEN}Good: Comprehensive section selection with {total_sections_selected} relevant sections.{RESET}",
                    "",
                ]
            )

        if analysis_plan.estimated_relevance_scores:
            scores_text = ", ".join([f"{k}: {v:.2f}" for k, v in analysis_plan.estimated_relevance_scores.items()])
            plan_display.append(f"{CYAN}Relevance Scores: {scores_text}{RESET}")

        return "\n".join(plan_display)

    def _create_filtered_document(self, original_document: Document, chunks: list[Chunk]) -> Document:
        """Create a new Document object containing only specified chunks.

        Args:
            original_document: Original document to copy
            chunks: Chunks to include in the filtered document

        Returns:
            New Document object with filtered chunks
        """
        # Create new document with same metadata but filtered chunks
        # Note: We avoid deepcopy to prevent issues with file handles and loggers
        new_doc = Document(
            source_path=original_document.source_path,
            text=original_document.text,
            metadata=original_document.metadata,
            processing_state=original_document.processing_state,
        )
        new_doc._chunks = chunks
        new_doc._split = True  # Filtered documents are always split
        return new_doc

    def _deduplicate_sections(self, sections: list[str]) -> list[str]:
        """Remove duplicate section names while preserving order.

        Args:
            sections: List of section names that may contain duplicates

        Returns:
            List of unique section names in original order
        """
        return list(dict.fromkeys(sections))

    def _validate_pdf_document(self, document: Document) -> None:
        """Validate that document is a PDF file.

        Args:
            document: Document to validate

        Raises:
            ValueError: If document is not a PDF
        """
        if not document.source_path or document.source_path.suffix.lower() != ".pdf":
            raise ValueError(
                "CoordinateAgent only supports PDF files. "
                f"Got: {document.source_path.suffix if document.source_path else 'unknown file type'}. "
                "Please provide a PDF document."
            )

    def _log_analysis_plan(self, analysis_plan: AnalysisPlan, section_names: list[str]):
        """Log the analysis plan for debugging.

        Args:
            analysis_plan: The analysis plan to log
            section_names: Available sections in the document
        """
        plan_display = self.display_analysis_plan(analysis_plan, section_names)
        self.logger.info(f"Analysis Plan:\n{plan_display}")

    def extract_abstract(self, document: Document) -> str:
        """Extract abstract from document using semantic sectioning.

        Args:
            document: Document object to extract abstract from

        Returns:
            Abstract text as string
        """
        self.logger.info("Extracting abstract from document using semantic sections")

        # Try to find abstract section using semantic sectioning
        abstract_section_names = []
        section_names = document.get_section_names()

        # Look for section names that indicate abstract content
        for section_name in section_names:
            section_lower = section_name.lower()
            # Check for abstract-like section names (handle noise like numbers)
            if any(keyword in section_lower for keyword in ["abstract", "summary", "overview"]):
                abstract_section_names.append(section_name)
                self.logger.debug(f"Found potential abstract section: '{section_name}'")
                break  # Take the first match

        # If found abstract sections, extract content
        if abstract_section_names:
            abstract_chunks = document.get_sections_by_name(abstract_section_names)
            if abstract_chunks:
                abstract_content = " ".join([chunk.content for chunk in abstract_chunks])
                self.logger.info(f"Abstract extracted from semantic sections: {len(abstract_content)} characters")
                return abstract_content

        # Fallback: Use first two chunks as "abstract" for planning purposes
        if document.chunks and len(document.chunks) >= 2:
            first_two_chunks = document.chunks[:2]
            fallback_text = " ".join([chunk.content for chunk in first_two_chunks])
            # Limit to reasonable length (first ~2000 chars)
            fallback_text = fallback_text[:2000]
            self.logger.warning(
                f"No abstract section found, using first 2 chunks ({len(fallback_text)} chars) as fallback for analysis planning"
            )
            return fallback_text
        elif document.chunks and len(document.chunks) == 1:
            # Use the single chunk
            fallback_text = document.chunks[0].content[:2000]
            self.logger.warning(
                f"No abstract section found, using first chunk ({len(fallback_text)} chars) as fallback for analysis planning"
            )
            return fallback_text

        # Last resort: use beginning of full text
        text = document.get_full_text() if document.chunks else document.text
        if text:
            fallback_text = text[:2000]
            self.logger.warning("No abstract section found, using first 2000 chars of document as fallback for analysis planning")
            return fallback_text

        self.logger.error("No abstract found and no text available in document")
        return ""

    async def plan_analysis(self, document: Document, section_names: list[str]) -> AnalysisPlan:
        """Plan which sub-agents to use based on document analysis and available sections.

        Args:
            document: Document to analyze
            section_names: List of available sections in the document

        Returns:
            AnalysisPlan with selected sub-agents, sections, and reasoning
        """
        # Extract abstract from document
        abstract = self.extract_abstract(document)

        self.logger.info(
            f"Planning analysis based on abstract ({len(abstract) if abstract else 0} chars) and {len(section_names)} available sections"
        )

        if not abstract or not abstract.strip():
            self.logger.warning("Empty abstract provided, using default comprehensive analysis plan")
            return AnalysisPlan(
                analyze_metadata=True,
                analyze_previous_methods=True,
                analyze_research_questions=True,
                analyze_methodology=True,
                analyze_experiments=True,
                analyze_future_directions=True,
                previous_methods_sections=[],
                research_questions_sections=[],
                methodology_sections=[],
                experiments_sections=[],
                future_directions_sections=[],
                reasoning="No abstract available, using comprehensive analysis plan with all sections",
            )

        prompt = build_analysis_planning_prompt(abstract, section_names)

        try:
            self.logger.debug(f"Calling controller agent with prompt length: {len(prompt)} chars")
            self.logger.debug(f"Controller agent model: {self.model_identifier}")

            # Create dependencies for planning
            deps = CoordinateDeps(document=document)

            # Run the coordinate agent to generate analysis plan
            result = await asyncio.wait_for(
                self.coordinate_agent.run("Generate analysis plan", deps=deps),
                timeout=self.timeout,
            )

            self.logger.info("Analysis plan created successfully")
            self.logger.debug(f"Result type: {type(result)}, Output type: {type(result.output)}")
            # Log controller agent interaction at debug level
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            output_preview = str(result.output)[:200] + "..." if len(str(result.output)) > 200 else str(result.output)
            self.logger.debug(f"[controller_agent] Prompt ({len(prompt)} chars): {prompt_preview}")
            self.logger.debug(f"[controller_agent] Output ({len(str(result.output))} chars): {output_preview}")

            # Validate that we got an actual plan, not empty defaults
            if not result.output.reasoning or not result.output.reasoning.strip():
                self.logger.warning("Controller agent returned plan with empty reasoning - may indicate generation issue")

            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Analysis planning timed out after {self.timeout} seconds")
            # Log controller agent timeout
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self.logger.debug(f"[controller_agent] TimeoutError - Prompt ({len(prompt)} chars): {prompt_preview}")
            self.logger.error(f"[controller_agent] Timeout after {self.timeout} seconds")
            # Return default plan
            return AnalysisPlan(
                analyze_metadata=True,
                analyze_previous_methods=True,
                analyze_research_questions=True,
                analyze_methodology=True,
                analyze_experiments=True,
                analyze_future_directions=True,
                previous_methods_sections=[],
                research_questions_sections=[],
                methodology_sections=[],
                experiments_sections=[],
                future_directions_sections=[],
                reasoning="Planning timed out, using default comprehensive analysis with all sections",
            )

        except Exception as e:
            self.logger.error(f"Analysis planning failed: {e}")
            # Log controller agent error
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self.logger.debug(f"[controller_agent] ERROR: {e!s} - Prompt ({len(prompt)} chars): {prompt_preview}")
            # Return default plan
            return AnalysisPlan(
                analyze_metadata=True,
                analyze_previous_methods=True,
                analyze_research_questions=True,
                analyze_methodology=True,
                analyze_experiments=True,
                analyze_future_directions=True,
                previous_methods_sections=[],
                research_questions_sections=[],
                methodology_sections=[],
                experiments_sections=[],
                future_directions_sections=[],
                reasoning=f"Planning failed ({e}), using default comprehensive analysis with all sections",
            )

    async def execute_sub_agents(
        self,
        document: Document,
        analysis_plan: AnalysisPlan,
    ) -> dict[str, Any]:
        """Execute selected sub-agents based on analysis plan with section filtering.

        REFACTORED: Uses dynamic dispatch with configuration-driven agent creation
        instead of hardcoded agent instances and repetitive if-blocks.

        Args:
            document: Document object to analyze
            analysis_plan: Plan determining which agents to execute and which sections to use

        Returns:
            Dictionary containing results from executed sub-agents
        """
        self.logger.info("Executing sub-agents based on analysis plan")
        start_time = asyncio.get_event_loop().time()

        results = {}
        tasks = []
        sections_analyzed = {}

        # Define analysis tasks with their corresponding sections fields
        analysis_tasks = [
            (
                "analyze_metadata",
                "metadata",
                "metadata",
                None,
            ),  # Special case: uses first 3 chunks
            (
                "analyze_previous_methods",
                "previous_methods",
                "previous_methods_sections",
                "previous_methods_sections",
            ),
            (
                "analyze_research_questions",
                "research_questions",
                "research_questions_sections",
                "research_questions_sections",
            ),
            (
                "analyze_methodology",
                "methodology",
                "methodology_sections",
                "methodology_sections",
            ),
            (
                "analyze_experiments",
                "experiments",
                "experiments_sections",
                "experiments_sections",
            ),
            (
                "analyze_future_directions",
                "future_directions",
                "future_directions_sections",
                "future_directions_sections",
            ),
        ]

        # Dynamically create and dispatch tasks based on analysis plan
        for plan_field, agent_key, result_key, sections_field in analysis_tasks:
            if getattr(analysis_plan, plan_field):
                # Determine sections for analysis
                if agent_key == "metadata":
                    # Special case: metadata uses first 3 chunks
                    sections_analyzed[result_key] = [f"First {min(3, len(document.chunks))} chunks"]
                else:
                    # Use planned sections or full document
                    sections = getattr(analysis_plan, sections_field)
                    if sections:
                        # Deduplicate sections within this agent
                        deduplicated_sections = self._deduplicate_sections(sections)
                        sections_analyzed[result_key] = deduplicated_sections
                        self.logger.debug(f"{agent_key} agent using specific sections: {deduplicated_sections}")
                    else:
                        # Use all sections
                        sections_analyzed[result_key] = ["All sections"]
                        self.logger.warning(
                            f"{agent_key} agent: No sections specified, using 'All sections' fallback. This may result in less focused analysis."
                        )

                # Create agent and task dynamically
                agent = self._create_expert_agent(agent_key)

                # Create dependencies for this expert agent
                expert_deps = ExpertAgentDeps(
                    document=document,
                    analysis_type=agent_key,
                    sections_to_analyze=sections_analyzed[result_key] if result_key in sections_analyzed else [],
                )

                task = asyncio.create_task(
                    self._safe_execute_agent(
                        agent.run("Execute expert analysis", deps=expert_deps),
                        agent_key,
                    )
                )
                tasks.append((result_key, task))

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

        # Add sections analyzed information to results
        results["_sections_analyzed"] = sections_analyzed

        return results

    async def _safe_execute_agent(self, coro, agent_name: str):
        """Safely execute an agent with error handling."""
        try:
            result = await coro
            # Extract the actual output from AgentRunResult
            return result.output
        except Exception as e:
            self.logger.error(f"Agent {agent_name} execution failed: {e}")
            raise

    async def synthesize_report(
        self,
        analysis_plan: AnalysisPlan,
        sub_agent_results: dict[str, Any],
        document: Document,
    ) -> str:
        """Synthesize final comprehensive report from sub-agent results using native patterns.

        Args:
            analysis_plan: The analysis plan that was executed
            sub_agent_results: Results from executed sub-agents
            document: Original document for context

        Returns:
            Comprehensive synthesized report
        """
        self.logger.info("Synthesizing final report from sub-agent results using native patterns")
        if not isinstance(sub_agent_results, dict):
            self.logger.error(
                f"ERROR: sub_agent_results is not a dict! Type: {type(sub_agent_results)}, Content: {str(sub_agent_results)[:500]}"
            )
            raise TypeError(f"Expected dict for sub_agent_results, got {type(sub_agent_results)}")

        # Extract paper title for structured reporting
        paper_title = "Unknown Paper"
        if document.metadata and document.metadata.title:
            paper_title = document.metadata.title
        else:
            # Try to extract from metadata results
            if sub_agent_results.get("metadata", {}).get("success"):
                metadata_result = sub_agent_results["metadata"]["result"]
                if hasattr(metadata_result, "title") and metadata_result.title:
                    paper_title = metadata_result.title

        # Validate we have successful analyses to synthesize
        successful_analyses = [
            name for name, result in sub_agent_results.items() if result.get("success", False) and name != "_sections_analyzed"
        ]

        if not successful_analyses:
            self.logger.error("No successful analyses available for synthesis")
            return "No successful analyses available for report synthesis. Please try again with different settings."

        # Create synthesis agent with dependency support for validation
        synthesis_agent = Agent(
            model=self.model,
            deps_type=dict,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            retries=self.max_retries,
        )

        # Add context-aware system prompt for synthesis validation
        @synthesis_agent.system_prompt
        async def synthesis_system_prompt(ctx: RunContext[dict]) -> str:
            """Generate synthesis system prompt with analysis context."""
            # The successful_analyses validation happens before calling the agent
            # This decorator provides access to context if needed
            return SYNTHESIS_SYSTEM_PROMPT

        # Build synthesis prompt with structured format requirements
        prompt = build_report_synthesis_prompt(
            paper_title=paper_title,
            source_path=(str(document.source_path) if document.source_path else "text document"),
            abstract=self.extract_abstract(document),
            sub_agent_results=sub_agent_results,
        )

        try:
            result = await asyncio.wait_for(
                synthesis_agent.run(prompt, deps={"successful_analyses": successful_analyses}),
                timeout=self.timeout,
            )
            self.logger.info("Report synthesis completed successfully")
            # Log synthesis agent interaction at debug level
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            output_preview = str(result.output)[:200] + "..." if len(str(result.output)) > 200 else str(result.output)
            self.logger.debug(f"[synthesis_agent] Prompt ({len(prompt)} chars): {prompt_preview}")
            self.logger.debug(f"[synthesis_agent] Output ({len(str(result.output))} chars): {output_preview}")
            return result.output

        except asyncio.TimeoutError:
            self.logger.error(f"Report synthesis timed out after {self.timeout} seconds")
            # Log synthesis agent timeout
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self.logger.debug(f"[synthesis_agent] TimeoutError - Prompt ({len(prompt)} chars): {prompt_preview}")
            self.logger.error(f"[synthesis_agent] Timeout after {self.timeout} seconds")
            return f"Report synthesis timed out after {self.timeout} seconds. Please try again or use a shorter document."

        except Exception as e:
            self.logger.error(f"Report synthesis failed: {e}")
            # Log synthesis agent error
            prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            self.logger.debug(f"[synthesis_agent] ERROR: {e!s} - Prompt ({len(prompt)} chars): {prompt_preview}")
            return f"Report synthesis failed: {e}. Please try again."

    async def analyze(
        self,
        document: Document,
        custom_plan: AnalysisPlan | None = None,
        **kwargs: Any,
    ) -> ComprehensiveAnalysisResult:
        """Perform comprehensive document analysis using expert sub-agents.

        Args:
            document: Document object to analyze (must be PDF, should be created with to_markdown=True for best results)
                     Note: When using to_markdown=True, MinerU API caching is enabled by default. This means:
                     - First request: Downloads from API and caches the result
                     - Subsequent requests with same PDF: Uses cached result (instant, no API call)
                     - Cache is stored in ~/.sciread/mineru_cache by default
            custom_plan: Optional custom analysis plan (if None, will auto-generate)
            **kwargs: Additional arguments (currently unused but kept for compatibility)

        Returns:
            ComprehensiveAnalysisResult with all sub-analyses and final report

        Example:
            # Create document with MinerU (caching enabled by default)
            doc = Document.from_file("paper.pdf", to_markdown=True, auto_split=True)

            # First analysis: Calls MinerU API and caches result
            result = await tool_agent.analyze_document(doc)

            # Subsequent analyses with same PDF: Uses cache (instant)
            result2 = await tool_agent.analyze_document(doc)
        """
        # Step 0: Validate PDF document
        self._validate_pdf_document(document)

        self.logger.info(f"Starting comprehensive document analysis: {document.source_path}")
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Extract section names
            section_names = document.get_section_names()
            self.logger.debug(f"Found {len(section_names)} sections: {section_names}")

            # Step 2: Plan analysis (use custom plan if provided)
            if custom_plan:
                analysis_plan = custom_plan
                self.logger.info("Using custom analysis plan")
            else:
                analysis_plan = await self.plan_analysis(document, section_names)
                self.logger.info("Generated automatic analysis plan")

            # Step 4: Display and log the plan
            # print(self.display_analysis_plan(analysis_plan, section_names))  # Show user the plan
            self._log_analysis_plan(analysis_plan, section_names)

            # Step 5: Execute sub-agents
            sub_agent_results = await self.execute_sub_agents(document, analysis_plan)

            # Step 6: Synthesize final report
            self.logger.debug(f"About to synthesize report, sub_agent_results type: {type(sub_agent_results)}")
            self.logger.debug(
                f"sub_agent_results keys: {list(sub_agent_results.keys()) if isinstance(sub_agent_results, dict) else 'Not a dict'}"
            )
            final_report = await self.synthesize_report(analysis_plan, sub_agent_results, document)

            # Step 7: Build comprehensive result
            total_execution_time = asyncio.get_event_loop().time() - start_time

            comprehensive_result = ComprehensiveAnalysisResult(
                analysis_plan=analysis_plan,
                metadata_result=(
                    sub_agent_results.get("metadata", {}).get("result") if sub_agent_results.get("metadata", {}).get("success") else None
                ),
                previous_methods_result=(
                    sub_agent_results.get("previous_methods", {}).get("result")
                    if sub_agent_results.get("previous_methods", {}).get("success")
                    else None
                ),
                research_questions_result=(
                    sub_agent_results.get("research_questions", {}).get("result")
                    if sub_agent_results.get("research_questions", {}).get("success")
                    else None
                ),
                methodology_result=(
                    sub_agent_results.get("methodology", {}).get("result")
                    if sub_agent_results.get("methodology", {}).get("success")
                    else None
                ),
                experiment_result=(
                    sub_agent_results.get("experiments", {}).get("result")
                    if sub_agent_results.get("experiments", {}).get("success")
                    else None
                ),
                future_directions_result=(
                    sub_agent_results.get("future_directions", {}).get("result")
                    if sub_agent_results.get("future_directions", {}).get("success")
                    else None
                ),
                execution_summary={
                    "total_agents_executed": len(sub_agent_results) - 1,  # Exclude _sections_analyzed
                    "successful_agents": len(
                        [r for name, r in sub_agent_results.items() if name != "_sections_analyzed" and r.get("success", False)]
                    ),
                    "failed_agents": len(
                        [r for name, r in sub_agent_results.items() if name != "_sections_analyzed" and not r.get("success", False)]
                    ),
                    "agent_results": {
                        name: {"success": data.get("success", False)}
                        for name, data in sub_agent_results.items()
                        if name != "_sections_analyzed"
                    },
                },
                final_report=final_report,
                total_execution_time=total_execution_time,
                sections_analyzed=sub_agent_results.get("_sections_analyzed", {}),
            )

            self.logger.info(f"Comprehensive document analysis completed in {total_execution_time:.2f} seconds")
            return comprehensive_result

        except Exception as e:
            self.logger.error(f"Comprehensive document analysis failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the CoordinateAgent."""
        return f"CoordinateAgent(model={self.model_identifier})"
