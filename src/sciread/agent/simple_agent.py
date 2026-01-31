"""Simple document analysis agent using pydantic-ai.

This module provides the main SimpleAgent class that can analyze academic papers
using pydantic-ai framework and the existing LLM provider infrastructure.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .error_handling import handle_model_retry
from .error_handling import safe_agent_execution
from .models.simple_models import SimpleAnalysisResult
from .prompts.simple import DEFAULT_SYSTEM_PROMPT
from .prompts.simple import build_analysis_prompt
from .text_utils import clean_academic_text
from .text_utils import remove_references as remove_references_func


@dataclass
class SimpleAnalysisDeps:
    """Dependencies for SimpleAgent analysis operations."""

    document: Document
    task_prompt: str
    include_metadata: bool = True
    remove_references: bool = True
    clean_text: bool = True
    additional_context: dict[str, Any] = field(default_factory=dict)


def _build_simple_content(
    document: Document,
    include_metadata: bool,
    remove_references: bool,
    clean_text: bool,
    max_tokens: int | None,
) -> str:
    """Assemble document content for SimpleAgent using unified helpers."""
    section_names = document.get_section_names()
    if remove_references:
        section_names = [
            name for name in section_names if not any(keyword in name.lower() for keyword in ["reference", "bibliography", "citation"])
        ]

    content = document.get_for_llm(
        section_names=section_names or None,
        max_tokens=max_tokens,
        include_headers=include_metadata,
        clean_text=clean_text,
    )

    if content and content.strip():
        if remove_references:
            content = remove_references_func(content)
        return content

    # Fallback to raw text if sections are unavailable
    text = document.get_full_text() if document.chunks else document.text
    if remove_references:
        text = remove_references_func(text)
    if clean_text:
        text = clean_academic_text(text)
    return text


class SimpleAgent:
    """Agent for analyzing academic documents and generating reports.

    This agent uses pydantic-ai to process academic papers and generate
    comprehensive analysis reports based on document content and custom prompts.
    """

    def __init__(
        self,
        model: str | OpenAIChatModel | AnthropicModel,
        system_prompt: str | None = None,
        max_retries: int = 3,
        timeout: float = 300.0,
    ):
        """Initialize the SimpleAgent.

        Args:
            model: Either a model identifier string or a pydantic-ai model instance
            system_prompt: Optional system prompt for the agent
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for analysis requests
        """
        self.logger = get_logger(__name__)
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize model
        if isinstance(model, str):
            self.model_identifier = model
            self.model = get_model(model)
            self.logger.info(f"Initialized agent with model: {model}")
        else:
            self.model = model
            self.model_identifier = getattr(model, "model_name", "unknown")
            self.logger.info("Initialized agent with provided model instance")

        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Create the pydantic-ai agent with dependencies and structured output
        self.agent = Agent(
            model=self.model,
            deps_type=SimpleAnalysisDeps,
            output_type=str,
            retries=max_retries,
        )

        # Add context-aware system prompt
        @self.agent.system_prompt
        async def get_system_prompt(ctx: RunContext[SimpleAnalysisDeps]) -> str:
            """Generate system prompt with document context."""
            deps = ctx.deps

            text = _build_simple_content(
                document=deps.document,
                include_metadata=deps.include_metadata,
                remove_references=deps.remove_references,
                clean_text=deps.clean_text,
                max_tokens=8000,
            )

            if not text or not text.strip():
                raise handle_model_retry(
                    ValueError("Document has no text content to analyze"),
                    "document content validation",
                    "Document appears to have no readable text. Please ensure the document is properly loaded and contains content.",
                )

            full_prompt = build_analysis_prompt(
                text=text,
                task_prompt=deps.task_prompt,
                document_metadata=None,  # Metadata is already included in the processed text
                **deps.additional_context,
            )

            return f"{self.system_prompt}\n\n{full_prompt}"

        self.logger.debug("SimpleAgent initialized successfully")

    async def analyze(
        self,
        document: Document,
        task_prompt: str,
        include_metadata: bool = True,
        remove_references: bool = True,
        clean_text: bool = True,
        **kwargs: Any,
    ) -> str:
        """Analyze a document and generate a report.

        Args:
            document: Document object to analyze
            task_prompt: Specific task prompt for the analysis
            include_metadata: Whether to include document metadata in the analysis
            remove_references: Whether to remove reference sections from the text
            clean_text: Whether to clean the text before analysis
            **kwargs: Additional arguments to pass to the agent

        Returns:
            Generated analysis report as a string
        """
        self.logger.debug(f"Starting document analysis for: {document.source_path or 'text document'}")

        # Create dependencies object
        deps = SimpleAnalysisDeps(
            document=document,
            task_prompt=task_prompt,
            include_metadata=include_metadata,
            remove_references=remove_references,
            clean_text=clean_text,
            additional_context=kwargs,
        )

        # Execute analysis with unified error handling
        try:
            result = await safe_agent_execution(
                self.agent.run("Analyze the document according to the task", deps=deps),
                timeout=self.timeout,
                operation_name="document analysis",
            )
            self.logger.info(f"Document analysis completed successfully. Total characters in report: {len(result.output)}")
            return result.output

        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            raise

    async def analyze_structured(
        self,
        document: Document,
        task_prompt: str,
        include_metadata: bool = True,
        remove_references: bool = True,
        clean_text: bool = True,
        **kwargs: Any,
    ) -> SimpleAnalysisResult:
        """Analyze a document and return structured results using native output types.

        Args:
            document: Document object to analyze
            task_prompt: Specific task prompt for the analysis
            include_metadata: Whether to include document metadata in the analysis
            remove_references: Whether to remove reference sections from the text
            clean_text: Whether to clean the text before analysis
            **kwargs: Additional arguments to pass to the agent

        Returns:
            Structured analysis result
        """
        # Create a temporary agent with structured output type
        structured_agent = Agent(
            model=self.model,
            deps_type=SimpleAnalysisDeps,
            output_type=SimpleAnalysisResult,
            retries=self.max_retries,
        )

        # Use the same system prompt as the main agent
        @structured_agent.system_prompt
        async def get_system_prompt(ctx: RunContext[SimpleAnalysisDeps]) -> str:
            """Generate system prompt with document context."""
            deps = ctx.deps

            text = _build_simple_content(
                document=deps.document,
                include_metadata=deps.include_metadata,
                remove_references=deps.remove_references,
                clean_text=deps.clean_text,
                max_tokens=8000,  # Reasonable limit for simple analysis
            )

            if not text or not text.strip():
                raise handle_model_retry(
                    ValueError("Document has no text content to analyze"),
                    "document content validation",
                    "Document appears to have no readable text. Please ensure the document is properly loaded and contains content.",
                )

            full_prompt = build_analysis_prompt(
                text=text,
                task_prompt=deps.task_prompt,
                document_metadata=None,  # Metadata is already included in the processed text
                **deps.additional_context,
            )

            return f"{self.system_prompt}\n\n{full_prompt}"

        # Create dependencies object
        deps = SimpleAnalysisDeps(
            document=document,
            task_prompt=task_prompt,
            include_metadata=include_metadata,
            remove_references=remove_references,
            clean_text=clean_text,
            additional_context=kwargs,
        )

        # Execute analysis with unified error handling
        try:
            result = await safe_agent_execution(
                structured_agent.run("Analyze the document according to the task", deps=deps),
                timeout=self.timeout,
                operation_name="structured document analysis",
            )
            self.logger.info(f"Structured document analysis completed successfully. Report length: {len(result.output.report)}")
            return result.output

        except Exception as e:
            self.logger.error(f"Structured document analysis failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the SimpleAgent."""
        return f"SimpleAgent(model={self.model_identifier})"
