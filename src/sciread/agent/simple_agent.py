"""Simple document analysis agent using pydantic-ai.

This module provides the main SimpleAgent class that can analyze academic papers
using pydantic-ai framework and the existing LLM provider infrastructure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .error_handling import (
    AgentError,
    AnalysisTimeoutError,
    handle_model_retry,
    safe_agent_execution,
    validate_document_content,
)
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
    additional_context: Dict[str, Any] = field(default_factory=dict)


class SimpleAgent:
    """Agent for analyzing academic documents and generating reports.

    This agent uses pydantic-ai to process academic papers and generate
    comprehensive analysis reports based on document content and custom prompts.
    """

    def __init__(
        self,
        model: Union[str, OpenAIChatModel, AnthropicModel],
        system_prompt: Optional[str] = None,
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

            # Process the document text based on dependencies
            if deps.document.chunks:
                text = deps.document.get_full_text()
            else:
                text = deps.document.text

            if not text or not text.strip():
                raise handle_model_retry(
                    ValueError("Document has no text content to analyze"),
                    "document content validation",
                    "Document appears to have no readable text. Please ensure the document is properly loaded and contains content."
                )

            # Process text based on dependencies
            if deps.remove_references:
                text = remove_references_func(text)
                self.logger.debug("Removed reference section from document text")

            if deps.clean_text:
                text = clean_academic_text(text)
                self.logger.debug("Cleaned document text for better processing")

            # Build the full prompt
            document_metadata = None
            if deps.include_metadata and deps.document:
                document_metadata = {
                    "source_path": (
                        str(deps.document.source_path)
                        if deps.document.source_path
                        else None
                    ),
                    "title": (
                        deps.document.metadata.title
                        if deps.document.metadata.title
                        else None
                    ),
                    "author": (
                        deps.document.metadata.author
                        if deps.document.metadata.author
                        else None
                    ),
                }

            full_prompt = build_analysis_prompt(
                text=text,
                task_prompt=deps.task_prompt,
                document_metadata=document_metadata,
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
        self.logger.debug(
            f"Starting document analysis for: {document.source_path or 'text document'}"
        )

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
                operation_name="document analysis"
            )
            self.logger.info(
                f"Document analysis completed successfully. Total characters in report: {len(result.output)}"
            )
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

            # Process the document text based on dependencies
            if deps.document.chunks:
                text = deps.document.get_full_text()
            else:
                text = deps.document.text

            if not text or not text.strip():
                raise handle_model_retry(
                    ValueError("Document has no text content to analyze"),
                    "document content validation",
                    "Document appears to have no readable text. Please ensure the document is properly loaded and contains content."
                )

            # Process text based on dependencies
            if deps.remove_references:
                text = remove_references_func(text)
            if deps.clean_text:
                text = clean_academic_text(text)

            # Build the full prompt
            document_metadata = None
            if deps.include_metadata and deps.document:
                document_metadata = {
                    "source_path": (
                        str(deps.document.source_path)
                        if deps.document.source_path
                        else None
                    ),
                    "title": (
                        deps.document.metadata.title
                        if deps.document.metadata.title
                        else None
                    ),
                    "author": (
                        deps.document.metadata.author
                        if deps.document.metadata.author
                        else None
                    ),
                }

            full_prompt = build_analysis_prompt(
                text=text,
                task_prompt=deps.task_prompt,
                document_metadata=document_metadata,
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
                operation_name="structured document analysis"
            )
            self.logger.info(
                f"Structured document analysis completed successfully. Report length: {len(result.output.report)}"
            )
            return result.output

        except Exception as e:
            self.logger.error(f"Structured document analysis failed: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the SimpleAgent."""
        return f"SimpleAgent(model={self.model_identifier})"
