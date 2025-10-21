"""Simple document analysis agent using pydantic-ai.

This module provides the main SimpleAgent class that can analyze academic papers
using pydantic-ai framework and the existing LLM provider infrastructure.
"""

import asyncio
from typing import Any
from typing import Optional
from typing import Union

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.simple_models import SimpleAnalysisResult
from .prompts.simple import DEFAULT_SYSTEM_PROMPT
from .prompts.simple import build_analysis_prompt
from .text_utils import clean_academic_text
from .text_utils import remove_references as remove_references_func


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

        # Create the pydantic-ai agent
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            retries=max_retries,
        )

        self.logger.info("SimpleAgent initialized successfully")

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
        self.logger.info(f"Starting document analysis for: {document.source_path or 'text document'}")

        # Get document text
        if document.chunks:
            text = document.get_full_text()
        else:
            text = document.text

        if not text or not text.strip():
            self.logger.error("Document has no text content to analyze")
            raise ValueError("Document has no text content to analyze")

        # Process text
        if remove_references:
            text = remove_references_func(text)
            self.logger.debug("Removed reference section from document text")

        if clean_text:
            text = clean_academic_text(text)
            self.logger.debug("Cleaned document text for better processing")

        # Build the full prompt
        document_metadata = None
        if include_metadata and document:
            document_metadata = {
                "source_path": str(document.source_path) if document.source_path else None,
                "title": document.metadata.title if document.metadata.title else None,
                "author": document.metadata.author if document.metadata.author else None,
            }

        full_prompt = build_analysis_prompt(
            text=text,
            task_prompt=task_prompt,
            document_metadata=document_metadata,
            **kwargs,
        )

        # Execute analysis
        try:
            self.logger.info("Executing document analysis with pydantic-ai agent")
            result = await asyncio.wait_for(
                self.agent.run(full_prompt),
                timeout=self.timeout,
            )
            self.logger.info("Document analysis completed successfully")
            return result.output

        except asyncio.TimeoutError as err:
            self.logger.error(f"Document analysis timed out after {self.timeout} seconds")
            raise TimeoutError(f"Document analysis timed out after {self.timeout} seconds") from err

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
        """Analyze a document and return structured results.

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
        result_text = await self.analyze(
            document=document,
            task_prompt=task_prompt,
            include_metadata=include_metadata,
            remove_references=remove_references,
            clean_text=clean_text,
            **kwargs,
        )

        # Return the result as a single report
        return SimpleAnalysisResult(report=result_text)

    def __repr__(self) -> str:
        """String representation of the SimpleAgent."""
        return f"SimpleAgent(model={self.model_identifier})"
