"""Document analysis agent using pydantic-ai.

This module provides the main DocumentAgent class that can analyze academic papers
using pydantic-ai framework and the existing LLM provider infrastructure.
"""

import asyncio
from typing import Any
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from ..document.document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .text_processor import clean_academic_text
from .text_processor import extract_document_metadata
from .text_processor import remove_references_section


class DocumentAnalysisResult(BaseModel):
    """Result of document analysis."""

    report: str


class DocumentAgent:
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
        """Initialize the DocumentAgent.

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
            self.model_identifier = getattr(model, 'model_name', 'unknown')
            self.logger.info("Initialized agent with provided model instance")

        # Default system prompt for academic document analysis
        default_system_prompt = """You are an expert academic document analyst with deep knowledge across multiple scientific disciplines. Your task is to carefully analyze academic papers and provide comprehensive, accurate, and insightful reports.

Key guidelines:
1. Read the entire document thoroughly before forming conclusions
2. Focus on the main research question, methodology, findings, and implications
3. Be objective and evidence-based in your analysis
4. Highlight both strengths and limitations of the research
5. Use clear, academic language appropriate for scholarly discourse
6. Structure your responses to be useful for researchers and students

Always provide citations or references to specific parts of the paper when making claims about its content."""

        self.system_prompt = system_prompt or default_system_prompt

        # Create the pydantic-ai agent
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            retries=max_retries,
        )

        self.logger.info("DocumentAgent initialized successfully")

    async def analyze_document(
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
            text = remove_references_section(text)
            self.logger.debug("Removed reference section from document text")

        if clean_text:
            text = clean_academic_text(text)
            self.logger.debug("Cleaned document text for better processing")

        # Build the full prompt
        full_prompt = self._build_analysis_prompt(
            text=text,
            task_prompt=task_prompt,
            document=document if include_metadata else None,
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

        except asyncio.TimeoutError:
            self.logger.error(f"Document analysis timed out after {self.timeout} seconds")
            raise TimeoutError(f"Document analysis timed out after {self.timeout} seconds")

        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            raise

    async def analyze_document_structured(
        self,
        document: Document,
        task_prompt: str,
        include_metadata: bool = True,
        remove_references: bool = True,
        clean_text: bool = True,
        **kwargs: Any,
    ) -> DocumentAnalysisResult:
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
        result_text = await self.analyze_document(
            document=document,
            task_prompt=task_prompt,
            include_metadata=include_metadata,
            remove_references=remove_references,
            clean_text=clean_text,
            **kwargs,
        )

        # Return the result as a single report
        return DocumentAnalysisResult(report=result_text)

    def _build_analysis_prompt(
        self,
        text: str,
        task_prompt: str,
        document: Optional[Document] = None,
        **kwargs: Any,
    ) -> str:
        """Build the full analysis prompt.

        Args:
            text: Document text content
            task_prompt: Specific task for the analysis
            document: Optional document object for metadata
            **kwargs: Additional context information

        Returns:
            Complete prompt for the agent
        """
        prompt_parts = []

        # Add task prompt first
        prompt_parts.append(f"ANALYSIS TASK:\n{task_prompt}")
        prompt_parts.append("")

        # Add document metadata if available
        if document:
            metadata_info = []
            if document.source_path:
                metadata_info.append(f"Source File: {document.source_path.name}")
            if document.metadata.title:
                metadata_info.append(f"Title: {document.metadata.title}")
            if document.metadata.author:
                metadata_info.append(f"Author: {document.metadata.author}")

            if metadata_info:
                prompt_parts.append("DOCUMENT METADATA:")
                for info in metadata_info:
                    prompt_parts.append(f"- {info}")
                prompt_parts.append("")

            # Try to extract additional metadata from text
            text_metadata = extract_document_metadata(text[:2000])  # Check first 2000 chars
            if text_metadata:
                prompt_parts.append("EXTRACTED METADATA:")
                for key, value in text_metadata.items():
                    if value:
                        prompt_parts.append(f"- {key.title()}: {value}")
                prompt_parts.append("")

        # Add the document text
        prompt_parts.append("DOCUMENT TEXT:")
        prompt_parts.append(text)
        prompt_parts.append("")

        # Add any additional context from kwargs
        if kwargs:
            prompt_parts.append("ADDITIONAL CONTEXT:")
            for key, value in kwargs.items():
                if value:
                    prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")

        # Add final instructions
        prompt_parts.append("Please provide a thorough analysis based on the document content and the specific task requirements above.")

        return "\n".join(prompt_parts)

  
    def __repr__(self) -> str:
        """String representation of the DocumentAgent."""
        return f"DocumentAgent(model={self.model_identifier})"