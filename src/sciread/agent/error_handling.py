"""Common error handling patterns for pydantic-ai agents.

This module provides unified error handling patterns and utilities
to be used across all agent implementations.
"""

import asyncio
from typing import Any

from pydantic_ai import ModelRetry

from ..logging_config import get_logger

logger = get_logger(__name__)


class AgentError(Exception):
    """Base class for all agent-related errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class DocumentProcessingError(AgentError):
    """Raised when document processing fails."""


class ContentValidationError(AgentError):
    """Raised when content validation fails during analysis."""


class AnalysisTimeoutError(AgentError):
    """Raised when analysis operations timeout."""


class SubAgentExecutionError(AgentError):
    """Raised when sub-agent execution fails in coordinate analysis."""


def handle_model_retry(
    error: Exception, context: str, fallback_message: str | None = None
) -> ModelRetry:
    """Convert various exceptions to ModelRetry for pydantic-ai retry mechanism.

    Args:
        error: The original exception
        context: Context description for where the error occurred
        fallback_message: Optional custom message for the retry

    Returns:
        ModelRetry exception with descriptive message
    """
    error_msg = fallback_message or f"Error in {context}: {error!s}"
    logger.warning(f"Model retry triggered in {context}: {error}")

    if isinstance(error, ValueError | TypeError):
        return ModelRetry(
            f"Data validation error in {context}: {error!s}. Please check your input and try again."
        )
    if isinstance(error, TimeoutError):
        return ModelRetry(
            f"Timeout occurred in {context}: {error!s}. Please try with a shorter document or simpler task."
        )
    return ModelRetry(error_msg)


def validate_document_content(document: Any, operation: str = "analysis") -> None:
    """Validate document has content for processing.

    Args:
        document: Document object to validate
        operation: Description of the operation being performed

    Raises:
        DocumentProcessingError: If document lacks content
    """
    try:
        # Try to get content from document
        if hasattr(document, "get_full_text"):
            text = document.get_full_text()
        elif hasattr(document, "text"):
            text = document.text
        else:
            raise DocumentProcessingError(
                f"Document object doesn't have expected content methods for {operation}"
            )

        if not text or not text.strip():
            raise DocumentProcessingError(
                f"Document has no text content for {operation}. Please ensure the document is properly loaded and contains readable text."
            )

    except Exception as e:
        if isinstance(e, DocumentProcessingError):
            raise
        raise DocumentProcessingError(
            f"Failed to validate document content for {operation}: {e}"
        ) from e


def validate_section_content(
    document: Any, sections: list[str], analysis_type: str
) -> None:
    """Validate that sections have content for analysis.

    Args:
        document: Document object
        sections: List of section names to validate
        analysis_type: Type of analysis being performed

    Raises:
        ContentValidationError: If sections lack content
    """
    if not sections:
        raise ContentValidationError(
            f"No sections specified for {analysis_type} analysis"
        )

    try:
        # Check if document has section retrieval method
        if not hasattr(document, "get_sections_by_name"):
            raise ContentValidationError(
                f"Document doesn't support section retrieval for {analysis_type} analysis"
            )

        # Validate each section has content
        empty_sections = []
        for section_name in sections:
            section_chunks = document.get_sections_by_name([section_name])
            if not section_chunks:
                empty_sections.append(section_name)

        if empty_sections:
            raise ContentValidationError(
                f"No content found for sections: {empty_sections} "
                f"in {analysis_type} analysis. Please check document sections and try again."
            )

    except Exception as e:
        if isinstance(e, ContentValidationError):
            raise
        raise ContentValidationError(
            f"Failed to validate section content for {analysis_type} analysis: {e}"
        ) from e


async def safe_agent_execution(
    coro, timeout: float, operation_name: str, error_type: type[AgentError] = AgentError
) -> Any:
    """Safely execute an agent coroutine with timeout and error handling.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation_name: Description of the operation for error messages
        error_type: Type of error to raise on failure

    Returns:
        Result of the coroutine execution

    Raises:
        error_type: If execution fails or times out
    """
    try:
        logger.debug(f"Executing {operation_name} with timeout {timeout}s")
        result = await asyncio.wait_for(coro, timeout=timeout)
        logger.info(f"{operation_name} completed successfully")
        return result

    except TimeoutError as e:
        logger.error(f"{operation_name} timed out after {timeout}s")
        raise error_type(
            f"{operation_name} timed out after {timeout} seconds. Please try with a shorter document or simpler task."
        ) from e

    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        raise error_type(f"{operation_name} failed: {e}") from e


def format_error_for_user(error: AgentError, operation: str) -> str:
    """Format agent errors for user-friendly display.

    Args:
        error: The agent error that occurred
        operation: Description of the operation that failed

    Returns:
        User-friendly error message
    """
    if isinstance(error, DocumentProcessingError):
        return f"Document Error: {error}. Please check that your document is a valid PDF or text file with readable content."
    elif isinstance(error, ContentValidationError):
        return f"Content Error: {error}. The document may not have the expected sections or sufficient content for analysis."
    elif isinstance(error, AnalysisTimeoutError):
        return f"Timeout Error: {error}. Please try with a shorter document or simpler analysis task."
    elif isinstance(error, SubAgentExecutionError):
        return f"Analysis Error: {error}. Some parts of the analysis may have failed. Please try again or check the document quality."
    else:
        return f"Error in {operation}: {error}. Please try again or contact support if the issue persists."


def create_retry_message(
    original_error: Exception, context: str, suggestions: list[str] | None = None
) -> str:
    """Create a detailed retry message for ModelRetry.

    Args:
        original_error: The original error that occurred
        context: Context where the error occurred
        suggestions: Optional list of suggestions for retry

    Returns:
        Formatted retry message
    """
    message = f"Error in {context}: {original_error!s}"

    if suggestions:
        message += "\n\nSuggestions to resolve this issue:"
        for i, suggestion in enumerate(suggestions, 1):
            message += f"\n{i}. {suggestion}"

    return message
