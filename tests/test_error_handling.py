"""Tests for shared agent error-handling helpers."""

import asyncio

import pytest
from pydantic_ai import ModelRetry

from sciread.agent.shared.error_handling import AgentError
from sciread.agent.shared.error_handling import AnalysisTimeoutError
from sciread.agent.shared.error_handling import ContentValidationError
from sciread.agent.shared.error_handling import DocumentProcessingError
from sciread.agent.shared.error_handling import SubAgentExecutionError
from sciread.agent.shared.error_handling import create_retry_message
from sciread.agent.shared.error_handling import format_error_for_user
from sciread.agent.shared.error_handling import handle_model_retry
from sciread.agent.shared.error_handling import safe_agent_execution
from sciread.agent.shared.error_handling import validate_document_content
from sciread.agent.shared.error_handling import validate_section_content


def test_handle_model_retry_maps_common_error_types() -> None:
    """Value, timeout, and generic failures should produce tailored retry messages."""
    validation_retry = handle_model_retry(ValueError("bad field"), "insight parsing")
    timeout_retry = handle_model_retry(TimeoutError("too slow"), "discussion round")
    generic_retry = handle_model_retry(RuntimeError("boom"), "consensus build", fallback_message="retry later")

    assert isinstance(validation_retry, ModelRetry)
    assert "Data validation error in insight parsing" in str(validation_retry)
    assert "bad field" in str(validation_retry)

    assert isinstance(timeout_retry, ModelRetry)
    assert "Timeout occurred in discussion round" in str(timeout_retry)

    assert isinstance(generic_retry, ModelRetry)
    assert str(generic_retry) == "retry later"


def test_validate_document_content_supports_common_document_shapes() -> None:
    """Validation should accept both get_full_text and text-backed documents."""

    class GetterDocument:
        def get_full_text(self) -> str:
            return "paper body"

    class TextDocument:
        text = "paper body"

    validate_document_content(GetterDocument(), operation="summary")
    validate_document_content(TextDocument(), operation="summary")


def test_validate_document_content_raises_specific_errors() -> None:
    """Missing content or unexpected failures should be wrapped consistently."""

    class EmptyDocument:
        text = "   "

    class BrokenDocument:
        def get_full_text(self) -> str:
            raise RuntimeError("extract failed")

    with pytest.raises(DocumentProcessingError, match="Document has no text content for review"):
        validate_document_content(EmptyDocument(), operation="review")

    with pytest.raises(DocumentProcessingError, match="Failed to validate document content for review: extract failed"):
        validate_document_content(BrokenDocument(), operation="review")

    with pytest.raises(DocumentProcessingError, match="doesn't have expected content methods"):
        validate_document_content(object(), operation="review")


def test_validate_section_content_handles_missing_and_broken_sections() -> None:
    """Section validation should catch empty inputs, missing sections, and document failures."""

    class SectionDocument:
        def __init__(self, sections: dict[str, list[str]]) -> None:
            self.sections = sections

        def get_sections_by_name(self, names: list[str]) -> list[str]:
            return self.sections.get(names[0], [])

    class BrokenSectionDocument:
        def get_sections_by_name(self, _names: list[str]) -> list[str]:
            raise RuntimeError("lookup failed")

    with pytest.raises(ContentValidationError, match="No sections specified for synthesis analysis"):
        validate_section_content(SectionDocument({}), [], "synthesis")

    with pytest.raises(ContentValidationError, match="doesn't support section retrieval"):
        validate_section_content(object(), ["Methods"], "synthesis")

    with pytest.raises(ContentValidationError, match="No content found for sections: \\['Methods'\\]"):
        validate_section_content(SectionDocument({"Introduction": ["intro"]}), ["Methods"], "synthesis")

    with pytest.raises(ContentValidationError, match="Failed to validate section content for synthesis analysis: lookup failed"):
        validate_section_content(BrokenSectionDocument(), ["Methods"], "synthesis")

    validate_section_content(SectionDocument({"Methods": ["body"]}), ["Methods"], "synthesis")


@pytest.mark.asyncio
async def test_safe_agent_execution_returns_results_and_wraps_failures() -> None:
    """Coroutine execution should return values and preserve causes on failure."""

    async def succeed() -> str:
        return "done"

    async def fail() -> str:
        raise RuntimeError("agent broke")

    result = await safe_agent_execution(succeed(), timeout=0.1, operation_name="simple analysis")

    assert result == "done"

    with pytest.raises(SubAgentExecutionError, match="subagent run failed: agent broke") as exc_info:
        await safe_agent_execution(fail(), timeout=0.1, operation_name="subagent run", error_type=SubAgentExecutionError)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_safe_agent_execution_times_out() -> None:
    """Timeouts should be translated into the requested domain error type."""

    async def slow() -> str:
        await asyncio.sleep(0.05)
        return "too late"

    with pytest.raises(AnalysisTimeoutError, match=r"discussion timed out after 0\.01 seconds"):
        await safe_agent_execution(slow(), timeout=0.01, operation_name="discussion", error_type=AnalysisTimeoutError)


def test_format_error_for_user_and_create_retry_message_cover_all_variants() -> None:
    """User-facing formatting should vary by error type and include suggestions when present."""
    assert format_error_for_user(DocumentProcessingError("missing text"), "analysis").startswith("Document Error:")
    assert format_error_for_user(ContentValidationError("missing section"), "analysis").startswith("Content Error:")
    assert format_error_for_user(AnalysisTimeoutError("too slow"), "analysis").startswith("Timeout Error:")
    assert format_error_for_user(SubAgentExecutionError("child failed"), "analysis").startswith("Analysis Error:")
    assert format_error_for_user(AgentError("generic"), "analysis").startswith("Error in analysis:")

    message = create_retry_message(
        RuntimeError("boom"),
        "consensus",
        suggestions=["Reduce the document size", "Retry with fewer sections"],
    )

    assert "Error in consensus: boom" in message
    assert "1. Reduce the document size" in message
    assert "2. Retry with fewer sections" in message
