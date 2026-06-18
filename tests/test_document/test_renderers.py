"""Tests for document rendering and section-content helpers."""

import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.document.structure.renderers import choose_best_section_match
from sciread.document.structure.renderers import clean_section_content
from sciread.document.structure.renderers import format_for_llm
from sciread.document.structure.renderers import format_section_choices
from sciread.document.structure.renderers import get_section_length_map
from sciread.document.structure.renderers import get_sections_content
from sciread.document.structure.renderers import is_likely_heading_only
from sciread.document.structure.renderers import remove_references_section
from sciread.document.structure.renderers import resolve_section_names


@pytest.fixture
def sample_document():
    """Create a sample document with sections for renderer tests."""
    chunks = [
        Chunk(
            content="This is the abstract of the paper with enough content.",
            chunk_name="abstract",
            position=0,
            confidence=0.9,
        ),
        Chunk(
            content="This is the introduction section with background and motivation.",
            chunk_name="introduction",
            position=1,
            confidence=0.8,
        ),
        Chunk(
            content="This describes the methodology in detail with sufficient length.",
            chunk_name="methodology",
            position=2,
            confidence=0.85,
        ),
        Chunk(
            content="Short.",
            chunk_name="empty_transition",
            position=3,
            confidence=0.4,
        ),
    ]
    doc = Document(
        text="Full document text...",
        metadata=DocumentMetadata(title="Sample Paper", author="Alice Smith"),
    )
    doc._set_chunks(chunks)
    return doc


def test_resolve_section_names_honors_order_and_limits(sample_document):
    """Section-name resolution should preserve document order and cap at max_sections."""
    names = resolve_section_names(sample_document, max_sections=2)
    assert names == ["abstract", "introduction"]

    explicit = resolve_section_names(sample_document, section_names=["methodology", "abstract"])
    assert explicit == ["methodology", "abstract"]


def test_clean_section_content_normalizes_whitespace_and_hyphens():
    """Content cleaning should collapse excessive whitespace and fix hyphenation."""
    raw = 'word-\n\n  broken\n\n\nextra   spaces  "quoted"'
    cleaned = clean_section_content(raw)
    assert cleaned == 'wordbroken\n\nextra spaces "quoted"'


def test_remove_references_section_strips_bibliography():
    """Reference/bibliography sections should be removed while preserving prior content."""
    text = "Introduction\n\nBody content.\n\nReferences\n[1] Citation\n[2] Citation"
    assert remove_references_section(text) == "Introduction\n\nBody content."
    assert remove_references_section("No references here.") == "No references here."


def test_get_section_length_map_returns_clean_text_lengths(sample_document):
    """Section length map should report cleaned content lengths."""
    lengths = get_section_length_map(sample_document)
    assert lengths["abstract"] > 0
    assert lengths["introduction"] > 0
    assert lengths["methodology"] > 0
    assert lengths["empty_transition"] == len("Short.")


def test_is_likely_heading_only_uses_threshold():
    """Heading-only detection should compare length to the configured threshold."""
    assert is_likely_heading_only(50, threshold=80) is True
    assert is_likely_heading_only(100, threshold=80) is False


def test_format_section_choices_includes_length_and_short_hint():
    """Section choice formatting should annotate short sections."""
    sections = ["Abstract", "Methods", "Short"]
    lengths = {"Abstract": 120, "Methods": 30, "Short": 10}
    output = format_section_choices(sections, lengths, numbered=True)
    assert output.startswith("1. Abstract")
    assert "Methods | 30 chars | 可能仅标题" in output
    assert "Short | 10 chars | 可能仅标题" in output


def test_choose_best_section_match_prefers_substantial_content():
    """Best match should prefer sections with substantial content over heading-only ones."""
    available = ["Abstract", "Methods", "Short"]
    lengths = {"Abstract": 120, "Methods": 40, "Short": 10}
    assert choose_best_section_match("method", available, lengths, threshold=80) == "Methods"
    assert choose_best_section_match("short heading", available, lengths, threshold=80) == "Short"
    assert choose_best_section_match("missing", available, lengths) is None


def test_format_for_llm_truncates_oversized_section():
    """LLM formatting should truncate a section that exceeds the token budget."""
    long_content = "word " * 500
    chunks = [
        Chunk(content=long_content, chunk_name="abstract", position=0, confidence=0.9),
        Chunk(content="Second section.", chunk_name="introduction", position=1, confidence=0.8),
    ]
    doc = Document(text="Full text", metadata=DocumentMetadata(title="Paper"))
    doc._set_chunks(chunks)

    content = format_for_llm(doc, max_tokens=80, include_headers=False)
    assert "...[truncated due to token limit]" in content
    assert "=== INTRODUCTION ===" not in content


def test_format_for_llm_returns_error_message_on_failure():
    """LLM formatting should return an error string instead of raising."""
    from types import SimpleNamespace

    broken_doc = SimpleNamespace(
        metadata=SimpleNamespace(title="Broken", author=None),
        get_section_names=lambda: ["abstract"],
        logger=SimpleNamespace(error=lambda *args, **kwargs: None),
    )

    content = format_for_llm(broken_doc)
    assert content.startswith("Error retrieving content:")


def test_get_sections_content_orders_and_truncates(sample_document):
    """get_sections_content should return ordered name/content pairs and honor per-section limits."""
    sections = get_sections_content(sample_document, max_chars_per_section=20)
    names = [name for name, _ in sections]
    assert names == ["abstract", "introduction", "methodology", "empty_transition"]
    assert sections[0][1].endswith("...[truncated]")


def test_renderer_helpers_handle_empty_document():
    """Renderer helpers should degrade gracefully on empty documents."""
    empty_doc = Document(text="")
    assert get_sections_content(empty_doc) == []
    assert get_section_length_map(empty_doc) == {}
    assert format_for_llm(empty_doc) == ""
