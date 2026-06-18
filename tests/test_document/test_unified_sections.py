"""Tests for unified section handling methods in Document class."""

import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata
from sciread.document.structure.renderers import get_sections_content


class TestUnifiedSectionHandling:
    """Test cases for unified section handling methods."""

    @pytest.fixture
    def sample_document_with_sections(self):
        """Create a sample document with sections for testing."""
        # Create chunks with different section names
        chunks = [
            Chunk(
                content="This is the abstract of the paper with enough content to meet minimum length requirements.",
                section_path=["abstract"],
                para_index=0,
                metadata={"splitter_confidence": 0.9},
            ),
            Chunk(
                content="This is the introduction section that is also sufficiently long to pass the minimum length check.",
                section_path=["introduction"],
                para_index=1,
                metadata={"splitter_confidence": 0.8},
            ),
            Chunk(
                content="This describes our methodology in detail with sufficient length.",
                section_path=["methodology"],
                para_index=2,
                metadata={"splitter_confidence": 0.85},
            ),
            Chunk(
                content="These are our experimental results with detailed findings.",
                section_path=["results"],
                para_index=3,
                metadata={"splitter_confidence": 0.9},
            ),
            Chunk(
                content="Here we discuss the findings and their implications.",
                section_path=["discussion"],
                para_index=4,
                metadata={"splitter_confidence": 0.8},
            ),
            Chunk(
                content="We conclude the paper and summarize our contributions.",
                section_path=["conclusion"],
                para_index=5,
                metadata={"splitter_confidence": 0.85},
            ),
        ]

        doc = Document(
            text="Full document text...",
            metadata=DocumentMetadata(title="Sample Paper"),
        )
        doc._set_chunks(chunks)
        return doc

    def test_get_for_llm_basic(self, sample_document_with_sections):
        """Test basic get_for_llm functionality."""
        doc = sample_document_with_sections

        content = doc.get_for_llm()

        assert "=== ABSTRACT ===" in content
        assert "=== INTRODUCTION ===" in content
        assert "This is the abstract of the paper with enough content to meet minimum length requirements." in content
        assert "DOCUMENT METADATA:" in content
        assert "Sample Paper" in content

    def test_get_sections_by_name(self, sample_document_with_sections):
        """Test get_sections_by_name returns matching chunks."""
        doc = sample_document_with_sections

        intro_chunks = doc.get_sections_by_name(["introduction"])
        assert len(intro_chunks) == 1
        assert intro_chunks[0].section_path == ["introduction"]
        assert (
            "This is the introduction section that is also sufficiently long to pass the minimum length check." in intro_chunks[0].content
        )

    def test_get_sections_content_helper(self, sample_document_with_sections):
        """Ensure helper returns ordered, truncated content consistently."""
        doc = sample_document_with_sections

        sections = get_sections_content(doc, max_sections=2)
        assert [name for name, _ in sections] == ["abstract", "introduction"]

        truncated = get_sections_content(
            doc,
            section_names=["abstract"],
            max_chars_per_section=20,
        )
        assert truncated[0][1].endswith("...[truncated]")

    def test_get_closest_section_name(self, sample_document_with_sections):
        """Test get_closest_section_name method."""
        doc = sample_document_with_sections

        # Test exact match
        match = doc.get_closest_section_name("introduction")
        assert match == "introduction"

        # Test fuzzy match
        match = doc.get_closest_section_name("intro", threshold=0.7)
        assert match == "introduction"

        # Test no match
        match = doc.get_closest_section_name("nonexistent", threshold=0.9)
        assert match is None

    def test_get_closest_section_name_with_patterns(self, sample_document_with_sections):
        """Test pattern-based section matching."""
        doc = sample_document_with_sections

        # Test various patterns
        match = doc.get_closest_section_name("method", threshold=0.7)
        assert match == "methodology"

        match = doc.get_closest_section_name("findings", threshold=0.7)
        assert match == "results"

    def test_content_cleaning(self, sample_document_with_sections):
        """Test content cleaning functionality."""
        doc = sample_document_with_sections

        # Create chunk with cleaning artifacts
        dirty_content = "This  has    extra  spaces\n\n\nand newlines."
        chunk = Chunk(content=dirty_content, section_path=["test_section"], para_index=0, metadata={"splitter_confidence": 0.9})
        doc._set_chunks([chunk])

        content = doc.get_for_llm(clean_text=True)

        # Should be cleaned
        assert "  has    extra  spaces" not in content
        assert "\n\n\n" not in content

    def test_token_limiting(self):
        """Test token limiting functionality truncates an oversized section."""
        long_content = "word " * 400  # ~2400 chars, far above a small token budget
        chunks = [
            Chunk(content=long_content, section_path=["abstract"], para_index=0, metadata={"splitter_confidence": 0.9}),
            Chunk(content="Short section two.", section_path=["introduction"], para_index=1, metadata={"splitter_confidence": 0.8}),
        ]
        doc = Document(text="Full document text...", metadata=DocumentMetadata(title="Sample Paper"))
        doc._set_chunks(chunks)

        content = doc.get_for_llm(max_tokens=80, include_headers=False)

        # Should truncate the long abstract and omit the following section
        assert "...[truncated due to token limit]" in content
        assert "=== ABSTRACT ===" in content
        assert "=== INTRODUCTION ===" not in content

    def test_error_handling(self, sample_document_with_sections):
        """Test error handling in unified methods."""

        # Test with empty document
        empty_doc = Document(text="")

        # Should handle gracefully
        result = empty_doc.get_sections_by_name(["abstract"])
        assert result == []
