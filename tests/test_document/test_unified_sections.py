"""Tests for unified section handling methods in Document class."""

import pytest

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.models import DocumentMetadata


class TestUnifiedSectionHandling:
    """Test cases for unified section handling methods."""

    @pytest.fixture
    def sample_document_with_sections(self):
        """Create a sample document with sections for testing."""
        # Create chunks with different section names
        chunks = [
            Chunk(
                content="This is the abstract of the paper with enough content to meet minimum length requirements.",
                chunk_name="abstract",
                position=0,
                confidence=0.9,
            ),
            Chunk(
                content="This is the introduction section that is also sufficiently long to pass the minimum length check.",
                chunk_name="introduction",
                position=1,
                confidence=0.8,
            ),
            Chunk(
                content="This describes our methodology in detail with sufficient length.",
                chunk_name="methodology",
                position=2,
                confidence=0.85,
            ),
            Chunk(content="These are our experimental results with detailed findings.", chunk_name="results", position=3, confidence=0.9),
            Chunk(content="Here we discuss the findings and their implications.", chunk_name="discussion", position=4, confidence=0.8),
            Chunk(content="We conclude the paper and summarize our contributions.", chunk_name="conclusion", position=5, confidence=0.85),
        ]

        doc = Document(text="Full document text...", metadata=DocumentMetadata(title="Sample Paper"))
        doc._set_chunks(chunks)
        return doc

    def test_print_for_human(self, sample_document_with_sections, capsys):
        """Test the print_for_human method."""
        doc = sample_document_with_sections

        # Test with default parameters
        doc.print_for_human()
        captured = capsys.readouterr()

        assert "Sample Paper" in captured.out
        assert "Section 1: abstract" in captured.out
        assert "Section 2: introduction" in captured.out
        assert "Confidence:" in captured.out

    def test_print_for_human_with_specific_sections(self, sample_document_with_sections, capsys):
        """Test print_for_human with specific sections."""
        doc = sample_document_with_sections

        doc.print_for_human(section_names=["abstract", "introduction"])
        captured = capsys.readouterr()

        assert "Section 1: abstract" in captured.out
        assert "Section 2: introduction" in captured.out
        assert "methodology" not in captured.out.lower()  # Should not be included

    def test_get_for_llm_basic(self, sample_document_with_sections):
        """Test basic get_for_llm functionality."""
        doc = sample_document_with_sections

        content = doc.get_for_llm()

        assert "=== ABSTRACT ===" in content
        assert "=== INTRODUCTION ===" in content
        assert "This is the abstract of the paper with enough content to meet minimum length requirements." in content
        assert "DOCUMENT METADATA:" in content
        assert "Sample Paper" in content

    def test_get_section_by_number(self, sample_document_with_sections):
        """Test get_section_by_number method."""
        doc = sample_document_with_sections

        # Test valid index
        section_name, content = doc.get_section_by_number(0)
        assert section_name == "abstract"
        assert "This is the abstract of the paper with enough content to meet minimum length requirements." in content

        # Test invalid index
        result = doc.get_section_by_number(10)
        assert result is None

    def test_get_section_by_name_exact(self, sample_document_with_sections):
        """Test get_section_by_name with exact matching."""
        doc = sample_document_with_sections

        section_name, content = doc.get_section_by_name("introduction")
        assert section_name == "introduction"
        assert "This is the introduction section that is also sufficiently long to pass the minimum length check." in content

    def test_get_section_by_name_fuzzy(self, sample_document_with_sections):
        """Test get_section_by_name with fuzzy matching."""
        doc = sample_document_with_sections

        # Test fuzzy matching
        section_name, content = doc.get_section_by_name("intro", fuzzy=True)
        assert section_name == "introduction"
        assert "This is the introduction section that is also sufficiently long to pass the minimum length check." in content

    def test_get_sections_content_helper(self, sample_document_with_sections):
        """Ensure helper returns ordered, truncated content consistently."""
        doc = sample_document_with_sections

        sections = doc.get_sections_content(max_sections=2)
        assert [name for name, _ in sections] == ["abstract", "introduction"]

        truncated = doc.get_sections_content(section_names=["abstract"], max_chars_per_section=20)
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

    def test_get_section_overview(self, sample_document_with_sections):
        """Test get_section_overview method."""
        doc = sample_document_with_sections

        overview = doc.get_section_overview()

        assert overview["document_title"] == "Sample Paper"
        assert overview["total_sections"] == 6
        assert overview["total_chunks"] == 6
        assert len(overview["sections"]) == 6

        # Check specific section info
        abstract_info = overview["sections"][0]
        assert abstract_info["name"] == "abstract"
        assert abstract_info["chunk_count"] == 1
        assert abstract_info["character_count"] > 0

    def test_get_sections_with_confidence(self, sample_document_with_sections):
        """Test get_sections_with_confidence method."""
        doc = sample_document_with_sections

        # Test with reasonable confidence threshold
        sections = doc.get_sections_with_confidence(min_confidence=0.7)

        assert len(sections) >= 4  # Should include sections with confidence >= 0.7

        # Check structure of returned data
        for section_name, content in sections:
            assert isinstance(section_name, str)
            assert isinstance(content, str)
            assert len(content) > 0

    def test_content_cleaning(self, sample_document_with_sections):
        """Test content cleaning functionality."""
        doc = sample_document_with_sections

        # Create chunk with cleaning artifacts
        dirty_content = "This  has    extra  spaces\n\n\nand newlines."
        chunk = Chunk(content=dirty_content, chunk_name="test_section", position=0, confidence=0.9)
        doc._set_chunks([chunk])

        content = doc.get_for_llm(clean_text=True)

        # Should be cleaned
        assert "  has    extra  spaces" not in content
        assert "\n\n\n" not in content

    def test_token_limiting(self, sample_document_with_sections):
        """Test token limiting functionality."""
        doc = sample_document_with_sections

        content = doc.get_for_llm(max_tokens=10)  # Very small limit

        # Should truncate content
        assert "...[truncated" in content or len(content) < 100

    def test_error_handling(self, sample_document_with_sections):
        """Test error handling in unified methods."""

        # Test with empty document
        empty_doc = Document(text="")

        # Should not crash
        overview = empty_doc.get_section_overview()
        assert overview["total_sections"] == 0

        # Should handle gracefully
        result = empty_doc.get_section_by_number(0)
        assert result is None
