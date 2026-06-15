"""Tests for the RegexSectionSplitter."""

import pytest

from sciread.document import Document
from sciread.document.structure.splitters.regex_section_splitter import RegexSectionSplitter


class TestRegexSectionSplitter:
    """Test cases for RegexSectionSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = RegexSectionSplitter()
        name = splitter.splitter_name
        assert "RegexSectionSplitter" in name
        assert "patterns=" in name
        assert "min_size=" in name

    def test_basic_splitting(self):
        """Test basic text splitting."""
        text = """Abstract

This is the abstract content.

Introduction

This is the introduction content.

Methods

This is the methods content."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        assert len(chunks) >= 2
        chunk_names = [chunk.chunk_name for chunk in chunks]
        assert "Abstract" in chunk_names or any(chunk.metadata.get("splitter") == "abstract" for chunk in chunks)
        assert "Introduction" in chunk_names or any(chunk.metadata.get("splitter") == "introduction" for chunk in chunks)
        assert any(chunk.metadata.get("splitter") == "methods" for chunk in chunks)

    def test_section_detection(self):
        """Test academic section detection."""
        text = """1. Introduction

This is section 1.

1.1 Background

This is a subsection.

2. Related Work

This is section 2."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        assert len(chunks) == 3
        chunk_names = [chunk.chunk_name for chunk in chunks]
        assert chunk_names == ["1 introduction", "1.1 background", "2 related work"]
        chunk_types = [chunk.metadata.get("splitter") for chunk in chunks]
        assert "introduction" in chunk_types
        assert "related_work" in chunk_types

    def test_confidence_scoring(self):
        """Test confidence scoring for different patterns."""
        text = """Abstract

This abstract contains enough text to exceed the minimum chunk size threshold and retain a high confidence score without being merged into adjacent content.

1. Introduction

This introduction contains enough text to exceed the minimum chunk size threshold and receive a solid confidence score for the introduction pattern.

Figure 1: Sample Figure

This figure caption and description provide the content for the figure chunk, which should receive a lower confidence score than the academic section chunks."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        by_type: dict[str, list] = {}
        for chunk in chunks:
            by_type.setdefault(chunk.metadata.get("splitter", "unknown"), []).append(chunk)

        assert "abstract" in by_type
        assert by_type["abstract"][0].confidence >= 0.8
        assert "introduction" in by_type
        assert 0.5 <= by_type["introduction"][0].confidence <= 0.9
        assert "figure" in by_type
        assert all(chunk.confidence <= 0.7 for chunk in by_type["figure"])

    def test_min_chunk_size_filtering(self):
        """Test minimum chunk size filtering."""
        text = """Abstract

Short intro.

Introduction

This is a much longer introduction content that should meet the minimum chunk size requirements. It contains multiple sentences and provides substantial information about the research topic and its significance in the field of study.

Methods

Another substantial piece of content that describes the methodology used in this research work with sufficient detail to meet the minimum size threshold."""

        splitter = RegexSectionSplitter(min_chunk_size=100)
        chunks = splitter.split(text)

        # Should have chunks with reasonable content
        assert len(chunks) >= 2

        # Check that chunks meet minimum size (except potentially the first one)
        substantial_chunks = list(chunks[1:])  # Skip first which might be small
        for chunk in substantial_chunks:
            assert len(chunk.content) >= 100 or chunk.confidence < 0.5

    def test_document_level_filtering(self):
        """Test that Document class handles confidence threshold filtering."""

        text = """Abstract

This is a comprehensive abstract that provides sufficient content to meet the minimum chunk size requirements. It contains a detailed summary of the research paper, including the methodology, key findings, and implications. This abstract is designed to be long enough to avoid the small chunk penalty that reduces confidence scores by 50%. The content discusses important aspects of the research and provides enough substance for meaningful analysis.

1. Introduction

This introduction section provides substantial background information and context for the research. It includes a comprehensive review of related work, establishes the research problem, and outlines the contributions of this paper. The introduction is deliberately made extensive to ensure it meets the minimum chunk size criteria and maintains high confidence scores without being penalized for brevity. This section sets the stage for the detailed methodology and results that follow in subsequent sections of the paper."""

        doc = Document.from_text(text)
        chunks = doc.get_chunks()  # Get all chunks

        assert len(chunks) > 0

        # Now test Document-level filtering
        high_quality_chunks = doc.get_quality_chunks(confidence_threshold=0.7)
        for chunk in high_quality_chunks:
            assert chunk.confidence >= 0.7

        # Should include abstract and introduction (high confidence patterns)
        chunk_splitter_types = [chunk.metadata.get("splitter") for chunk in high_quality_chunks]
        if len(high_quality_chunks) > 0:
            assert any(ct in ["abstract", "introduction", "section"] for ct in chunk_splitter_types)

    def test_custom_patterns(self):
        """Test adding custom patterns."""
        text = """START_SECTION

This is custom content.

END_SECTION

This is more content."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)

        # Add custom pattern
        splitter.add_custom_pattern("start_section", r"START_SECTION", confidence=0.8)

        chunks = splitter.split(text)

        # Should detect the custom pattern
        assert len(chunks) >= 1

        # Check that the splitter name reflects the added pattern
        name = splitter.splitter_name
        assert int(name.split("patterns=")[1].split(",")[0]) >= 15  # Default patterns + 1 custom

    def test_remove_pattern(self):
        """Test removing patterns."""
        splitter = RegexSectionSplitter()

        # Check that abstract pattern exists initially
        initial_count = len(splitter.patterns)
        assert "abstract" in splitter.patterns

        # Remove abstract pattern
        result = splitter.remove_pattern("abstract")
        assert result is True
        assert "abstract" not in splitter.patterns
        assert len(splitter.patterns) == initial_count - 1

        # Try to remove non-existent pattern
        result = splitter.remove_pattern("non_existent")
        assert result is False

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        splitter = RegexSectionSplitter()

        # Try to add invalid pattern
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            splitter.add_custom_pattern("invalid", "[unclosed_bracket")

    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = RegexSectionSplitter()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            splitter.split("")

    def test_non_string_input(self):
        """Test handling of non-string input."""
        splitter = RegexSectionSplitter()

        with pytest.raises(TypeError, match="Input text must be a string"):
            splitter.split(123)

    def test_text_without_clear_patterns(self):
        """Test text without clear section patterns."""
        text = """This is some plain text without clear academic section headers.
It just flows continuously without any specific structure or formatting.
The content is just a regular paragraph that continues on and on without
any obvious section breaks or structural indicators that would help with
chunking based on academic paper patterns."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        # Should still produce at least one chunk
        assert len(chunks) >= 1

        # Should have lower confidence for unstructured text
        for chunk in chunks:
            if chunk.chunk_name == "unknown":
                assert chunk.confidence < 0.5

    def test_char_ranges(self):
        """Test character range calculation."""
        text = """Abstract

This is the abstract.

Introduction

This is the introduction."""

        splitter = RegexSectionSplitter()
        chunks = splitter.split(text)

        # Check that chunks have character ranges
        for chunk in chunks:
            assert chunk.char_range is not None
            assert isinstance(chunk.char_range, tuple)
            assert len(chunk.char_range) == 2
            assert chunk.char_range[0] <= chunk.char_range[1]

            # Verify the character range matches the content (allowing for whitespace differences)
            start, end = chunk.char_range
            expected_content = text[start:end]
            # Strip leading/trailing whitespace from expected content to match chunk content
            expected_content = expected_content.strip()
            assert expected_content == chunk.content

    def test_chunk_positions(self):
        """Test chunk position sequencing."""
        text = """Abstract

Abstract content.

Introduction

Introduction content.

Methods

Methods content."""

        splitter = RegexSectionSplitter()
        chunks = splitter.split(text)

        # Check that positions are sequential
        positions = [chunk.position for chunk in chunks]
        assert positions == list(range(len(chunks)))

    def test_case_insensitive_patterns(self):
        """Test case-insensitive pattern matching."""
        text = """ABSTRACT

This is abstract in uppercase.

abstract

This is abstract in lowercase.

Abstract

This is abstract in title case."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        # Should detect at least abstract pattern (may be grouped into one chunk)
        abstract_chunks = [c for c in chunks if c.metadata.get("splitter") == "abstract"]
        assert len(abstract_chunks) >= 1

    def test_multiple_figure_table_references(self):
        """Test multiple figure and table references."""
        text = """Introduction

This is the introduction.

Figure 1: First figure

This describes figure 1.

Table 1: First table

This describes table 1.

Figure 2: Second figure

This describes figure 2."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        figure_table_chunks = [c for c in chunks if c.metadata.get("splitter") in ("figure", "table")]
        assert len(figure_table_chunks) >= 3
        chunk_names = [c.chunk_name for c in figure_table_chunks]
        assert any("Figure 1" in name for name in chunk_names)
        assert any("Figure 2" in name for name in chunk_names)
        assert any("Table 1" in name for name in chunk_names)

    def test_paragraph_break_pattern(self):
        """Test paragraph break detection."""
        text = """First paragraph with some content.


Second paragraph after double line break.


Third paragraph after more spacing."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        assert len(chunks) == 3
        assert all(chunk.confidence < 0.5 for chunk in chunks)
        assert [chunk.chunk_name for chunk in chunks] == [
            "First paragraph with some content.",
            "Second paragraph after double line break.",
            "Third paragraph after more spacing.",
        ]

    def test_output_chunk_contains_new_metadata_fields(self):
        """Test regex splitter initializes expanded chunk metadata fields."""
        text = """Abstract

This is a short abstract section for testing chunk metadata."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        assert len(chunks) >= 1
        chunk = chunks[0]

        assert chunk.chunk_id
        assert chunk.id == chunk.chunk_id
        assert chunk.content_plain == chunk.content
        assert isinstance(chunk.section_path, list)
        assert chunk.token_count is not None
        assert chunk.token_count > 0
        assert chunk.prev_chunk_id is None
        assert chunk.next_chunk_id is None
        assert chunk.citation_key == chunk.chunk_id
        assert chunk.retrievable is True
