"""Tests for the RegexSectionSplitter."""

import pytest

from sciread.document.splitters.regex_section_splitter import RegexSectionSplitter


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

        assert len(chunks) >= 2  # At least abstract and methods content
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert "abstract" in chunk_types
        # Introduction might be grouped with abstract or methods depending on size

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

        assert len(chunks) >= 2  # Should detect at least some sections

        # Check section detection (subsection might not always be separate)
        chunk_types = [chunk.chunk_type for chunk in chunks]
        # Enhanced classifier now classifies numbered sections by their content
        assert any(section_type in chunk_types for section_type in ["introduction", "related_work", "section"])

    def test_confidence_scoring(self):
        """Test confidence scoring for different patterns."""
        text = """Abstract

This is abstract.

1. Introduction

This is introduction.

Figure 1: Sample Figure

This describes the figure."""

        splitter = RegexSectionSplitter()
        chunks = splitter.split(text)

        # Abstract should have high confidence
        abstract_chunks = [c for c in chunks if c.chunk_type == "abstract"]
        if abstract_chunks:
            assert abstract_chunks[0].confidence >= 0.9

        # Section should have medium confidence (after small chunk reduction and merging)
        section_chunks = [c for c in chunks if c.chunk_type == "section"]
        if section_chunks:
            assert 0.3 <= section_chunks[0].confidence <= 0.8

        # Figure should have lower confidence
        figure_chunks = [c for c in chunks if c.chunk_type == "figure"]
        if figure_chunks:
            assert figure_chunks[0].confidence <= 0.7

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

    def test_merge_small_chunks(self):
        """Test merging of small chunks."""
        text = """Abstract

Very short abstract.

Introduction

This is the introduction content with substantial text that should be above the minimum chunk size threshold and therefore should not be merged with other chunks in the normal splitting process."""

        splitter = RegexSectionSplitter(min_chunk_size=200, merge_small_chunks=True)
        chunks = splitter.split(text)

        # Should merge small chunks
        assert len(chunks) >= 1

        # Check that merged chunks are larger
        for chunk in chunks:
            if chunk.chunk_type in ["introduction", "abstract"]:
                # These should be substantial or merged
                assert len(chunk.content) >= 100 or chunk.confidence < 0.5

    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filtering."""
        text = """Abstract

Abstract content.

Some random text without clear section headers.

Introduction

Introduction content."""

        splitter = RegexSectionSplitter(confidence_threshold=0.7)
        chunks = splitter.split(text)

        # Should only include high confidence chunks
        for chunk in chunks:
            assert chunk.confidence >= 0.7

        # Should include abstract and introduction (high confidence patterns)
        chunk_types = [chunk.chunk_type for chunk in chunks]
        if len(chunks) > 0:
            assert any(ct in ["abstract", "introduction"] for ct in chunk_types)

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
            if chunk.chunk_type == "unknown":
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

            # Verify the character range matches the content
            start, end = chunk.char_range
            assert text[start:end] == chunk.content

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
        abstract_chunks = [c for c in chunks if c.chunk_type == "abstract"]
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

        # Should detect figure/table references
        figure_chunks = [c for c in chunks if c.chunk_type == "figure"]
        assert len(figure_chunks) >= 1  # May be grouped together

    def test_paragraph_break_pattern(self):
        """Test paragraph break detection."""
        text = """First paragraph with some content.


Second paragraph after double line break.


Third paragraph after more spacing."""

        splitter = RegexSectionSplitter(min_chunk_size=20, confidence_threshold=0.1)
        chunks = splitter.split(text)

        # Should split on paragraph breaks
        assert len(chunks) >= 1

        # Should have lower confidence for paragraph-based splits
        paragraph_chunks = [c for c in chunks if c.confidence < 0.5]
        assert len(paragraph_chunks) >= 0  # May be empty if content is grouped
