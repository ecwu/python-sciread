"""Tests for HybridSplitter."""

import pytest

from sciread.document.splitters.hybrid import HybridSplitter


class TestHybridSplitter:
    """Test cases for HybridSplitter."""

    @pytest.fixture
    def splitter(self):
        """Create a HybridSplitter instance."""
        return HybridSplitter(
            min_section_size=50,
            chunk_size=200,
            chunk_overlap=20,
            prefer_rule_based=True,
        )

    def test_splitter_name(self, splitter):
        """Test splitter name property."""
        assert "HybridSplitter" in splitter.splitter_name
        assert "True" in splitter.splitter_name

    def test_initialization(self):
        """Test splitter initialization."""
        # Test default initialization
        hybrid = HybridSplitter()
        assert hybrid.rule_based_splitter.min_section_size == 50
        assert hybrid.fixed_size_splitter.chunk_size == 1000
        assert hybrid.fixed_size_splitter.chunk_overlap == 100
        assert hybrid.prefer_rule_based is True

        # Test custom initialization
        hybrid = HybridSplitter(
            min_section_size=100,
            chunk_size=500,
            chunk_overlap=50,
            prefer_rule_based=False,
        )
        assert hybrid.rule_based_splitter.min_section_size == 100
        assert hybrid.fixed_size_splitter.chunk_size == 500
        assert hybrid.fixed_size_splitter.chunk_overlap == 50
        assert hybrid.prefer_rule_based is False

    def test_split_empty_text(self, splitter):
        """Test splitting empty text."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            splitter.split("")

    def test_split_invalid_input_type(self, splitter):
        """Test splitting invalid input types."""
        with pytest.raises(TypeError, match="Input text must be a string"):
            splitter.split(123)

    def test_split_with_sections(self, splitter, sample_academic_text):
        """Test splitting text with clear sections (should use rule-based)."""
        chunks = splitter.split(sample_academic_text)

        # Should detect sections and use rule-based splitting
        assert len(chunks) > 1

        # Should have some classified sections
        chunk_types = [chunk.chunk_type for chunk in chunks]
        classified_chunks = [ct for ct in chunk_types if ct != "unknown"]
        assert len(classified_chunks) > 0

    def test_split_without_sections(self, splitter):
        """Test splitting text without clear sections (should use fixed-size)."""
        text = "This is a long text without any clear section headings. " * 50
        chunks = splitter.split(text)

        # Should produce at least one chunk
        assert len(chunks) >= 1

        # Most chunks should be unknown type when no sections detected
        chunk_types = [chunk.chunk_type for chunk in chunks]
        unknown_chunks = [ct for ct in chunk_types if ct == "unknown"]
        assert len(unknown_chunks) > 0

    def test_rule_based_effective(self, splitter):
        """Test when rule-based splitting is effective."""
        text = """Abstract
This is the abstract content with sufficient length to be considered effective.

Introduction
This is the introduction section that is also sufficiently long to be considered an effective section. It contains multiple sentences and provides substantial content about the research topic.

Methods
This section describes the methodology used in the research with enough detail to be considered effective.

Results
The results section presents the findings with sufficient content to be effective.

Discussion
This discussion section interprets the results with adequate content."""
        chunks = splitter.split(text)

        # Should detect multiple sections
        assert len(chunks) >= 3

        # Should have high confidence sections
        classified_chunks = [chunk for chunk in chunks if chunk.chunk_type != "unknown"]
        assert len(classified_chunks) >= 2

    def test_prefer_rule_based_true(self, sample_academic_text):
        """Test preferring rule-based when both strategies are reasonable."""
        hybrid = HybridSplitter(prefer_rule_based=True)
        chunks = hybrid.split(sample_academic_text)

        # When both are reasonable and prefer_rule_based is True,
        # should prefer rule-based results
        chunk_types = [chunk.chunk_type for chunk in chunks]
        classified_chunks = [ct for ct in chunk_types if ct != "unknown"]
        # Should have some classified chunks from rule-based
        assert len(classified_chunks) > 0

    def test_prefer_rule_based_false(self, sample_academic_text):
        """Test preferring fixed-size when both strategies are reasonable."""
        hybrid = HybridSplitter(prefer_rule_based=False)
        chunks = hybrid.split(sample_academic_text)

        # When both are reasonable and prefer_rule_based is False,
        # should use fixed-size results
        assert len(chunks) > 0
        # Results should still be valid chunks

    def test_fallback_mechanism(self, splitter):
        """Test fallback to fixed-size when rule-based fails."""
        # Text that might confuse rule-based splitter
        text = "Weird formatted text\n\n1. Some number\n\n2. Another number\n\nNo clear sections really.\n" * 20
        chunks = splitter.split(text)

        # Should still produce chunks via fixed-size fallback
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content.strip()
            assert isinstance(chunk.position, int)
            assert isinstance(chunk.confidence, float)

    def test_chunk_properties(self, splitter, sample_academic_text):
        """Test that chunks have proper properties regardless of strategy."""
        chunks = splitter.split(sample_academic_text)

        for chunk in chunks:
            assert isinstance(chunk.content, str)
            assert len(chunk.content) > 0
            assert isinstance(chunk.chunk_type, str)
            assert isinstance(chunk.position, int)
            assert isinstance(chunk.confidence, float)
            assert chunk.confidence >= 0.0
            assert chunk.confidence <= 1.0
            assert isinstance(chunk.word_count, int)
            assert chunk.word_count > 0
            assert chunk.processed is False

    def test_fixed_size_reasonable_check(self):
        """Test the fixed-size reasonableness check."""
        hybrid = HybridSplitter(chunk_size=100, chunk_overlap=10)

        # Text that should produce reasonable fixed-size chunks
        text = "word " * 1000  # Long repetitive text
        chunks = hybrid.split(text)

        assert len(chunks) >= 1

        # Check average chunk size is reasonable
        avg_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
        assert 100 <= avg_size <= 5000

    def test_rule_based_effective_check(self, splitter):
        """Test the rule-based effectiveness check."""
        # Text that should trigger effective rule-based splitting
        text = """Abstract
This abstract provides a comprehensive overview of the research conducted in this paper. It presents the main findings and contributions of the work.

Introduction
The introduction section provides extensive background information about the research domain. It discusses the current state of the art and identifies the research gaps that this work aims to address.

Methods
The methods section describes in detail the experimental setup and procedures used in this research. It includes comprehensive information about data collection, analysis techniques, and validation procedures.

Results
The results section presents detailed findings from the experiments with comprehensive statistical analysis and interpretation of the outcomes.

Discussion
This discussion section provides in-depth analysis of the results and their implications for the field. It compares the findings with previous work and discusses the limitations and contributions of this research."""
        chunks = splitter.split(text)

        # Should detect this as effective rule-based splitting
        assert len(chunks) >= 3

        # Should have good classification coverage
        total_chunks = len(chunks)
        classified_chunks = sum(1 for chunk in chunks if chunk.chunk_type != "unknown")
        if total_chunks > 1:
            classification_ratio = classified_chunks / total_chunks
            assert classification_ratio > 0.5  # More than 50% should be classified

    def test_short_text_handling(self, splitter):
        """Test handling of very short text."""
        text = "Short text."
        chunks = splitter.split(text)

        # Should handle short text gracefully
        assert len(chunks) >= 1
        assert chunks[0].content == text
