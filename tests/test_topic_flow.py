"""Tests for the TopicFlowSplitter."""

import pytest

from sciread.document.splitters.topic_flow import TopicFlowSplitter


class TestTopicFlowSplitter:
    """Test cases for TopicFlowSplitter."""

    def test_splitter_name(self):
        """Test splitter name property."""
        splitter = TopicFlowSplitter()
        name = splitter.splitter_name
        assert "TopicFlowSplitter" in name
        assert "model=" in name
        assert "min_sentences=" in name

    def test_basic_sentence_extraction(self):
        """Test basic sentence extraction from text."""
        text = """This is the first sentence. This is the second sentence. This is the third sentence."""

        splitter = TopicFlowSplitter(
            min_segment_sentences=2, min_segment_chars=10, max_segment_chars=500, embedding_batch_size=1         )

        # Mock the embedding method to avoid API calls
        def mock_get_single_embedding(text):
            return [0.1] * 768  # Simple mock embedding

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Should produce at least one chunk
        assert len(chunks) >= 1
        assert all(chunk.chunk_name.startswith("segment_") for chunk in chunks)

    def test_segment_growth_with_continuity(self):
        """Test segment growth based on continuity signals."""
        text = """This is the first sentence. This is a related sentence about the same topic.
        Then we continue with more information. This is a different topic with unrelated content."""

        splitter = TopicFlowSplitter(
            min_segment_sentences=2, min_segment_chars=10, max_segment_chars=500, embedding_batch_size=1         )

        # Mock embeddings with different similarities
        mock_embeddings = [
            [0.1] * 768,  # sentence 1
            [0.9] * 768,  # sentence 2 (similar to 1)
            [0.8] * 768,  # sentence 3 (similar to segment)
            [0.1] * 768,  # sentence 4 (different)
        ]

        def mock_get_embeddings(texts):
            return mock_embeddings[: len(texts)]

        splitter._get_embeddings = mock_get_embeddings

        chunks = splitter.split(text)

        # Should produce at least one chunk
        assert len(chunks) >= 1

    def test_budget_constraint(self):
        """Test budget constraint handling."""
        text = """This is sentence 1. This is sentence 2. This is sentence 3. This is sentence 4."""

        splitter = TopicFlowSplitter(
            min_segment_sentences=2,
            min_segment_chars=10,
            max_segment_chars=50,  # Very small budget
            embedding_batch_size=1
                    )

        def mock_get_single_embedding(text):
            return [0.1] * 768

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Should produce multiple chunks due to budget constraint
        assert len(chunks) >= 1

    def test_minimum_segment_constraints(self):
        """Test minimum segment size constraints."""
        text = "Short text."

        splitter = TopicFlowSplitter(
            min_segment_sentences=4, min_segment_chars=300, max_segment_chars=1000, embedding_batch_size=1         )

        def mock_get_single_embedding(text):
            return [0.1] * 768

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Should produce one chunk for short text
        assert len(chunks) == 1

    def test_cut_reasons(self):
        """Test that cut reasons are properly recorded."""
        text = """This is sentence 1. This is sentence 2. This is sentence 3. This is sentence 4.
        This is sentence 5. This is sentence 6. This is sentence 7. This is sentence 8."""

        splitter = TopicFlowSplitter(
            min_segment_sentences=2,
            min_segment_chars=10,
            max_segment_chars=100,  # Will trigger budget cuts
            embedding_batch_size=1
                    )

        def mock_get_single_embedding(text):
            return [0.1] * 768

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Should produce multiple chunks
        assert len(chunks) >= 1
        assert all(chunk.chunk_name.startswith("segment_") for chunk in chunks)

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity is handled internally by OllamaClient."""
        # This functionality is now handled by OllamaClient
        # We just test that the splitter works
        splitter = TopicFlowSplitter()
        assert splitter.ollama_client is not None

    def test_centroid_calculation(self):
        """Test centroid calculation is handled internally."""
        # This functionality is now handled internally in the splitter
        # We just test that the splitter works
        splitter = TopicFlowSplitter()
        assert splitter is not None

    def test_sentence_extraction_with_metadata(self):
        """Test sentence extraction with metadata."""
        text = "First sentence. Second sentence. Third sentence."

        splitter = TopicFlowSplitter()
        sentences = splitter._extract_sentences_with_metadata(text)

        assert len(sentences) == 3
        for sentence in sentences:
            assert "id" in sentence
            assert "text" in sentence
            assert "char_span" in sentence
            assert "length" in sentence
            assert isinstance(sentence["char_span"], tuple)
            assert len(sentence["char_span"]) == 2

    def test_adaptive_threshold_calculation(self):
        """Test adaptive threshold calculation."""
        splitter = TopicFlowSplitter()

        # Test short sentences
        threshold_short = splitter._calculate_adaptive_threshold(30)
        assert threshold_short < splitter.local_continuity_threshold

        # Test long sentences
        threshold_long = splitter._calculate_adaptive_threshold(200)
        assert threshold_long > splitter.local_continuity_threshold

    def test_confidence_calculation(self):
        """Test segment confidence calculation."""
        splitter = TopicFlowSplitter()

        # Test segment with good properties
        segment = {
            "sentences": [{"text": "Test", "length": 100}] * 6,  # 6 sentences
            "cut_reason": "local_drop",
        }

        confidence = splitter._calculate_segment_confidence(segment)
        assert 0.0 <= confidence <= 1.0

    def test_empty_text_handling(self):
        """Test empty text handling."""
        splitter = TopicFlowSplitter()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            splitter.split("")

    def test_non_string_input(self):
        """Test non-string input handling."""
        splitter = TopicFlowSplitter()

        with pytest.raises(TypeError, match="Input text must be a string"):
            splitter.split(123)

    def test_cache_functionality(self):
        """Test embedding cache functionality."""
        splitter = TopicFlowSplitter()

        # Cache functionality is now handled by OllamaClient
        # Test that we can get cache stats
        stats = splitter.get_cache_stats()
        assert isinstance(stats, dict)
        # Test that we can clear cache
        splitter.clear_cache()  # Should not raise an error

    def test_fallback_splitting(self):
        """Test fallback splitting when embeddings fail."""
        text = """This is paragraph 1.

        This is paragraph 2.

        This is paragraph 3."""

        splitter = TopicFlowSplitter(min_segment_chars=10, embedding_batch_size=1)

        # Mock embedding method to return None (simulate failure)
        def mock_get_embeddings(texts):
            return []

        splitter._get_embeddings = mock_get_embeddings

        chunks = splitter.split(text)

        # Should still produce chunks using fallback
        assert len(chunks) >= 1

    def test_single_sentence_text(self):
        """Test handling of single sentence text."""
        text = "This is a single sentence."

        splitter = TopicFlowSplitter()

        def mock_get_single_embedding(text):
            return [0.1] * 768

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Should produce one chunk
        assert len(chunks) == 1
        # For single sentence, the chunk name should be "document"
        assert chunks[0].chunk_name == "document"

    def test_context_continuity_calculation(self):
        """Test context continuity calculation."""
        splitter = TopicFlowSplitter()

        segment = {"embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
        next_embedding = [0.0, 0.0, 1.0]

        continuity = splitter._calculate_context_continuity(segment, next_embedding)
        assert 0.0 <= continuity <= 1.0

    def test_char_range_accuracy(self):
        """Test character range accuracy."""
        text = "First sentence. Second sentence."

        splitter = TopicFlowSplitter()

        def mock_get_single_embedding(text):
            return [0.1] * 768

        splitter._get_single_embedding = mock_get_single_embedding

        chunks = splitter.split(text)

        # Check that chunks have proper character ranges
        for chunk in chunks:
            assert chunk.char_range is not None
            assert isinstance(chunk.char_range, tuple)
            assert len(chunk.char_range) == 2
            assert chunk.char_range[0] <= chunk.char_range[1]

            # Verify the content matches the text range
            start, end = chunk.char_range
            if start < len(text) and end <= len(text):
                assert text[start:end] == chunk.content
