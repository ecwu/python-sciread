"""ConsecutiveFlow splitter: Adjacent sentence similarity-based chunking."""

from typing import Any

import regex

from ..external_clients import OllamaClient
from ..models import Chunk
from .base import BaseSplitter


class ConsecutiveFlowSplitter(BaseSplitter):
    """
    Sentence splitter that compares similarity between adjacent sentences.

    Uses consecutive similarity signal:
    - Compares each sentence with the next sentence (i → i+1)
    - Splits when similarity falls below threshold
    """

    # Enhanced sentence regex pattern from semantic-chunkers
    regex_pattern = r"""
        # Negative lookbehind for word boundary, word char, dot, word char
        (?<!\b\w\.\w.)
        # Negative lookbehind for single uppercase initials like "A."
        (?<!\b[A-Z][a-z]\.)
        # Negative lookbehind for abbreviations like "U.S."
        (?<!\b[A-Z]\.)
        # Negative lookbehind for abbreviations with uppercase letters and dots
        (?<!\b\p{Lu}\.\p{Lu}.)
        # Negative lookbehind for numbers, to avoid splitting decimals
        (?<!\b\p{N}\.)
        # Positive lookbehind for punctuation followed by whitespace
        (?<=\.|\?|\!|\:|\.\.\.)\s+
        # Positive lookahead for uppercase letter or opening quote at word boundary
        (?="?(?=[A-Z])|"\b)
        # OR
        |
        # Splits after punctuation that follows closing punctuation, followed by
        # whitespace
        (?<=[\"\'\]\)\}][\.!?])\s+(?=[\"\'\(A-Z])
        # OR
        |
        # Splits after punctuation if not preceded by a period
        (?<=[^\.][\.!?])\s+(?=[A-Z])
        # OR
        |
        # Handles splitting after ellipses
        (?<=\.\.\.)\s+(?=[A-Z])
        # OR
        |
        # Matches and removes control characters and format characters
        [\p{Cc}\p{Cf}]+
        # OR
        |
        # Splits after punctuation marks followed by another punctuation mark
        (?<=[\.!?])(?=[\.!?])
        # OR
        |
        # Splits after exclamation or question marks followed by whitespace or end of string
        (?<=[!?])(?=\s|$)
    """

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        # Similarity threshold
        similarity_threshold: float = 0.45,
        # Size constraints
        min_segment_sentences: int = 2,
        min_segment_chars: int = 200,
        max_segment_chars: int = 2000,
        # Processing parameters
        embedding_batch_size: int = 10,
    ):
        """
        Initialize ConsecutiveFlow splitter.

        Args:
            ollama_client: OllamaClient instance for embeddings (optional)
            similarity_threshold: Threshold for consecutive similarity (adjacent sentences)
            min_segment_sentences: Minimum sentences per segment
            min_segment_chars: Minimum characters per segment
            max_segment_chars: Maximum characters per segment (hard budget limit)
            embedding_batch_size: Number of sentences to embed in one request
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.similarity_threshold = similarity_threshold
        self.min_segment_sentences = min_segment_sentences
        self.min_segment_chars = min_segment_chars
        self.max_segment_chars = max_segment_chars
        self.embedding_batch_size = embedding_batch_size

        self.sentence_pattern = regex.compile(self.regex_pattern, regex.VERBOSE)

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"ConsecutiveFlowSplitter(model={self.ollama_client.model}, threshold={self.similarity_threshold})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using consecutive similarity algorithm."""
        text = self._validate_text(text)

        # Step 1: Extract sentences with metadata
        sentences = self._extract_sentences_with_metadata(text)
        if len(sentences) <= 1:
            return self._create_single_chunk(text)

        # Step 2: Get embeddings for all sentences
        embeddings = self._get_embeddings([s["text"] for s in sentences])
        if not embeddings:
            return self._fallback_split(text)

        # Step 3: Create chunks using consecutive similarity
        chunks = self._create_chunks_using_consecutive_similarity(sentences, embeddings)

        return chunks

    def _extract_sentences_with_metadata(self, text: str) -> list[dict[str, Any]]:
        """Extract sentences with char_span and metadata."""
        sentences = []
        _current_pos = 0

        # Use the regex pattern to split text
        split_positions = []
        for match in self.sentence_pattern.finditer(text):
            split_positions.append(match.end())

        # Add the end of text as final split position
        split_positions.append(len(text))

        prev_pos = 0
        sentence_id = 0

        for split_pos in split_positions:
            if split_pos > prev_pos:
                sentence_text = text[prev_pos:split_pos].strip()
                if sentence_text:  # Keep all non-empty fragments
                    sentences.append(
                        {
                            "id": sentence_id,
                            "text": sentence_text,
                            "char_span": (prev_pos, split_pos),
                            "length": len(sentence_text),
                        }
                    )
                    sentence_id += 1
            prev_pos = split_pos

        return sentences

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts using Ollama client."""
        return self.ollama_client.get_embeddings(texts, self.embedding_batch_size)

    def _create_chunks_using_consecutive_similarity(self, sentences: list[dict[str, Any]], embeddings: list[list[float]]) -> list[Chunk]:
        """Create chunks using consecutive similarity between adjacent sentences."""
        if len(sentences) != len(embeddings):
            return self._fallback_split(" ".join(s["text"] for s in sentences))

        chunks = []
        current_segment_sentences = [sentences[0]]
        current_segment_start = sentences[0]["char_span"][0]
        current_segment_chars = sentences[0]["length"]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            next_embedding = embeddings[i]
            prev_embedding = embeddings[i - 1]

            # Calculate consecutive similarity
            similarity_score = self.ollama_client.cosine_similarity(prev_embedding, next_embedding)

            # Check if adding this sentence would exceed budget
            would_exceed_budget = current_segment_chars + sentence["length"] > self.max_segment_chars

            # Check if we have enough content to make a split decision
            ready_for_split = (
                len(current_segment_sentences) >= self.min_segment_sentences and current_segment_chars >= self.min_segment_chars
            )

            # Decision logic
            should_split = False
            split_reason = None

            if would_exceed_budget:
                should_split = True
                split_reason = "budget"
            elif ready_for_split and similarity_score < self.similarity_threshold:
                should_split = True
                split_reason = "similarity_drop"

            if should_split:
                # Create chunk from current segment
                chunk = self._create_chunk_from_sentences(
                    current_segment_sentences,
                    current_segment_start,
                    split_reason,
                    similarity_score if split_reason == "similarity_drop" else None,
                )
                chunks.append(chunk)

                # Start new segment
                current_segment_sentences = [sentence]
                current_segment_start = sentence["char_span"][0]
                current_segment_chars = sentence["length"]
            else:
                # Add to current segment
                current_segment_sentences.append(sentence)
                current_segment_chars += sentence["length"]

        # Add final segment
        if current_segment_sentences:
            chunk = self._create_chunk_from_sentences(
                current_segment_sentences,
                current_segment_start,
                "final",
                None,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_sentences(
        self,
        sentences: list[dict[str, Any]],
        start_char: int,
        split_reason: str,
        similarity_score: float | None,
    ) -> Chunk:
        """Create a Chunk from a list of sentences."""
        # Combine sentences into content
        content = " ".join(s["text"] for s in sentences)

        # Calculate character range
        end_char = start_char + sum(s["length"] for s in sentences) + len(sentences) - 1  # Account for spaces

        # Calculate confidence based on split reason and segment quality
        confidence = self._calculate_chunk_confidence(sentences, split_reason, similarity_score)

        # Store metadata
        metadata = {
            "split_reason": split_reason,
            "sentence_count": len(sentences),
            "segment_chars": sum(s["length"] for s in sentences),
        }
        if similarity_score is not None:
            metadata["similarity_score"] = similarity_score

        chunk = Chunk(
            content=content,
            chunk_name=f"consecutive_chunk_{len(content)}",  # Give it a searchable name
            position=0,  # Will be set when added to chunks list
            char_range=(start_char, end_char),
            confidence=confidence,
            metadata={**metadata, "splitter": "consecutive_flow"},
        )
        return chunk

    def _calculate_chunk_confidence(
        self,
        sentences: list[dict[str, Any]],
        split_reason: str,
        similarity_score: float | None,
    ) -> float:
        """Calculate confidence score for a chunk."""
        if not sentences:
            return 0.0

        # Base confidence
        base_confidence = 0.7

        # Adjust based on split reason
        split_reason_boosts = {
            "final": 0.1,
            "budget": 0.05,
            "similarity_drop": 0.1,
        }

        reason_boost = split_reason_boosts.get(split_reason, 0.0)

        # Adjust based on similarity score if available
        similarity_boost = 0.0
        if similarity_score is not None:
            # Higher confidence for very clear splits (very low similarity)
            if similarity_score < 0.3:
                similarity_boost = 0.1
            elif similarity_score < 0.4:
                similarity_boost = 0.05

        # Adjust based on segment size
        sentence_count = len(sentences)
        if self.min_segment_sentences <= sentence_count <= 8:
            size_boost = 0.1
        elif sentence_count > 12:
            size_boost = -0.05
        else:
            size_boost = 0.0

        # Adjust based on content length
        total_chars = sum(s["length"] for s in sentences)
        if self.min_segment_chars <= total_chars <= 1000:
            length_boost = 0.05
        elif total_chars > 1500:
            length_boost = -0.05
        else:
            length_boost = 0.0

        confidence = base_confidence + reason_boost + similarity_boost + size_boost + length_boost
        return max(0.0, min(1.0, confidence))

    def _create_single_chunk(self, text: str) -> list[Chunk]:
        """Create a single chunk for short texts."""
        chunk = Chunk(
            content=text,
            chunk_name="document",
            position=0,
            char_range=(0, len(text)),
            confidence=0.8,
            metadata={"splitter": "consecutive_flow"},
        )
        return [chunk]

    def _fallback_split(self, text: str) -> list[Chunk]:
        """Fallback splitting when embeddings are not available."""
        # Simple paragraph-based fallback
        paragraphs = text.split("\n\n")
        chunks = []

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) >= self.min_segment_chars // 2:
                chunk = Chunk(
                    content=paragraph,
                    chunk_name=f"paragraph_{i + 1}",
                    position=i,
                    char_range=(0, len(paragraph)),
                    confidence=0.3,
                    metadata={"splitter": "fallback"},
                )
                chunks.append(chunk)

        return chunks if chunks else self._create_single_chunk(text)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.ollama_client.clear_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return self.ollama_client.get_cache_stats()

    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        return self.ollama_client.test_connection()
