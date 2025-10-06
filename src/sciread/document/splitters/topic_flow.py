"""TopicFlow splitter: Bottom-up sentence merging with semantic continuity detection."""

import math
from typing import Any
from typing import Optional

import regex
import requests

from ..models import Chunk
from .base import BaseSplitter


class TopicFlowSplitter(BaseSplitter):
    """
    Bottom-up sentence splitter that grows segments based on semantic continuity.

    Uses two continuity signals:
    - Local continuity: similarity between adjacent sentences (i → i+1)
    - Context continuity: similarity between current segment and next sentence
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
        (?<=\.|\?|!|:|\.\.\.)\s+
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
        model: str = "embeddinggemma:latest",
        base_url: str = "http://localhost:11434",
        # Continuity thresholds
        local_continuity_threshold: float = 0.6,
        context_continuity_threshold: float = 0.65,
        # Size constraints
        min_segment_sentences: int = 4,
        min_segment_chars: int = 300,
        max_segment_chars: int = 2000,
        # Processing parameters
        embedding_batch_size: int = 10,
        timeout: int = 30,
        cache_embeddings: bool = True,
        # Adaptive thresholds
        adaptive_floor: float = 0.4,
        soft_target: float = 0.7,
    ):
        """
        Initialize TopicFlow splitter.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama API base URL
            local_continuity_threshold: Threshold for local continuity (adjacent sentences)
            context_continuity_threshold: Threshold for context continuity (segment vs sentence)
            min_segment_sentences: Minimum sentences per segment for content-based cuts
            min_segment_chars: Minimum characters per segment
            max_segment_chars: Maximum characters per segment (hard budget limit)
            embedding_batch_size: Number of sentences to embed in one request
            timeout: Request timeout in seconds
            cache_embeddings: Whether to cache embeddings
            adaptive_floor: Adaptive floor for local continuity detection
            soft_target: Soft target for context continuity
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.local_continuity_threshold = local_continuity_threshold
        self.context_continuity_threshold = context_continuity_threshold
        self.min_segment_sentences = min_segment_sentences
        self.min_segment_chars = min_segment_chars
        self.max_segment_chars = max_segment_chars
        self.embedding_batch_size = embedding_batch_size
        self.timeout = timeout
        self.cache_embeddings = cache_embeddings
        self.adaptive_floor = adaptive_floor
        self.soft_target = soft_target

        self.sentence_pattern = regex.compile(self.regex_pattern, regex.VERBOSE)
        self.embedding_cache: dict[str, list[float]] = {}

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"TopicFlowSplitter(model={self.model}, min_sentences={self.min_segment_sentences})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using TopicFlow algorithm."""
        text = self._validate_text(text)

        # Step 1: Extract sentences with metadata
        sentences = self._extract_sentences_with_metadata(text)
        if len(sentences) <= 1:
            return self._create_single_chunk(text)

        # Step 2: Get embeddings for all sentences
        embeddings = self._get_embeddings([s["text"] for s in sentences])
        if not embeddings:
            return self._fallback_split(text)

        # Step 3: Grow segments using continuity signals
        segments = self._grow_segments(sentences, embeddings)

        # Step 4: Create chunks from segments
        chunks = self._create_chunks_from_segments(segments)

        return chunks

    def _extract_sentences_with_metadata(self, text: str) -> list[dict[str, Any]]:
        """Extract sentences with char_span and page metadata."""
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
                if sentence_text and len(sentence_text) > 10:  # Filter very short fragments
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
        """Get embeddings for texts using Ollama API."""
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            batch_embeddings = self._get_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def _get_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts."""
        batch_embeddings = []

        for text in texts:
            # Check cache first
            cache_key = f"{self.model}:{hash(text)}"
            if self.cache_embeddings and cache_key in self.embedding_cache:
                batch_embeddings.append(self.embedding_cache[cache_key])
                continue

            # Get embedding from Ollama
            try:
                embedding = self._get_single_embedding(text)
                if embedding:
                    batch_embeddings.append(embedding)
                    if self.cache_embeddings:
                        self.embedding_cache[cache_key] = embedding
                else:
                    batch_embeddings.append([0.0] * 768)  # Fallback
            except Exception:
                batch_embeddings.append([0.0] * 768)  # Fallback

        return batch_embeddings

    def _get_single_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for a single text from Ollama API."""
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {"model": self.model, "prompt": text}

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    return data["embedding"]

            return None
        except Exception:
            return None

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _grow_segments(self, sentences: list[dict[str, Any]], embeddings: list[list[float]]) -> list[dict[str, Any]]:
        """Grow segments using continuity signals."""
        if len(sentences) != len(embeddings):
            return [{"sentences": sentences, "cut_reason": "embedding_mismatch"}]

        segments = []
        current_segment = {
            "sentences": [sentences[0]],
            "embeddings": [embeddings[0]],
            "char_span": sentences[0]["char_span"],
            "cut_reason": None,
        }

        # Calculate adaptive thresholds based on text statistics
        avg_sentence_length = sum(s["length"] for s in sentences) / len(sentences)
        adaptive_local_threshold = self._calculate_adaptive_threshold(avg_sentence_length)

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            embedding = embeddings[i]

            # Calculate continuity signals
            local_continuity = self._cosine_similarity(embeddings[i - 1], embedding)
            context_continuity = self._calculate_context_continuity(current_segment, embedding)

            # Check budget constraint
            current_chars = sum(s["length"] for s in current_segment["sentences"])
            would_exceed_budget = current_chars + sentence["length"] > self.max_segment_chars

            # Check readiness for content-based cuts
            ready_for_content_cut = (
                len(current_segment["sentences"]) >= self.min_segment_sentences and current_chars >= self.min_segment_chars
            )

            # Decision logic
            should_cut = False
            cut_reason = None

            if would_exceed_budget:
                should_cut = True
                cut_reason = "budget"
            elif ready_for_content_cut:
                # Check content-based cut conditions
                local_drop = local_continuity < adaptive_local_threshold and local_continuity < self.adaptive_floor
                context_drop = context_continuity < self.soft_target

                if local_drop or context_drop:
                    should_cut = True
                    cut_reason = "local_drop" if local_drop else "context_drop"

            # Apply decision
            if should_cut:
                # Finalize current segment
                current_segment["cut_reason"] = cut_reason
                segments.append(current_segment)

                # Start new segment
                current_segment = {
                    "sentences": [sentence],
                    "embeddings": [embedding],
                    "char_span": sentence["char_span"],
                    "cut_reason": None,
                }
            else:
                # Add to current segment
                current_segment["sentences"].append(sentence)
                current_segment["embeddings"].append(embedding)

                # Update char span
                start_pos = current_segment["char_span"][0]
                end_pos = sentence["char_span"][1]
                current_segment["char_span"] = (start_pos, end_pos)

        # Add final segment
        if current_segment["sentences"]:
            current_segment["cut_reason"] = "final"
            segments.append(current_segment)

        return segments

    def _calculate_adaptive_threshold(self, avg_sentence_length: float) -> float:
        """Calculate adaptive local continuity threshold based on sentence length."""
        # Shorter sentences might have less stable embeddings, require lower threshold
        if avg_sentence_length < 50:
            return self.local_continuity_threshold - 0.1
        elif avg_sentence_length > 150:
            return self.local_continuity_threshold + 0.05
        return self.local_continuity_threshold

    def _calculate_context_continuity(self, segment: dict[str, Any], next_embedding: list[float]) -> float:
        """Calculate context continuity between segment and next sentence."""
        if not segment["embeddings"]:
            return 0.0

        # Calculate segment embedding (centroid of all sentence embeddings)
        segment_embedding = self._calculate_centroid(segment["embeddings"])

        return self._cosine_similarity(segment_embedding, next_embedding)

    def _calculate_centroid(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate centroid of embeddings."""
        if not embeddings:
            return []

        n = len(embeddings[0])
        centroid = [0.0] * n

        for embedding in embeddings:
            for i, value in enumerate(embedding):
                centroid[i] += value

        return [value / len(embeddings) for value in centroid]

    def _create_chunks_from_segments(self, segments: list[dict[str, Any]]) -> list[Chunk]:
        """Create Chunk objects from segments."""
        chunks = []

        for i, segment in enumerate(segments):
            # Combine sentences into content
            content = " ".join(s["text"] for s in segment["sentences"])

            # Calculate confidence based on segment quality
            confidence = self._calculate_segment_confidence(segment)

            chunk = Chunk(
                content=content,
                chunk_type="topic_flow",
                position=i,
                char_range=segment["char_span"],
                confidence=confidence,
            )
            chunks.append(chunk)

        return chunks

    def _calculate_segment_confidence(self, segment: dict[str, Any]) -> float:
        """Calculate confidence score for a segment."""
        if not segment["sentences"]:
            return 0.0

        # Base confidence
        base_confidence = 0.7

        # Adjust based on cut reason
        cut_reason_boosts = {
            "final": 0.1,
            "budget": 0.05,
            "local_drop": 0.1,
            "context_drop": 0.1,
            "embedding_mismatch": -0.2,
        }

        cut_reason = segment.get("cut_reason", "unknown")
        reason_boost = cut_reason_boosts.get(cut_reason, 0.0)

        # Adjust based on segment size (prefer segments with optimal sentence count)
        sentence_count = len(segment["sentences"])
        if self.min_segment_sentences <= sentence_count <= 8:
            size_boost = 0.1
        elif sentence_count > 12:
            size_boost = -0.05
        else:
            size_boost = 0.0

        # Adjust based on content length
        total_chars = sum(s["length"] for s in segment["sentences"])
        if self.min_segment_chars <= total_chars <= 1000:
            length_boost = 0.05
        elif total_chars > 1500:
            length_boost = -0.05
        else:
            length_boost = 0.0

        confidence = base_confidence + reason_boost + size_boost + length_boost
        return max(0.0, min(1.0, confidence))

    def _create_single_chunk(self, text: str) -> list[Chunk]:
        """Create a single chunk for short texts."""
        chunk = Chunk(
            content=text,
            chunk_type="topic_flow",
            position=0,
            char_range=(0, len(text)),
            confidence=0.8,
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
                    chunk_type="fallback",
                    position=i,
                    char_range=(0, len(paragraph)),
                    confidence=0.3,
                )
                chunks.append(chunk)

        return chunks if chunks else self._create_single_chunk(text)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self.embedding_cache),
            "model": self.model,
            "cache_enabled": self.cache_embeddings,
        }

    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
