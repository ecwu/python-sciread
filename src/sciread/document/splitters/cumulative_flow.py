"""CumulativeFlow splitter: Cumulative sentence similarity-based chunking."""

import uuid
from typing import Any

from ..models import Chunk
from .sentence_flow_base import SentenceFlowSplitter


class CumulativeFlowSplitter(SentenceFlowSplitter):
    """
    Sentence splitter that compares cumulative segment similarity with next sentence.

    Uses cumulative similarity signal:
    - Compares concatenated segment (all sentences in current segment) with next sentence
    - Splits when similarity falls below threshold
    """

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"CumulativeFlowSplitter(model={self.ollama_client.model}, threshold={self.similarity_threshold})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using cumulative similarity algorithm."""
        text = self._validate_text(text)

        # Step 1: Extract sentences with metadata
        sentences = self._extract_sentences_with_metadata(text)
        if len(sentences) <= 1:
            return self._create_single_chunk(text)

        # Step 2: Get embeddings for all sentences
        embeddings = self._get_embeddings([s["text"] for s in sentences])
        if not embeddings:
            return self._fallback_split(text)

        # Step 3: Create chunks using cumulative similarity
        chunks = self._create_chunks_using_cumulative_similarity(sentences, embeddings)

        return chunks

    def _create_chunks_using_cumulative_similarity(self, sentences: list[dict[str, Any]], embeddings: list[list[float]]) -> list[Chunk]:
        """Create chunks using cumulative similarity between segment and next sentence."""
        if len(sentences) != len(embeddings):
            return self._fallback_split(" ".join(s["text"] for s in sentences))

        chunks = []
        current_segment_sentences = [sentences[0]]
        current_segment_embeddings = [embeddings[0]]
        current_segment_start = sentences[0]["char_span"][0]
        current_segment_chars = sentences[0]["length"]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]

            # Calculate cumulative segment embedding (centroid of all sentence embeddings)
            segment_embedding = self.ollama_client.calculate_centroid(current_segment_embeddings)

            # Calculate similarity between cumulative segment and next sentence
            similarity_score = self.ollama_client.cosine_similarity(segment_embedding, sentence_embedding)

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
                split_reason = "cumulative_similarity_drop"

            if should_split:
                # Create chunk from current segment
                chunk = self._create_chunk_from_sentences(
                    current_segment_sentences,
                    current_segment_start,
                    split_reason,
                    (similarity_score if split_reason == "cumulative_similarity_drop" else None),
                )
                chunks.append(chunk)

                # Start new segment
                current_segment_sentences = [sentence]
                current_segment_embeddings = [sentence_embedding]
                current_segment_start = sentence["char_span"][0]
                current_segment_chars = sentence["length"]
            else:
                # Add to current segment
                current_segment_sentences.append(sentence)
                current_segment_embeddings.append(sentence_embedding)
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

        # Reassign positions to ensure continuity
        for i, chunk in enumerate(chunks):
            chunk.position = i

        return chunks

    def _create_chunk_from_sentences(
        self,
        sentences: list[dict[str, Any]],
        start_char: int,
        split_reason: str,
        similarity_score: float | None,
    ) -> Chunk:
        """Create a Chunk from a list of sentences with cumulative-specific metadata."""
        content = " ".join(s["text"] for s in sentences)
        end_char = start_char + sum(s["length"] for s in sentences) + len(sentences) - 1

        # Calculate confidence based on split reason
        confidence = 0.7
        split_reason_boosts = {
            "final": 0.1,
            "budget": 0.05,
            "cumulative_similarity_drop": 0.1,
        }
        reason_boost = split_reason_boosts.get(split_reason, 0.0)

        # Boost confidence based on similarity score
        similarity_boost = 0.0
        if similarity_score is not None:
            if similarity_score < 0.3:
                similarity_boost = 0.1
            elif similarity_score < 0.4:
                similarity_boost = 0.05

        confidence = max(0.0, min(1.0, confidence + reason_boost + similarity_boost))

        metadata = {
            "split_reason": split_reason,
            "sentence_count": len(sentences),
            "segment_chars": sum(s["length"] for s in sentences),
            "splitter": "cumulative_flow",
        }
        if similarity_score is not None:
            metadata["similarity_score"] = similarity_score

        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            content=content,
            chunk_id=chunk_id,
            doc_id="",
            content_plain=content,
            section_path=["cumulative"],
            page_start=None,
            page_end=None,
            para_index=0,
            chunk_name=f"cumulative_chunk_{len(content)}",
            position=0,
            char_range=(start_char, end_char),
            token_count=len(content.split()),
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_section_id="cumulative",
            citation_key=chunk_id,
            retrievable=True,
            confidence=confidence,
            metadata=metadata,
        )
        return chunk

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.ollama_client.clear_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return self.ollama_client.get_cache_stats()

    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server."""
        return self.ollama_client.test_connection()
