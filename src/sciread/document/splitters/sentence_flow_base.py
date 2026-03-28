"""Base class for sentence-similarity-based text splitters."""

import uuid
from typing import Any

import regex

from ...embedding_provider import OllamaClient
from ..models import Chunk
from .base import BaseSplitter


class SentenceFlowSplitter(BaseSplitter):
    """
    Abstract base class for sentence-similarity-based text splitters.

    Provides common functionality for splitters that:
    - Split text into sentences using regex
    - Compute embeddings for sentences
    - Use similarity scores to determine split points
    """

    # Enhanced sentence regex pattern from semantic-chunkers
    SENTENCE_REGEX_PATTERN = r"""
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
        # Splits after punctuation that follows closing punctuation, followed by whitespace
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
        similarity_threshold: float = 0.45,
        min_segment_sentences: int = 2,
        min_segment_chars: int = 200,
        max_segment_chars: int = 2000,
        embedding_batch_size: int = 10,
    ):
        """
        Initialize SentenceFlowSplitter base class.

        Args:
            ollama_client: OllamaClient instance for embeddings (optional).
            similarity_threshold: Threshold for similarity-based splits.
            min_segment_sentences: Minimum number of sentences per segment.
            min_segment_chars: Minimum characters per segment.
            max_segment_chars: Maximum characters per segment (hard limit).
            embedding_batch_size: Number of sentences to embed in one request.
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.similarity_threshold = similarity_threshold
        self.min_segment_sentences = min_segment_sentences
        self.min_segment_chars = min_segment_chars
        self.max_segment_chars = max_segment_chars
        self.embedding_batch_size = embedding_batch_size
        self.sentence_pattern = regex.compile(
            self.SENTENCE_REGEX_PATTERN, regex.VERBOSE
        )

    def _extract_sentences_with_metadata(self, text: str) -> list[dict[str, Any]]:
        """Extract sentences with char_span and metadata."""
        sentences = []

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
        """Get embeddings for texts using OllamaClient."""
        return self.ollama_client.get_embeddings(texts, self.embedding_batch_size)

    def _create_single_chunk(self, text: str) -> list[Chunk]:
        """Create a single chunk from text when no splitting is possible."""
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            content=text,
            chunk_id=chunk_id,
            doc_id="",
            content_plain=text,
            section_path=["document"],
            page_start=None,
            page_end=None,
            para_index=0,
            chunk_name="document",
            position=0,
            char_range=(0, len(text)),
            token_count=len(text.split()),
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_section_id="document",
            citation_key=chunk_id,
            retrievable=True,
            confidence=0.5,
            metadata={"splitter": "single_chunk"},
        )
        return [chunk]

    def _fallback_split(self, text: str) -> list[Chunk]:
        """Fallback to simple paragraph-based splitting when embeddings fail."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        for i, para in enumerate(paragraphs):
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                content=para,
                chunk_id=chunk_id,
                doc_id="",
                content_plain=para,
                section_path=["paragraph"],
                page_start=None,
                page_end=None,
                para_index=i,
                chunk_name="paragraph",
                position=i,
                char_range=(0, len(para)),
                token_count=len(para.split()),
                prev_chunk_id=None,
                next_chunk_id=None,
                parent_section_id="paragraph",
                citation_key=chunk_id,
                retrievable=True,
                confidence=0.3,
                metadata={"splitter": "fallback_paragraphs"},
            )
            chunks.append(chunk)

        return chunks if chunks else self._create_single_chunk(text)

    def _create_chunk_from_sentences(
        self,
        sentences: list[dict[str, Any]],
        start_pos: int,
        split_reason: str | None = None,
        similarity_score: float | None = None,
    ) -> Chunk:
        """Create a Chunk from a list of sentences."""
        content = " ".join(s["text"] for s in sentences)
        end_pos = start_pos + len(content)
        word_count = sum(len(s["text"].split()) for s in sentences)

        # Calculate confidence based on split reason
        confidence = 0.7
        if split_reason == "budget":
            confidence = 0.6
        elif split_reason == "similarity_drop":
            confidence = 0.8
            if similarity_score is not None:
                # Boost confidence if similarity drop was significant
                confidence = min(0.95, confidence + (1.0 - similarity_score) * 0.1)

        metadata = {
            "splitter": "sentence_flow",
            "sentence_count": len(sentences),
            "split_reason": split_reason,
        }
        if similarity_score is not None:
            metadata["similarity_score"] = similarity_score

        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            content=content,
            chunk_id=chunk_id,
            doc_id="",
            content_plain=content,
            section_path=["text_segment"],
            page_start=None,
            page_end=None,
            para_index=0,
            chunk_name="text_segment",
            position=0,
            char_range=(start_pos, end_pos),
            token_count=word_count,
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_section_id="text_segment",
            citation_key=chunk_id,
            retrievable=True,
            confidence=confidence,
            metadata=metadata,
            word_count=word_count,
        )
        return chunk
