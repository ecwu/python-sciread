"""Fixed-size text splitter implementation."""

import re

from ..models import Chunk
from .base import BaseSplitter


class FixedSizeSplitter(BaseSplitter):
    """Split text into chunks of fixed size with optional overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize splitter with chunk size and overlap."""
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return (
            f"FixedSizeSplitter(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )

    def split(self, text: str) -> list[Chunk]:
        """Split text into fixed-size chunks."""
        text = self._validate_text(text)
        chunks = []

        # Clean up excessive whitespace
        text = re.sub(r"\s+", " ", text)

        if len(text) <= self.chunk_size:
            # If text is shorter than chunk size, return single chunk
            chunks.append(
                Chunk(
                    content=text,
                    chunk_type="unknown",
                    position=0,
                    char_range=(0, len(text)),
                    confidence=1.0,
                )
            )
            return chunks

        # Split into chunks with overlap
        start_pos = 0
        position = 0
        max_iterations = len(text) + 1000  # Safety limit
        iteration = 0

        while start_pos < len(text) and iteration < max_iterations:
            iteration += 1

            if iteration >= max_iterations - 1:
                # Safety break to prevent infinite loop
                import warnings

                warnings.warn(
                    "FixedSizeSplitter: Safety break triggered to prevent infinite loop"
                )
                break
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, len(text))

            # Try to end at word boundary if possible
            if end_pos < len(text):
                # Look for word boundary near the end
                word_boundary = text.rfind(
                    " ", start_pos + self.chunk_size // 2, end_pos
                )
                if word_boundary > start_pos + self.chunk_size // 2:
                    end_pos = word_boundary + 1  # Include the space

            chunk_text = text[start_pos:end_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_type="unknown",
                        position=position,
                        char_range=(start_pos, end_pos),
                        confidence=0.8,  # Lower confidence for arbitrary splits
                    )
                )
                position += 1

            # Move to next position with overlap
            new_start_pos = end_pos - self.chunk_overlap

            # Ensure progress is made
            if new_start_pos <= start_pos:
                start_pos = end_pos  # Move past current chunk
            else:
                start_pos = new_start_pos

        return chunks
