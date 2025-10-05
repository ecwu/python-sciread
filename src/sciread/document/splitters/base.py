"""Base interface for text splitters."""

from abc import ABC
from abc import abstractmethod

from ..models import Chunk


class BaseSplitter(ABC):
    """Abstract base class for text splitters."""

    @property
    @abstractmethod
    def splitter_name(self) -> str:
        """Return the name of this splitter."""

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        """Split text into chunks."""

    def _validate_text(self, text: str) -> str:
        """Validate and clean input text."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")

        # Clean up common text issues
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text

    def _calculate_char_range(self, start_char: int, chunk_text: str) -> tuple[int, int]:
        """Calculate character range for a chunk."""
        end_char = start_char + len(chunk_text)
        return (start_char, end_char)
