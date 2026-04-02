"""Base interface for text splitters."""

from abc import ABC
from abc import abstractmethod

from sciread.document.models import Chunk


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

    def _extract_code_blocks(self, text: str, patterns: dict) -> tuple[str, list[dict]]:
        """Extract code blocks and replace them with placeholders.

        Args:
            text: Input text to process.
            patterns: Dictionary of compiled regex patterns containing 'fenced_code' and 'indented_code'.

        Returns:
            Tuple of (modified_text, list_of_code_blocks).
        """
        code_blocks = []
        placeholder_pattern = "__CODE_BLOCK_{}__"

        # Extract fenced code blocks
        if "fenced_code" in patterns:
            for i, match in enumerate(patterns["fenced_code"].finditer(text)):
                block_text = match.group(0)
                code_blocks.append(
                    {
                        "placeholder": placeholder_pattern.format(i),
                        "content": block_text,
                        "type": "fenced_code",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                text = text.replace(block_text, placeholder_pattern.format(i), 1)

        # Extract indented code blocks
        if "indented_code" in patterns:
            offset = len(code_blocks)
            for i, match in enumerate(patterns["indented_code"].finditer(text)):
                block_text = match.group(0)
                code_blocks.append(
                    {
                        "placeholder": placeholder_pattern.format(offset + i),
                        "content": block_text,
                        "type": "indented_code",
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
                text = text.replace(block_text, placeholder_pattern.format(offset + i), 1)

        return text, code_blocks

    def _restore_code_blocks(self, chunks: list[Chunk], code_blocks: list[dict]) -> list[Chunk]:
        """Restore extracted code blocks to their original positions.

        Args:
            chunks: List of chunks with placeholder content.
            code_blocks: List of extracted code block metadata.

        Returns:
            List of chunks with restored code blocks.
        """
        for chunk in chunks:
            for code_block in code_blocks:
                if code_block["placeholder"] in chunk.content:
                    chunk.content = chunk.content.replace(code_block["placeholder"], code_block["content"])
        return chunks
