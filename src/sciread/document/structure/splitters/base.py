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

    def _validate_chunk_overlap(self, chunk_overlap: int) -> int:
        """Validate overlap configuration shared by splitters."""
        if not isinstance(chunk_overlap, int):
            raise TypeError("chunk_overlap must be an integer")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        return chunk_overlap

    def _extract_trimmed_slice(self, text: str, start_char: int, end_char: int) -> tuple[str, tuple[int, int]]:
        """Extract a slice and synchronize its trimmed range with the content."""
        chunk_text = text[start_char:end_char]
        trimmed_text = chunk_text.strip()

        if not trimmed_text:
            return "", (start_char, start_char)

        leading_whitespace = len(chunk_text) - len(chunk_text.lstrip())
        trailing_whitespace = len(chunk_text) - len(chunk_text.rstrip())
        adjusted_start = start_char + leading_whitespace
        adjusted_end = end_char - trailing_whitespace
        return trimmed_text, (adjusted_start, adjusted_end)

    def _apply_chunk_overlap(self, text: str, chunks: list[Chunk]) -> list[Chunk]:
        """Apply backward overlap to chunk boundaries without changing default behavior."""
        chunk_overlap = self._validate_chunk_overlap(getattr(self, "chunk_overlap", 0))
        if chunk_overlap == 0 or len(chunks) < 2:
            self._sync_chunk_overlap_metadata(chunks)
            return chunks

        for index in range(1, len(chunks)):
            chunk = chunks[index]
            if chunk.char_range is None:
                continue

            _original_start, original_end = chunk.char_range
            overlap_start = max(0, _original_start - chunk_overlap)

            overlapped_content, overlapped_range = self._extract_trimmed_slice(text, overlap_start, original_end)
            if not overlapped_content:
                continue

            chunk.content = overlapped_content
            chunk.content_plain = overlapped_content
            chunk.display_text = overlapped_content
            chunk.retrieval_text = overlapped_content
            chunk.char_range = overlapped_range
            chunk.word_count = len(overlapped_content.split())
            chunk.token_count = chunk.word_count

        self._sync_chunk_overlap_metadata(chunks)
        return chunks

    def _sync_chunk_overlap_metadata(self, chunks: list[Chunk]) -> None:
        """Derive overlap metadata directly from the current chunk ranges."""
        for chunk in chunks:
            chunk.overlap_prev_chars = 0
            chunk.overlap_next_chars = 0

        for index in range(1, len(chunks)):
            previous_chunk = chunks[index - 1]
            current_chunk = chunks[index]

            if previous_chunk.char_range is None or current_chunk.char_range is None:
                continue

            previous_start, previous_end = previous_chunk.char_range
            current_start, current_end = current_chunk.char_range
            overlap_chars = max(0, min(previous_end, current_end) - max(previous_start, current_start))

            if overlap_chars <= 0:
                continue

            previous_chunk.overlap_next_chars = overlap_chars
            current_chunk.overlap_prev_chars = overlap_chars

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
