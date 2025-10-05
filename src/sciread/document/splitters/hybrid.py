"""Hybrid text splitter that combines rule-based and fixed-size strategies."""

from ..models import Chunk
from .base import BaseSplitter
from .fixed_size import FixedSizeSplitter
from .rule_based import RuleBasedSplitter


class HybridSplitter(BaseSplitter):
    """Hybrid splitter that uses rule-based splitting first, falls back to fixed-size."""

    def __init__(
        self,
        min_section_size: int = 50,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        prefer_rule_based: bool = True,
    ):
        """Initialize hybrid splitter with configuration."""
        self.rule_based_splitter = RuleBasedSplitter(min_section_size=min_section_size)
        self.fixed_size_splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.prefer_rule_based = prefer_rule_based

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"HybridSplitter(rule_based={self.prefer_rule_based})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using hybrid strategy."""
        text = self._validate_text(text)

        # Try rule-based splitting first
        rule_based_chunks = self.rule_based_splitter.split(text)

        # Evaluate the quality of rule-based splitting
        if self._is_rule_based_splitting_effective(rule_based_chunks, text):
            return rule_based_chunks

        # Fall back to fixed-size splitting
        fixed_size_chunks = self.fixed_size_splitter.split(text)

        # If both strategies produced reasonable results, prefer the configured one
        if self._is_fixed_size_splitting_reasonable(fixed_size_chunks, text):
            if self.prefer_rule_based and rule_based_chunks:
                return rule_based_chunks
            else:
                return fixed_size_chunks

        # Return the better result
        return rule_based_chunks if rule_based_chunks else fixed_size_chunks

    def _is_rule_based_splitting_effective(self, chunks: list[Chunk], original_text: str) -> bool:
        """Check if rule-based splitting was effective."""
        if not chunks:
            return False

        # Check if we have multiple sections with good classification
        classified_sections = sum(1 for chunk in chunks if chunk.chunk_type != "unknown")
        total_chunks = len(chunks)

        # Effective if we have multiple classified sections covering most of the text
        coverage_ratio = sum(len(chunk.content) for chunk in chunks) / len(original_text)
        has_multiple_sections = total_chunks > 1
        has_good_classification = classified_sections / total_chunks > 0.5

        return has_multiple_sections and has_good_classification and coverage_ratio > 0.8

    def _is_fixed_size_splitting_reasonable(self, chunks: list[Chunk], original_text: str) -> bool:
        """Check if fixed-size splitting produced reasonable results."""
        if not chunks:
            return False

        # Check if chunks are not too small or too large
        avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
        reasonable_size = 100 <= avg_chunk_size <= 5000

        # Check if we covered most of the text
        coverage_ratio = sum(len(chunk.content) for chunk in chunks) / len(original_text)

        return reasonable_size and coverage_ratio > 0.9
