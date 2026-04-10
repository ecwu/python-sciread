"""Shared retrieval models used by search-oriented agents."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from sciread.document.models import Chunk


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized retrieval result for any retrieval strategy."""

    chunk: Chunk
    score: float
    strategy: str
    matched_terms: list[str] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    expanded_context: str = ""

    @property
    def section_path_text(self) -> str:
        """Return a readable section path."""
        return " > ".join(self.section_path) if self.section_path else self.chunk.chunk_name
