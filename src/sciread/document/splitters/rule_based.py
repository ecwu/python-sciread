"""Rule-based text splitter for academic papers."""

import re
from re import Pattern

from ..models import Chunk
from .base import BaseSplitter


class RuleBasedSplitter(BaseSplitter):
    """Split academic papers using section detection rules."""

    def __init__(self, min_section_size: int = 50):
        """Initialize splitter with minimum section size."""
        if min_section_size < 0:
            raise ValueError("Minimum section size cannot be negative")
        self.min_section_size = min_section_size

        # Section patterns for academic papers
        self.section_patterns: dict[str, Pattern] = {
            "abstract": re.compile(r"^\s*(?:abstract|summary)\s*[:.]?\s*$", re.IGNORECASE | re.MULTILINE),
            "introduction": re.compile(
                r"^\s*(?:introduction|introduction\s+and\s+background|overview)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "related_work": re.compile(
                r"^\s*(?:related\s+work|literature\s+review|background)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "methods": re.compile(
                r"^\s*(?:method(?:s|ology)?|approach|materials\s+and\s+methods|experimental\s+setup|design)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "results": re.compile(
                r"^\s*(?:results|findings|outcomes|evaluation)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "discussion": re.compile(
                r"^\s*(?:discussion|analysis|interpretation)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "conclusion": re.compile(
                r"^\s*(?:conclusion|conclusions|future\s+work|limitations)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "references": re.compile(
                r"^\s*(?:references|bibliography|works\s+cited)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
            "appendix": re.compile(
                r"^\s*(?:appendix|appendices|supplementary\s+material)\s*[:.]?\s*$",
                re.IGNORECASE | re.MULTILINE,
            ),
        }

        # Additional patterns for section detection
        self.numbered_section_pattern = re.compile(r"^\s*(\d+)\.\s+([A-Z][^\n]*?)\s*$", re.MULTILINE)
        self.subsection_pattern = re.compile(r"^\s*(\d+\.\d+)\s+([^\n]*?)\s*$", re.MULTILINE)

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"RuleBasedSplitter(min_size={self.min_section_size})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using academic paper section rules."""
        text = self._validate_text(text)
        chunks = []

        # Find all section boundaries
        section_boundaries = self._find_section_boundaries(text)

        if not section_boundaries:
            # No sections found, return single chunk
            chunks.append(
                Chunk(
                    content=text,
                    chunk_type="unknown",
                    position=0,
                    char_range=(0, len(text)),
                    confidence=0.5,  # Low confidence when no sections detected
                )
            )
            return chunks

        # Create chunks from sections
        position = 0
        for i, (start, end, section_type, confidence) in enumerate(section_boundaries):
            section_text = text[start:end].strip()
            if len(section_text) >= self.min_section_size:
                chunks.append(
                    Chunk(
                        content=section_text,
                        chunk_type=section_type,
                        position=position,
                        char_range=(start, end),
                        confidence=confidence,
                    )
                )
                position += 1

        return chunks

    def _find_section_boundaries(self, text: str) -> list[tuple[int, int, str, float]]:
        """Find section boundaries using various patterns."""
        boundaries = []
        text_lower = text.lower()

        # Check each section pattern
        for section_type, pattern in self.section_patterns.items():
            for match in pattern.finditer(text):
                start = match.start()
                # Find the end of this section (start of next section or end of text)
                end = self._find_section_end(text, start)
                boundaries.append((start, end, section_type, 1.0))

        # Check for numbered sections
        for match in self.numbered_section_pattern.finditer(text):
            section_title = match.group(2).strip()
            start = match.start()
            end = self._find_section_end(text, start)

            # Try to classify the section based on title
            section_type = self._classify_section_by_title(section_title)
            boundaries.append((start, end, section_type, 0.9))

        # Sort boundaries by start position
        boundaries.sort(key=lambda x: x[0])

        # Remove overlaps and merge adjacent boundaries
        return self._merge_overlapping_boundaries(boundaries)

    def _find_section_end(self, text: str, section_start: int) -> int:
        """Find the end of a section."""
        section_patterns = list(self.section_patterns.values())
        section_patterns.extend([self.numbered_section_pattern, self.subsection_pattern])

        # Look for the next section heading
        earliest_end = len(text)
        for pattern in section_patterns:
            for match in pattern.finditer(text, section_start + 1):
                if match.start() > section_start + 50:  # Don't match the current heading
                    earliest_end = min(earliest_end, match.start())
                    break

        return earliest_end

    def _classify_section_by_title(self, title: str) -> str:
        """Classify a section based on its title."""
        title_lower = title.lower()

        # Keywords for different section types
        classification_keywords = {
            "introduction": ["introduction", "overview", "background", "motivation"],
            "related_work": [
                "related",
                "literature",
                "survey",
                "background",
                "previous",
            ],
            "methods": [
                "method",
                "approach",
                "algorithm",
                "technique",
                "design",
                "implementation",
            ],
            "results": [
                "results",
                "evaluation",
                "experiment",
                "findings",
                "performance",
            ],
            "discussion": ["discussion", "analysis", "interpretation", "implications"],
            "conclusion": [
                "conclusion",
                "summary",
                "future",
                "limitations",
                "concluding",
            ],
            "references": ["references", "bibliography", "citation"],
            "abstract": ["abstract", "summary"],
        }

        for section_type, keywords in classification_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type

        return "unknown"

    def _merge_overlapping_boundaries(self, boundaries: list[tuple[int, int, str, float]]) -> list[tuple[int, int, str, float]]:
        """Remove overlapping boundaries and merge adjacent sections."""
        if not boundaries:
            return []

        merged = [boundaries[0]]

        for current in boundaries[1:]:
            last = merged[-1]

            # If current overlaps with last, merge them
            if current[0] < last[1]:
                # Extend the last boundary
                merged[-1] = (last[0], max(last[1], current[1]), last[2], last[3])
            else:
                # Add as new boundary
                merged.append(current)

        return merged
