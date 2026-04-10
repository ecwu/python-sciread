"""Markdown-specific text splitter that leverages markdown structure for accurate chunking."""

import re
import uuid

from sciread.document.models import Chunk
from sciread.document.structure.paths import clean_section_name
from sciread.document.structure.paths import get_parent_section_id

from .base import BaseSplitter


class MarkdownSplitter(BaseSplitter):
    """Markdown-aware splitter that uses markdown structure for intelligent chunking."""

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 0,
        preserve_code_blocks: bool = True,
        split_on_headers: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """Initialize markdown splitter with configuration.

        Args:
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            chunk_overlap: Number of backward-overlap characters to include in each non-initial chunk.
            preserve_code_blocks: Whether to keep code blocks intact.
            split_on_headers: Whether to split on markdown headers.
            confidence_threshold: Minimum confidence score for chunks.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = self._validate_chunk_overlap(chunk_overlap)
        self.preserve_code_blocks = preserve_code_blocks
        self.split_on_headers = split_on_headers
        self.confidence_threshold = confidence_threshold

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        overlap_suffix = f", overlap={self.chunk_overlap}" if self.chunk_overlap else ""
        return f"MarkdownSplitter(min={self.min_chunk_size}, max={self.max_chunk_size}{overlap_suffix})"

    def _compile_patterns(self):
        """Compile markdown-specific regex patterns."""
        self.patterns = {
            # Headers (highest confidence)
            "h1": re.compile(r"^(#{1})\s+(.+)$", re.MULTILINE),
            "h2": re.compile(r"^(#{2})\s+(.+)$", re.MULTILINE),
            "h3": re.compile(r"^(#{3})\s+(.+)$", re.MULTILINE),
            "h4": re.compile(r"^(#{4})\s+(.+)$", re.MULTILINE),
            "h5": re.compile(r"^(#{5})\s+(.+)$", re.MULTILINE),
            "h6": re.compile(r"^(#{6})\s+(.+)$", re.MULTILINE),
            # Code blocks (high confidence)
            "fenced_code": re.compile(r"^```[\w]*\n.*?\n```", re.MULTILINE | re.DOTALL),
            "indented_code": re.compile(r"^(?:\t| {4}).+(?:\n(?:\t| {4}).+)*", re.MULTILINE),
            "inline_code": re.compile(r"`[^`]+`"),
            # Lists (medium confidence)
            "unordered_list": re.compile(r"^[*+-]\s+.+$", re.MULTILINE),
            "ordered_list": re.compile(r"^\d+\.\s+.+$", re.MULTILINE),
            # Structural elements (medium confidence)
            "blockquote": re.compile(r"^>\s+.+$", re.MULTILINE),
            "horizontal_rule": re.compile(r"^-{3,}$|^\*{3,}$|^_{3,}$", re.MULTILINE),
            # Tables (medium confidence)
            "table": re.compile(r"^\|.*\|$", re.MULTILINE),
            # Links and emphasis (low confidence)
            "link": re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
            "bold": re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__"),
            "italic": re.compile(r"\*([^*]+)\*|_([^_]+)_"),
        }

        # Confidence scores for different markdown elements
        self.confidence_scores = {
            "h1": 0.95,
            "h2": 0.90,
            "h3": 0.85,
            "h4": 0.80,
            "h5": 0.75,
            "h6": 0.70,
            "fenced_code": 0.90,
            "indented_code": 0.80,
            "unordered_list": 0.60,
            "ordered_list": 0.60,
            "blockquote": 0.65,
            "horizontal_rule": 0.50,
            "table": 0.65,
            "paragraph": 0.40,
        }

    def split(self, text: str) -> list[Chunk]:
        """Split markdown text into chunks based on markdown structure."""
        text = self._validate_text(text)

        # Extract and preserve code blocks if requested
        code_blocks = []
        if self.preserve_code_blocks:
            text, code_blocks = self._extract_code_blocks(text, self.patterns)

        # Find all split points based on markdown structure
        split_points = self._find_markdown_split_points(text)

        # Create chunks based on split points
        chunks = self._create_markdown_chunks(text, split_points)
        chunks = self._apply_chunk_overlap(text, chunks)

        # Restore code blocks if they were extracted
        if self.preserve_code_blocks and code_blocks:
            chunks = self._restore_code_blocks(chunks, code_blocks)

        # Ensure continuity by reassigning positions
        for i, chunk in enumerate(chunks):
            chunk.position = i

        return chunks

    def _clean_section_name(self, title: str) -> str:
        """Clean section name: lowercase, remove symbols, trim spaces.

        Args:
            title: Raw section title from markdown header.

        Returns:
            Cleaned section name suitable for identification.
        """
        return clean_section_name(title)

    def _find_markdown_split_points(self, text: str) -> list[tuple[int, str, float, str, int]]:
        """Find split points based on markdown structure.

        Returns:
            List of (position, element_type, confidence, section_name, level) tuples.
        """
        split_points: list[tuple[int, str, float, str, int]] = []

        if self.split_on_headers:
            # Find all headers as primary split points
            for level in range(1, 7):
                pattern = self.patterns[f"h{level}"]
                confidence = self.confidence_scores[f"h{level}"]

                for match in pattern.finditer(text):
                    # Extract the header title and clean it for section name
                    raw_title = match.group(2).strip()
                    section_name = self._clean_section_name(raw_title)
                    split_points.append((match.start(), f"h{level}", confidence, section_name, level))

        # Sort split points by position
        split_points.sort(key=lambda x: x[0])

        # Remove duplicates and keep highest confidence for same position
        filtered_points = []
        for pos, name, conf, section_name, level in split_points:
            if not filtered_points or pos != filtered_points[-1][0]:
                filtered_points.append((pos, name, conf, section_name, level))
            elif conf > filtered_points[-1][2]:
                filtered_points[-1] = (pos, name, conf, section_name, level)

        return filtered_points

    def _create_markdown_chunks(self, text: str, split_points: list[tuple[int, str, float, str, int]]) -> list[Chunk]:
        """Create chunks based on markdown split points."""
        if not split_points:
            # No markdown structure found, treat as single chunk
            return [self._create_chunk(text, 0, len(text), "no_structure", 0.3)]

        chunks = []
        prev_pos = 0
        active_path: list[str] = []

        for i, (pos, element_type, confidence, section_name, level) in enumerate(split_points):
            if pos > prev_pos:
                chunk_text = text[prev_pos:pos].strip()
                if chunk_text:
                    chunk = self._create_chunk_from_content(
                        chunk_text,
                        prev_pos,
                        pos,
                        element_type,
                        confidence,
                        active_path or (["preamble"] if i == 0 else []),
                    )
                    chunks.append(chunk)

            active_path = active_path[: max(level - 1, 0)]
            active_path.append(section_name)
            prev_pos = pos

        # Add final chunk
        if prev_pos < len(text):
            chunk_text = text[prev_pos:].strip()
            if chunk_text:
                chunk = self._create_chunk_from_content(
                    chunk_text,
                    prev_pos,
                    len(text),
                    "final",
                    0.5,
                    active_path.copy(),
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk_from_content(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        split_reason: str,
        default_confidence: float,
        section_name: str | list[str] | None = None,
    ) -> Chunk:
        """Create a chunk and determine its type and confidence based on content."""
        # Analyze content to determine the most appropriate chunk type
        chunk_type, confidence = self._analyze_chunk_content(content, default_confidence)

        # If content starts with a header, extract section info
        if section_name is None:
            section_name = self._extract_section_from_content(content)

        if isinstance(section_name, list):
            section_path = [part for part in section_name if part]
            section_value = section_path[-1] if section_path else "unknown"
        else:
            section_value = section_name if section_name else "unknown"
            section_path = [section_value] if section_value != "unknown" else []
        chunk_id = str(uuid.uuid4())

        chunk = Chunk(
            content=content,
            chunk_id=chunk_id,
            doc_id="",
            content_plain=content,
            section_path=section_path,
            page_start=None,
            page_end=None,
            para_index=0,
            chunk_name=section_value,
            position=0,  # Will be assigned later
            char_range=(start_pos, end_pos),
            token_count=len(content.split()),
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_section_id=get_parent_section_id(section_path),
            citation_key=chunk_id,
            retrievable=True,
            confidence=confidence,
            metadata={"splitter": chunk_type},
        )

        return chunk

    def _extract_section_from_content(self, content: str) -> str | None:
        """Extract section name from content if it starts with a header."""
        first_line = content.split("\n", maxsplit=1)[0].strip()

        # Check if first line is a markdown header
        header_match = re.match(r"^(#{1,6})\s+(.+)$", first_line)
        if header_match:
            raw_title = header_match.group(2).strip()
            return self._clean_section_name(raw_title)

        return None

    def _analyze_chunk_content(self, content: str, default_confidence: float) -> tuple[str, float]:
        """Analyze chunk content to determine type and confidence."""
        content_lower = content.lower()

        # Check for code content first (highest priority)
        if self.patterns["fenced_code"].search(content) or self.patterns["indented_code"].search(content):
            return "code", self.confidence_scores["fenced_code"]

        # Check for tables
        if self.patterns["table"].search(content):
            return "table", self.confidence_scores["table"]

        # Check for lists
        if self.patterns["unordered_list"].search(content) or self.patterns["ordered_list"].search(content):
            return "list", self.confidence_scores["unordered_list"]

        # Check for blockquotes
        if self.patterns["blockquote"].search(content):
            return "blockquote", self.confidence_scores["blockquote"]

        # Check for academic paper sections by content analysis
        academic_types = [
            ("abstract", ["abstract", "summary"]),
            ("introduction", ["introduction", "overview", "background"]),
            ("methods", ["method", "approach", "methodology", "experimental"]),
            ("results", ["results", "findings", "evaluation", "outcome"]),
            ("discussion", ["discussion", "analysis", "interpretation"]),
            ("conclusion", ["conclusion", "summary", "future work"]),
            ("references", ["references", "bibliography", "citation"]),
        ]

        for section_type, keywords in academic_types:
            if any(keyword in content_lower for keyword in keywords):
                return section_type, 0.8

        # Check for headers only if content starts with a header
        first_line = content.split("\n", maxsplit=1)[0].strip()
        for element_type, pattern in self.patterns.items():
            if element_type.startswith("h") and pattern.match(first_line):
                confidence = self.confidence_scores.get(element_type, default_confidence)
                return element_type, confidence

        # Default classification
        return "content", default_confidence

    def _create_chunk(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        chunk_type: str,
        confidence: float,
    ) -> Chunk:
        """Create a basic chunk."""
        chunk_id = str(uuid.uuid4())
        return Chunk(
            content=content,
            chunk_id=chunk_id,
            doc_id="",
            content_plain=content,
            section_path=[],
            page_start=None,
            page_end=None,
            para_index=0,
            chunk_name="unknown",
            position=0,  # Will be assigned later
            char_range=(start_pos, end_pos),
            token_count=len(content.split()),
            prev_chunk_id=None,
            next_chunk_id=None,
            parent_section_id=None,
            citation_key=chunk_id,
            retrievable=True,
            confidence=confidence,
            metadata={"splitter": chunk_type},
        )

    def _restore_code_blocks(self, chunks: list[Chunk], code_blocks: list[dict]) -> list[Chunk]:
        """Restore extracted code blocks to their original positions."""
        return super()._restore_code_blocks(chunks, code_blocks)

    def add_pattern(self, name: str, pattern: str, confidence: float) -> None:
        """Add a custom markdown pattern."""
        try:
            compiled_pattern = re.compile(pattern, re.MULTILINE)
            self.patterns[name] = compiled_pattern
            self.confidence_scores[name] = confidence
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    def remove_pattern(self, name: str) -> bool:
        """Remove a custom pattern."""
        if name in self.patterns:
            del self.patterns[name]
            if name in self.confidence_scores:
                del self.confidence_scores[name]
            return True
        return False
