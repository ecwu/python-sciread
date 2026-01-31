"""Markdown-specific text splitter that leverages markdown structure for accurate chunking."""

import re

from ..models import Chunk
from .base import BaseSplitter


class MarkdownSplitter(BaseSplitter):
    """Markdown-aware splitter that uses markdown structure for intelligent chunking."""

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        preserve_code_blocks: bool = True,
        split_on_headers: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """Initialize markdown splitter with configuration.

        Args:
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            preserve_code_blocks: Whether to keep code blocks intact.
            split_on_headers: Whether to split on markdown headers.
            confidence_threshold: Minimum confidence score for chunks.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_code_blocks = preserve_code_blocks
        self.split_on_headers = split_on_headers
        self.confidence_threshold = confidence_threshold

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"MarkdownSplitter(min={self.min_chunk_size}, max={self.max_chunk_size})"

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
            text, code_blocks = self._extract_code_blocks(text)

        # Find all split points based on markdown structure
        split_points = self._find_markdown_split_points(text)

        # Create chunks based on split points
        chunks = self._create_markdown_chunks(text, split_points)

        # Restore code blocks if they were extracted
        if self.preserve_code_blocks and code_blocks:
            chunks = self._restore_code_blocks(chunks, code_blocks)

        # Ensure continuity by reassigning positions
        for i, chunk in enumerate(chunks):
            chunk.position = i

        return chunks

    def _extract_code_blocks(self, text: str) -> tuple[str, list[dict]]:
        """Extract code blocks and replace them with placeholders."""
        code_blocks = []
        placeholder_pattern = "__CODE_BLOCK_{}__"

        # Extract fenced code blocks
        for i, match in enumerate(self.patterns["fenced_code"].finditer(text)):
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
        offset = len(code_blocks)
        for i, match in enumerate(self.patterns["indented_code"].finditer(text)):
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

    def _clean_section_name(self, title: str) -> str:
        """Clean section name: lowercase, remove symbols, trim spaces.

        Args:
            title: Raw section title from markdown header.

        Returns:
            Cleaned section name suitable for identification.
        """
        # Convert to lowercase
        cleaned = title.lower()

        # Remove markdown symbols, brackets, and special characters
        # Keep letters, numbers, spaces, and hyphens only
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)

        # Replace multiple spaces with single space and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Return "untitled" if empty after cleaning
        return cleaned if cleaned else "untitled"

    def _find_markdown_split_points(self, text: str) -> list[tuple[int, str, float, str]]:
        """Find split points based on markdown structure.

        Returns:
            List of (position, element_type, confidence, section_name) tuples.
        """
        split_points = []

        if self.split_on_headers:
            # Find all headers as primary split points
            for level in range(1, 7):
                pattern = self.patterns[f"h{level}"]
                confidence = self.confidence_scores[f"h{level}"]

                for match in pattern.finditer(text):
                    # Extract the header title and clean it for section name
                    raw_title = match.group(2).strip()
                    section_name = self._clean_section_name(raw_title)
                    split_points.append((match.start(), f"h{level}", confidence, section_name))

        # Sort split points by position
        split_points.sort(key=lambda x: x[0])

        # Remove duplicates and keep highest confidence for same position
        filtered_points = []
        for pos, name, conf, section_name in split_points:
            if not filtered_points or pos != filtered_points[-1][0]:
                filtered_points.append((pos, name, conf, section_name))
            elif conf > filtered_points[-1][2]:
                filtered_points[-1] = (pos, name, conf, section_name)

        return filtered_points

    def _create_markdown_chunks(self, text: str, split_points: list[tuple[int, str, float, str]]) -> list[Chunk]:
        """Create chunks based on markdown split points."""
        if not split_points:
            # No markdown structure found, treat as single chunk
            return [self._create_chunk(text, 0, len(text), "no_structure", 0.3)]

        chunks = []
        prev_pos = 0

        for _i, (pos, element_type, confidence, _section_name) in enumerate(split_points):
            if pos > prev_pos:
                chunk_text = text[prev_pos:pos].strip()
                if chunk_text:
                    # For content before first header, use "preamble" as section name
                    # For content between headers, don't assign the next header's section name
                    content_section_name = None
                    if _i == 0:
                        content_section_name = "preamble"
                    chunk = self._create_chunk_from_content(
                        chunk_text,
                        prev_pos,
                        pos,
                        element_type,
                        confidence,
                        content_section_name,
                    )
                    chunks.append(chunk)
            prev_pos = pos

        # Add final chunk
        if prev_pos < len(text):
            chunk_text = text[prev_pos:].strip()
            if chunk_text:
                # Use section_name from the last split point if available
                last_section_name = split_points[-1][3] if split_points else None
                chunk = self._create_chunk_from_content(chunk_text, prev_pos, len(text), "final", 0.5, last_section_name)
                chunks.append(chunk)

        return chunks

    def _create_chunk_from_content(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        split_reason: str,
        default_confidence: float,
        section_name: str | None = None,
    ) -> Chunk:
        """Create a chunk and determine its type and confidence based on content."""
        # Analyze content to determine the most appropriate chunk type
        chunk_type, confidence = self._analyze_chunk_content(content, default_confidence)

        # If content starts with a header, extract section info
        if section_name is None:
            section_name = self._extract_section_from_content(content)

        chunk = Chunk(
            content=content,
            chunk_name=section_name if section_name else "unknown",
            position=0,  # Will be assigned later
            char_range=(start_pos, end_pos),
            confidence=confidence,
            metadata={"splitter": chunk_type},
        )

        return chunk

    def _extract_section_from_content(self, content: str) -> str | None:
        """Extract section name from content if it starts with a header."""
        first_line = content.split("\n")[0].strip()

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
        first_line = content.split("\n")[0].strip()
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
        return Chunk(
            content=content,
            chunk_name="unknown",
            position=0,  # Will be assigned later
            char_range=(start_pos, end_pos),
            confidence=confidence,
            metadata={"splitter": chunk_type},
        )

    def _restore_code_blocks(self, chunks: list[Chunk], code_blocks: list[dict]) -> list[Chunk]:
        """Restore extracted code blocks to their original positions."""
        for chunk in chunks:
            for code_block in code_blocks:
                if code_block["placeholder"] in chunk.content:
                    chunk.content = chunk.content.replace(code_block["placeholder"], code_block["content"])
        return chunks

    def add_custom_pattern(self, name: str, pattern: str, confidence: float = 0.5):
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
