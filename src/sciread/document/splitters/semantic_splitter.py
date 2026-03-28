"""Semantic splitter combining academic paper and markdown structure for intelligent chunking."""

import re

from ..models import Chunk
from .base import BaseSplitter


class SemanticSplitter(BaseSplitter):
    """Unified semantic splitter that combines academic paper and markdown structure for intelligent chunking."""

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        preserve_code_blocks: bool = True,
        split_on_headers: bool = True,
        confidence_threshold: float = 0.7,
        enable_academic_patterns: bool = True,
        enable_markdown_patterns: bool = True,
    ):
        """
        Initialize semantic splitter with configuration.

        Args:
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
            preserve_code_blocks: Whether to keep code blocks intact.
            split_on_headers: Whether to split on headers.
            confidence_threshold: Minimum confidence score for chunks.
            enable_academic_patterns: Whether to enable academic paper pattern detection.
            enable_markdown_patterns: Whether to enable markdown pattern detection.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_code_blocks = preserve_code_blocks
        self.split_on_headers = split_on_headers
        self.confidence_threshold = confidence_threshold
        self.enable_academic_patterns = enable_academic_patterns
        self.enable_markdown_patterns = enable_markdown_patterns

        # Pre-compile regex patterns for better performance
        self._compile_patterns()

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"SemanticSplitter(min={self.min_chunk_size}, max={self.max_chunk_size})"

    def _compile_patterns(self):
        """Compile unified regex patterns for both academic and markdown content."""
        self.patterns = {}

        # Academic paper patterns
        if self.enable_academic_patterns:
            academic_patterns = {
                # High confidence academic sections
                "abstract": re.compile(
                    r"^(?:abstract|summary)\s*[:\-]?\s*$", re.IGNORECASE | re.MULTILINE
                ),
                "introduction": re.compile(
                    r"^(?:introduction|overview|background)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "related_work": re.compile(
                    r"^(?:related\s+work|literature\s+review|background\s+work)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "methodology": re.compile(
                    r"^(?:method(?:ology)?|approach|experimental\s+(?:method|setup|design))\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "methods": re.compile(
                    r"^(?:methods|materials\s+and\s+methods)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "results": re.compile(
                    r"^(?:results|findings|evaluation|outcome)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "discussion": re.compile(
                    r"^(?:discussion|analysis|interpretation)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "conclusion": re.compile(
                    r"^(?:conclusion|summary|future\s+work)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "references": re.compile(
                    r"^(?:references|bibliography|citations?)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                "acknowledgments": re.compile(
                    r"^(?:acknowledgments?|acknowledge?ments?)\s*[:\-]?\s*$",
                    re.IGNORECASE | re.MULTILINE,
                ),
                # Medium confidence academic patterns
                "section": re.compile(r"^(\d+)\.?\s+([^\n]+)$", re.MULTILINE),
                "subsection": re.compile(r"^(\d+\.\d+)\.?\s+([^\n]+)$", re.MULTILINE),
                "subsection2": re.compile(
                    r"^(\d+\.\d+\.\d+)\.?\s+([^\n]+)$", re.MULTILINE
                ),
            }
            self.patterns.update(academic_patterns)

        # Markdown patterns
        if self.enable_markdown_patterns:
            markdown_patterns = {
                # Headers (highest confidence)
                "h1": re.compile(r"^(#{1})\s+(.+)$", re.MULTILINE),
                "h2": re.compile(r"^(#{2})\s+(.+)$", re.MULTILINE),
                "h3": re.compile(r"^(#{3})\s+(.+)$", re.MULTILINE),
                "h4": re.compile(r"^(#{4})\s+(.+)$", re.MULTILINE),
                "h5": re.compile(r"^(#{5})\s+(.+)$", re.MULTILINE),
                "h6": re.compile(r"^(#{6})\s+(.+)$", re.MULTILINE),
                # Code blocks (high confidence)
                "fenced_code": re.compile(
                    r"^```[\w]*\n.*?\n```", re.MULTILINE | re.DOTALL
                ),
                "indented_code": re.compile(
                    r"^(?:\t| {4}).+(?:\n(?:\t| {4}).+)*", re.MULTILINE
                ),
                "inline_code": re.compile(r"`[^`]+`"),
                # Lists (medium confidence)
                "unordered_list": re.compile(r"^[*+-]\s+.+$", re.MULTILINE),
                "ordered_list": re.compile(r"^\d+\.\s+.+$", re.MULTILINE),
                # Structural elements (medium confidence)
                "blockquote": re.compile(r"^>\s+.+$", re.MULTILINE),
                "horizontal_rule": re.compile(
                    r"^-{3,}$|^\*{3,}$|^_{3,}$", re.MULTILINE
                ),
                # Tables (medium confidence)
                "table": re.compile(r"^\|.*\|$", re.MULTILINE),
                # Links and emphasis (low confidence)
                "link": re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
                "bold": re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__"),
                "italic": re.compile(r"\*([^*]+)\*|_([^_]+)_"),
            }
            self.patterns.update(markdown_patterns)

        # Confidence scores for different elements
        self.confidence_scores = {
            # Academic sections (highest confidence)
            "abstract": 0.95,
            "introduction": 0.90,
            "related_work": 0.90,
            "methodology": 0.90,
            "methods": 0.90,
            "results": 0.90,
            "discussion": 0.90,
            "conclusion": 0.90,
            "references": 0.95,
            "acknowledgments": 0.85,
            # Markdown headers (high confidence)
            "h1": 0.95,
            "h2": 0.90,
            "h3": 0.85,
            "h4": 0.80,
            "h5": 0.75,
            "h6": 0.70,
            # Code blocks (high confidence)
            "fenced_code": 0.90,
            "indented_code": 0.80,
            # Academic numbering (medium confidence)
            "section": 0.85,
            "subsection": 0.80,
            "subsection2": 0.75,
            # Other markdown elements (lower confidence)
            "unordered_list": 0.60,
            "ordered_list": 0.60,
            "blockquote": 0.65,
            "horizontal_rule": 0.50,
            "table": 0.65,
            "paragraph": 0.40,
        }

    def split(self, text: str) -> list[Chunk]:
        """Split text using semantic patterns."""
        text = self._validate_text(text)

        # Extract and preserve code blocks if requested
        code_blocks = []
        if self.preserve_code_blocks and self.enable_markdown_patterns:
            text, code_blocks = self._extract_code_blocks(text, self.patterns)

        # Find all split points based on semantic structure
        split_points = self._find_semantic_split_points(text)

        # Create chunks based on split points
        chunks = self._create_semantic_chunks(text, split_points)

        # Restore code blocks if they were extracted
        if self.preserve_code_blocks and self.enable_markdown_patterns and code_blocks:
            chunks = self._restore_code_blocks(chunks, code_blocks)

        # Ensure continuity by reassigning positions
        for i, chunk in enumerate(chunks):
            chunk.position = i

        return chunks

    def _clean_section_name(self, title: str) -> str:
        """Clean section name: lowercase, remove symbols, trim spaces."""
        # Convert to lowercase
        cleaned = title.lower()

        # Remove markdown symbols, brackets, and special characters
        # Keep letters, numbers, spaces, and hyphens only
        cleaned = re.sub(r"[^\w\s-]", "", cleaned)

        # Replace multiple spaces with single space and trim
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Return "untitled" if empty after cleaning
        return cleaned if cleaned else "untitled"

    def _find_semantic_split_points(
        self, text: str
    ) -> list[tuple[int, str, float, str]]:
        """Find split points based on semantic structure."""
        split_points = []

        if self.split_on_headers:
            # Find academic paper sections first (highest priority)
            if self.enable_academic_patterns:
                for pattern_name, pattern in self.patterns.items():
                    if pattern_name in [
                        "abstract",
                        "introduction",
                        "related_work",
                        "methodology",
                        "methods",
                        "results",
                        "discussion",
                        "conclusion",
                        "references",
                        "acknowledgments",
                    ]:
                        confidence = self.confidence_scores.get(pattern_name, 0.8)
                        for match in pattern.finditer(text):
                            section_name = pattern_name
                            split_points.append(
                                (match.start(), pattern_name, confidence, section_name)
                            )

            # Find markdown headers
            if self.enable_markdown_patterns:
                for level in range(1, 7):
                    pattern_name = f"h{level}"
                    if pattern_name in self.patterns:
                        pattern = self.patterns[pattern_name]
                        confidence = self.confidence_scores.get(pattern_name, 0.7)
                        for match in pattern.finditer(text):
                            # Extract the header title and clean it for section name
                            raw_title = match.group(2).strip()
                            section_name = self._clean_section_name(raw_title)
                            split_points.append(
                                (match.start(), pattern_name, confidence, section_name)
                            )

            # Find numbered academic sections
            if self.enable_academic_patterns:
                for pattern_name in ["section", "subsection", "subsection2"]:
                    if pattern_name in self.patterns:
                        pattern = self.patterns[pattern_name]
                        confidence = self.confidence_scores.get(pattern_name, 0.7)
                        for match in pattern.finditer(text):
                            # Extract section number and title
                            section_title = match.group(2).strip()
                            section_name = self._clean_section_name(section_title)
                            split_points.append(
                                (match.start(), pattern_name, confidence, section_name)
                            )

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

    def _create_semantic_chunks(
        self, text: str, split_points: list[tuple[int, str, float, str]]
    ) -> list[Chunk]:
        """Create chunks based on semantic split points."""
        if not split_points:
            # No semantic structure found, treat as single chunk
            return [self._create_chunk(text, 0, len(text), "no_structure", 0.3)]

        chunks = []
        prev_pos = 0

        for _i, (pos, element_type, confidence, _section_name) in enumerate(
            split_points
        ):
            if pos > prev_pos:
                chunk_text = text[prev_pos:pos].strip()
                if chunk_text:
                    # For content before first header, use "preamble" as section name
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
                chunk = self._create_chunk_from_content(
                    chunk_text, prev_pos, len(text), "final", 0.5, last_section_name
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
        section_name: str | None = None,
    ) -> Chunk:
        """Create a chunk and determine its type and confidence based on content."""
        # Analyze content to determine the most appropriate chunk type
        chunk_type, confidence = self._analyze_chunk_content(
            content, default_confidence
        )

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

        # Check for academic section headers
        if self.enable_academic_patterns:
            for pattern_name in [
                "abstract",
                "introduction",
                "related_work",
                "methodology",
                "methods",
                "results",
                "discussion",
                "conclusion",
                "references",
                "acknowledgments",
            ]:
                if pattern_name in self.patterns:
                    pattern = self.patterns[pattern_name]
                    if pattern.match(first_line):
                        return pattern_name

        # Check for markdown headers
        if self.enable_markdown_patterns:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", first_line)
            if header_match:
                raw_title = header_match.group(2).strip()
                return self._clean_section_name(raw_title)

        return None

    def _analyze_chunk_content(
        self, content: str, default_confidence: float
    ) -> tuple[str, float]:
        """Analyze chunk content to determine type and confidence."""
        content_lower = content.lower()

        # Check for code content first (highest priority)
        if self.enable_markdown_patterns:
            if "fenced_code" in self.patterns and self.patterns["fenced_code"].search(
                content
            ):
                return "code", self.confidence_scores["fenced_code"]
            if "indented_code" in self.patterns and self.patterns[
                "indented_code"
            ].search(content):
                return "code", self.confidence_scores["indented_code"]
            if "table" in self.patterns and self.patterns["table"].search(content):
                return "table", self.confidence_scores["table"]
            if "unordered_list" in self.patterns and self.patterns[
                "unordered_list"
            ].search(content):
                return "list", self.confidence_scores["unordered_list"]
            if "ordered_list" in self.patterns and self.patterns["ordered_list"].search(
                content
            ):
                return "list", self.confidence_scores["ordered_list"]
            if "blockquote" in self.patterns and self.patterns["blockquote"].search(
                content
            ):
                return "blockquote", self.confidence_scores["blockquote"]

        # Check for academic paper sections by content analysis
        if self.enable_academic_patterns:
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
                confidence = self.confidence_scores.get(
                    element_type, default_confidence
                )
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

    def _restore_code_blocks(
        self, chunks: list[Chunk], code_blocks: list[dict]
    ) -> list[Chunk]:
        """Restore extracted code blocks to their original positions."""
        return super()._restore_code_blocks(chunks, code_blocks)

    def add_custom_pattern(self, name: str, pattern: str, confidence: float = 0.5):
        """Add a custom pattern."""
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
