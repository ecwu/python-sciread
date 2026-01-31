"""Advanced regex-based text splitter for academic documents."""

import argparse
import re
from pathlib import Path
from re import Pattern

from ..models import Chunk
from .base import BaseSplitter


class RegexSectionSplitter(BaseSplitter):
    """Advanced regex-based text splitter using pattern library."""

    def __init__(
        self,
        patterns: dict[str, str] | None = None,
        min_chunk_size: int = 200,
        confidence_threshold: float = 0.3,
    ):
        """Initialize regex splitter with configuration.

        Args:
            patterns: Custom patterns dictionary. If None, uses default academic patterns.
            min_chunk_size: Minimum chunk size in characters.
            confidence_threshold: Minimum confidence score for chunks.
        """
        self.min_chunk_size = min_chunk_size
        self.confidence_threshold = confidence_threshold

        # Load patterns (custom or default)
        self.patterns = patterns or self._get_default_patterns()
        self.compiled_patterns = self._compile_patterns()

    @property
    def splitter_name(self) -> str:
        """Return the splitter name."""
        return f"RegexSectionSplitter(patterns={len(self.patterns)}, min_size={self.min_chunk_size})"

    def split(self, text: str) -> list[Chunk]:
        """Split text using regex patterns."""
        text = self._validate_text(text)

        # Apply patterns to find split points
        split_points = self._find_split_points(text)

        # Create chunks based on split points
        raw_chunks = self._create_raw_chunks(text, split_points)

        # Create Chunk objects with metadata
        chunks = self._create_chunks(raw_chunks)

        # Reassign positions to ensure continuous ordering
        for i, chunk in enumerate(chunks):
            chunk.position = i

        return chunks

    def _get_default_patterns(self) -> dict[str, str]:
        """Get default academic paper patterns."""
        return {
            # Major section boundaries (high confidence) - enhanced from RuleBasedSplitter
            "abstract": r"^\s*(?:abstract|summary)\s*[:\.]?\s*$",
            "introduction": r"^\s*(?:introduction|introduction\s+and\s+background|overview)\s*[:\.]?\s*$",
            "related_work": r"^\s*(?:related\s+work|literature\s+review|background)\s*[:\.]?\s*$",
            "methodology": r"^\s*(?:method(?:s|ology)?|approach|materials\s+and\s+methods|experimental\s+setup|design)\s*[:\.]?\s*$",
            "results": r"^\s*(?:results|findings|outcomes|evaluation|experiment|experiments)\s*[:\.]?\s*$",
            "discussion": r"^\s*(?:discussion|analysis|interpretation)\s*[:\.]?\s*$",
            "conclusion": r"^\s*(?:conclusion|conclusions|future\s+work|limitations|summary)\s*[:\.]?\s*$",
            "references": r"^\s*(?:references|bibliography|works\s+cited)\s*[:\.]?\s*$",
            "appendix": r"^\s*(?:appendix|appendices|supplementary\s+material)\s*[:\.]?\s*$",
            # Numbered sections (medium confidence) - enhanced from RuleBasedSplitter
            "section_number": r"^\s*(\d+)\.\s+([A-Z][^\n]*?)\s*$",
            "subsection": r"^\s*(\d+\.\d+)\s+([^\n]*?)\s*$",
            # Medium confidence patterns
            "figure_table": r"(?i)^(?:Figure|Table)\s+\d+[:\.\s]",
            # Structural patterns (lower confidence)
            "paragraph_break": r"\n\s*\n",
            "page_break": r"\f",
            "citation_block": r"\n\s*\[\d+\][^\n]*\n",
            # Common academic phrases
            "contributions": r"(?i)^(?:\s*(?:Our\s+contributions|This\s+paper\s+contributes))",
            "limitations": r"(?i)^(?:\s*(?:Limitations|Limitation))",
        }

    def _compile_patterns(self) -> dict[str, Pattern]:
        """Compile regex patterns with confidence scores."""
        pattern_confidence = {
            # High confidence (0.8-0.95)
            "abstract": 0.95,
            "introduction": 0.9,
            "methodology": 0.9,
            "results": 0.85,
            "discussion": 0.85,
            "conclusion": 0.9,
            "references": 0.95,
            "appendix": 0.9,
            # Medium confidence (0.5-0.7)
            "related_work": 0.8,
            "section_number": 0.9,  # Enhanced confidence for numbered sections
            "subsection": 0.7,  # Enhanced confidence for subsections
            "figure_table": 0.6,
            # Lower confidence (0.2-0.4)
            "paragraph_break": 0.3,
            "page_break": 0.4,
            "citation_block": 0.4,
            "contributions": 0.5,
            "limitations": 0.5,
        }

        compiled = {}
        for name, pattern in self.patterns.items():
            try:
                # All patterns should be multiline for academic paper detection
                compiled[name] = {
                    "pattern": re.compile(pattern, re.IGNORECASE | re.MULTILINE),
                    "confidence": pattern_confidence.get(name, 0.5),
                }
            except re.error:
                # Skip invalid patterns
                continue

        return compiled

    def _find_split_points(self, text: str) -> list[tuple[int, str, float]]:
        """Find split points in text using patterns.

        Returns:
            List of (position, pattern_name, confidence) tuples.
        """
        split_points = []

        # Process patterns in order of confidence (high to low)
        pattern_items = sorted(
            self.compiled_patterns.items(),
            key=lambda x: x[1]["confidence"],
            reverse=True,
        )

        for pattern_name, pattern_info in pattern_items:
            pattern = pattern_info["pattern"]
            confidence = pattern_info["confidence"]

            for match in pattern.finditer(text):
                split_points.append((match.start(), pattern_name, confidence))

        # Sort by position
        split_points.sort(key=lambda x: x[0])

        # Remove duplicates and prioritize higher confidence patterns
        filtered_points = []
        for pos, name, conf in split_points:
            if not filtered_points or pos != filtered_points[-1][0]:
                filtered_points.append((pos, name, conf))
            else:
                # Keep the highest confidence for this position
                if conf > filtered_points[-1][2]:
                    filtered_points[-1] = (pos, name, conf)

        # Filter out low-confidence split points that are too close to high-confidence ones
        final_points = []
        for _i, (pos, name, conf) in enumerate(filtered_points):
            # Keep medium-confidence patterns (0.5-0.7) even if near high-confidence ones
            if conf >= 0.5:
                final_points.append((pos, name, conf))
                continue

            # For low-confidence points, check if there's a high-confidence point nearby
            if conf < 0.5:
                # Check if there's a high-confidence point nearby
                has_nearby_high_conf = any(
                    abs(pos - other_pos) < 20 and other_conf >= 0.7
                    for other_pos, other_name, other_conf in filtered_points
                    if other_pos != pos
                )
                if has_nearby_high_conf:
                    continue
            final_points.append((pos, name, conf))

        return final_points

    def _create_raw_chunks(self, text: str, split_points: list[tuple[int, str, float]]) -> list[dict]:
        """Create raw chunks based on split points."""
        if not split_points:
            return [
                {
                    "text": text,
                    "pattern": "no_split",
                    "confidence": 0.1,
                    "start_pos": 0,
                    "end_pos": len(text),
                }
            ]

        chunks = []
        prev_pos = 0

        for _i, (pos, _pattern_name, _confidence) in enumerate(split_points):
            if pos > prev_pos:
                chunk_text = text[prev_pos:pos].strip()
                if chunk_text:
                    # Find the best pattern for this chunk by checking what it contains
                    best_pattern, best_confidence = self._find_pattern_for_chunk(chunk_text)
                    chunks.append(
                        {
                            "text": chunk_text,
                            "pattern": best_pattern,
                            "confidence": best_confidence,
                            "start_pos": prev_pos,
                            "end_pos": pos,
                        }
                    )
            prev_pos = pos

        # Add final chunk
        if prev_pos < len(text):
            chunk_text = text[prev_pos:].strip()
            if chunk_text:
                # Find the best pattern for the final chunk
                best_pattern, best_confidence = self._find_pattern_for_chunk(chunk_text)
                chunks.append(
                    {
                        "text": chunk_text,
                        "pattern": best_pattern,
                        "confidence": best_confidence,
                        "start_pos": prev_pos,
                        "end_pos": len(text),
                    }
                )

        return chunks

    def _find_pattern_for_chunk(self, chunk_text: str) -> tuple[str, float]:
        """Find the best matching pattern for a chunk of text."""
        best_pattern = "unknown"
        best_confidence = 0.1

        # Check all patterns to see what matches this chunk
        for pattern_name, pattern_info in self.compiled_patterns.items():
            pattern = pattern_info["pattern"]
            confidence = pattern_info["confidence"]

            # Check if pattern matches within this chunk
            match = pattern.search(chunk_text)
            if match:
                # Enhanced handling for numbered sections
                if pattern_name in ["section_number", "subsection"] and match.groups():
                    section_title = match.group(2) if len(match.groups()) >= 2 else ""
                    if section_title:
                        # Classify the section by title for better accuracy
                        classified_type = self._classify_section_by_title(section_title.strip())
                        if classified_type != "unknown":
                            best_pattern = classified_type
                            best_confidence = confidence
                            continue

                # Use the pattern with highest confidence
                if confidence > best_confidence:
                    best_pattern = pattern_name
                    best_confidence = confidence

        return best_pattern, best_confidence

    def _create_chunks(self, raw_chunks: list[dict]) -> list[Chunk]:
        """Create Chunk objects from raw chunks."""
        chunks = []

        for i, raw_chunk in enumerate(raw_chunks):
            content = raw_chunk["text"]
            if len(content) < self.min_chunk_size:
                confidence = raw_chunk["confidence"] * 0.5  # Reduce confidence for small chunks
            else:
                confidence = raw_chunk["confidence"]

            # Determine chunk type based on the split point pattern (not the current chunk pattern)
            # For the first chunk, check if it starts with a high-confidence pattern
            if i == 0:
                # Check if content starts with a section header
                chunk_type = self._infer_chunk_type_from_content(content)
            else:
                # Use the pattern that caused the split before this chunk
                chunk_type = self._infer_chunk_type(raw_chunk["pattern"], content)

            # Extract actual section name from content
            section_name = self._extract_section_name_from_content(content)

            chunk = Chunk(
                content=content,
                chunk_name=section_name if section_name else "unknown",
                position=i,
                char_range=(raw_chunk["start_pos"], raw_chunk["end_pos"]),
                confidence=confidence,
                metadata={"splitter": chunk_type},
            )
            chunks.append(chunk)

        return chunks

    def _infer_chunk_type_from_content(self, content: str) -> str:
        """Infer chunk type from the content itself."""
        _content_lower = content.lower()
        content_stripped = content.strip()

        # Check for section headers at the start
        if content_stripped.startswith(("abstract", "abstract\n")):
            return "abstract"
        elif content_stripped.startswith(("introduction", "introduction\n")):
            return "introduction"
        elif content_stripped.startswith(("method", "methodology", "methods")):
            return "methods"
        elif content_stripped.startswith(("result", "experiment", "evaluation")):
            return "results"
        elif content_stripped.startswith(("discussion", "conclusion")):
            return "discussion"
        elif content_stripped.startswith(("references", "bibliography")):
            return "references"

        return self._infer_chunk_type("", content)

    def _infer_chunk_type(self, pattern_name: str, content: str) -> str:
        """Infer chunk type based on pattern and content."""
        # Direct mapping from pattern to chunk type
        pattern_mapping = {
            "abstract": "abstract",
            "introduction": "introduction",
            "related_work": "related_work",
            "methodology": "methods",
            "experiments": "results",
            "discussion": "discussion",
            "conclusion": "conclusion",
            "references": "references",
            "appendix": "appendix",
            "figure_table": "figure",
            "section_number": "section",
            "subsection": "subsection",
        }

        if pattern_name in pattern_mapping:
            return pattern_mapping[pattern_name]

        # Enhanced heuristic inference based on content (from RuleBasedSplitter)
        return self._classify_section_by_content(content)

    def _classify_section_by_content(self, content: str) -> str:
        """Classify a section based on its content using enhanced keywords (from RuleBasedSplitter)."""
        content_lower = content.lower()

        # Enhanced keywords for different section types
        classification_keywords = {
            "introduction": ["introduction", "overview", "background", "motivation"],
            "related_work": [
                "related",
                "literature",
                "survey",
                "background",
                "previous",
                "prior",
            ],
            "methods": [
                "method",
                "approach",
                "algorithm",
                "technique",
                "design",
                "implementation",
                "procedure",
                "experimental",
            ],
            "results": [
                "results",
                "evaluation",
                "experiment",
                "findings",
                "performance",
                "outcome",
                "data",
            ],
            "discussion": ["discussion", "analysis", "interpretation", "implications"],
            "conclusion": [
                "conclusion",
                "summary",
                "future",
                "limitations",
                "concluding",
                "final",
            ],
            "references": ["references", "bibliography", "citation", "cite"],
            "abstract": ["abstract", "summary"],
        }

        # Check each category with keyword weighting
        for section_type, keywords in classification_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_count >= 2:  # Require at least 2 keywords for classification
                return section_type

        # Single keyword fallback
        for section_type, keywords in classification_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return section_type

        return "unknown"

    def _classify_section_by_title(self, title: str) -> str:
        """Classify a section based on its title (from RuleBasedSplitter)."""
        return self._classify_section_by_content(title)

    def _extract_section_name_from_content(self, content: str) -> str | None:
        """Extract the actual section name from content for display/searching."""
        lines = content.strip().split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Check for numbered sections (e.g., "1. Introduction", "2. Methods")
        numbered_match = re.match(r"^\d+(?:\.\d+)*\s+(.+)$", first_line)
        if numbered_match:
            return numbered_match.group(1).strip()

        # Check for common section headers
        section_patterns = [
            r"^abstract\s*(?:[:\-])?\s*(.+)$",
            r"^introduction\s*(?:[:\-])?\s*(.+)$",
            r"^(?:method|methodology|methods?)\s*(?:[:\-])?\s*(.+)$",
            r"^results?\s*(?:[:\-])?\s*(.+)$",
            r"^discussion\s*(?:[:\-])?\s*(.+)$",
            r"^conclusion\s*(?:[:\-])?\s*(.+)$",
            r"^references?\s*(?:[:\-])?\s*(.+)$",
        ]

        for pattern in section_patterns:
            match = re.match(pattern, first_line, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.group(1).strip() else match.group(0).strip()

        # If it's a single word section header
        single_word_patterns = [
            r"^abstract$",
            r"^introduction$",
            r"^methodology$",
            r"^methods$",
            r"^results$",
            r"^discussion$",
            r"^conclusion$",
            r"^references$",
            r"^acknowledgments?$",
            r"^appendix$",
            r"^background$",
        ]

        for pattern in single_word_patterns:
            if re.match(pattern, first_line, re.IGNORECASE):
                return first_line

        # Return first line as section name if it's reasonably short
        if len(first_line) < 100 and len(first_line.split()) <= 10:
            return first_line

        return None

    def add_custom_pattern(self, name: str, pattern: str, confidence: float = 0.5):
        """Add a custom pattern."""
        self.patterns[name] = pattern
        try:
            compiled_pattern = re.compile(pattern, re.MULTILINE)
            self.compiled_patterns[name] = {
                "pattern": compiled_pattern,
                "confidence": confidence,
            }
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name."""
        if name in self.patterns:
            del self.patterns[name]
            if name in self.compiled_patterns:
                del self.compiled_patterns[name]
            return True
        return False


def main():
    """Main function to demonstrate RegexSectionSplitter on a txt file."""
    parser = argparse.ArgumentParser(description="Split a text file using RegexSectionSplitter and display chunks with metadata")
    parser.add_argument("file_path", type=str, help="Path to the text file to split")
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=200,
        help="Minimum chunk size in characters (default: 200)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for chunks (default: 0.3)",
    )

    args = parser.parse_args()

    # Check if file exists
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{args.file_path}' not found.")
        return 1

    # Initialize the splitter
    splitter = RegexSectionSplitter(
        min_chunk_size=args.min_chunk_size,
        confidence_threshold=args.confidence_threshold,
    )

    try:
        # Read the text file with encoding detection
        print(f"Reading file: {args.file_path}")
        text = None

        # Try different encodings in order of preference
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings_to_try:
            try:
                with Path(file_path).open(encoding=encoding) as f:
                    text = f.read()
                print(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            print(f"Error: Could not read file with any of the attempted encodings: {', '.join(encodings_to_try)}")
            return 1

        if not text.strip():
            print("Error: File is empty.")
            return 1

        print(f"File loaded: {len(text)} characters")
        print(f"Splitter configuration: {splitter.splitter_name}")
        print(f"Patterns loaded: {len(splitter.patterns)}")
        print("-" * 80)

        # Split the text
        print("Splitting text using RegexSectionSplitter...")
        chunks = splitter.split(text)

        if not chunks:
            print("No chunks were generated.")
            return 1

        print(f"Generated {len(chunks)} chunks")
        print("-" * 80)

        # Display chunks with metadata
        for i, chunk in enumerate(chunks, 1):
            word_count = len(chunk.content.split())
            confidence_str = f"{chunk.confidence:.2f}" if chunk.confidence is not None else "N/A"

            # Get pattern/matching info from chunk metadata
            split_reason = chunk.chunk_name
            if hasattr(chunk, "metadata") and chunk.metadata:
                split_reason = chunk.metadata.get("pattern", chunk.chunk_name)

            header = (
                f"============= Chunk #{i} ({word_count} words) ============= "
                f"Conf: {confidence_str} ============= Type: {split_reason} ============="
            )
            print(header)
            print(chunk.content)
            print("-" * 80)

        # Print summary
        total_words = sum(len(chunk.content.split()) for chunk in chunks)
        avg_confidence = (
            sum(c.confidence for c in chunks if c.confidence is not None) / len([c for c in chunks if c.confidence is not None])
            if any(c.confidence for c in chunks)
            else 0
        )

        # Chunk type distribution
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_name
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        print("\nSummary:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total words: {total_words}")
        print(f"  Average confidence: {avg_confidence:.2f}")
        print(f"  Chunk types: {dict(type_counts)}")

        # Show patterns used
        print("\nPatterns used in splitting:")
        for pattern_name, pattern_info in splitter.compiled_patterns.items():
            print(f"  {pattern_name}: confidence={pattern_info['confidence']:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
