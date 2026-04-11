"""Tests for the semantic splitter."""

import re

from sciread.document.structure.splitters.semantic_splitter import SemanticSplitter


def test_splitter_name_reports_overlap_when_enabled() -> None:
    """Splitter name should surface non-default overlap configuration."""
    splitter = SemanticSplitter(min_chunk_size=10, max_chunk_size=50, chunk_overlap=12)

    assert splitter.splitter_name == "SemanticSplitter(min=10, max=50, overlap=12)"


def test_find_semantic_split_points_keeps_highest_confidence_for_same_position() -> None:
    """Split-point deduplication should keep the strongest match for one position."""
    splitter = SemanticSplitter(enable_markdown_patterns=False)
    splitter.patterns = {
        "abstract": re.compile(r"^Abstract$", re.MULTILINE),
        "introduction": re.compile(r"^Abstract$", re.MULTILINE),
    }
    splitter.confidence_scores = {"abstract": 0.95, "introduction": 0.4}

    split_points = splitter._find_semantic_split_points("Abstract\nBody text")

    assert split_points == [(0, "abstract", 0.95, "abstract")]


def test_create_semantic_chunks_without_structure_returns_single_chunk() -> None:
    """Texts without semantic markers should fall back to a single content chunk."""
    splitter = SemanticSplitter(enable_academic_patterns=False, enable_markdown_patterns=False)

    chunks = splitter._create_semantic_chunks("Plain body text only.", [])

    assert len(chunks) == 1
    assert chunks[0].content == "Plain body text only."
    assert chunks[0].metadata["splitter"] == "no_structure"
    assert chunks[0].chunk_name == "unknown"


def test_split_builds_numbered_paths_restores_code_blocks_and_overlap() -> None:
    """Semantic splitting should preserve numbered section paths, code blocks, and overlap metadata."""
    splitter = SemanticSplitter(min_chunk_size=1, chunk_overlap=8)
    text = "\n".join(
        [
            "1 Introduction",
            "Intro body.",
            "",
            "1.1 Setup",
            "```python",
            "print('x')",
            "```",
            "Setup body.",
            "",
            "2 Results",
            "Results body.",
        ]
    )

    chunks = splitter.split(text)

    assert len(chunks) == 3
    assert chunks[0].section_path == ["1 introduction"]
    assert chunks[1].section_path == ["1 introduction", "1.1 setup"]
    assert chunks[2].section_path == ["2 results"]
    assert "```python" in chunks[1].content
    assert "__CODE_BLOCK_" not in chunks[1].content
    assert chunks[0].overlap_next_chars > 0
    assert chunks[1].overlap_prev_chars > 0
