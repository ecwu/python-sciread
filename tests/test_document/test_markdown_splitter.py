"""Tests for the markdown-aware splitter."""

import pytest

from sciread.document.structure.splitters.markdown_splitter import MarkdownSplitter


def test_splitter_name_reports_configuration() -> None:
    """Splitter name should reflect min and max sizes."""
    splitter = MarkdownSplitter(min_chunk_size=10, max_chunk_size=50)

    assert splitter.splitter_name == "MarkdownSplitter(min=10, max=50)"


def test_split_without_headers_returns_single_content_chunk() -> None:
    """Plain text with header splitting disabled should produce one chunk."""
    splitter = MarkdownSplitter(split_on_headers=False)

    chunks = splitter.split("Plain body text without markdown headers.")

    assert len(chunks) == 1
    assert chunks[0].content == "Plain body text without markdown headers."
    assert chunks[0].metadata["splitter"] == "no_structure"
    assert chunks[0].position == 0


def test_split_with_preamble_and_headers_assigns_sections_and_positions() -> None:
    """Markdown headers should create section-aware chunks."""
    splitter = MarkdownSplitter()
    text = "\n".join(
        [
            "Preface text before the first heading.",
            "# Introduction",
            "Intro content",
            "## Results",
            "Result content",
        ]
    )

    chunks = splitter.split(text)

    assert [chunk.chunk_name for chunk in chunks] == ["preamble", "introduction", "results"]
    assert [chunk.position for chunk in chunks] == [0, 1, 2]
    assert chunks[1].section_path == ["introduction"]
    assert chunks[2].section_path == ["results"]


def test_split_restores_preserved_code_blocks() -> None:
    """Fenced code blocks should survive placeholder extraction and restoration."""
    splitter = MarkdownSplitter(preserve_code_blocks=True)
    text = "\n".join(
        [
            "# Method",
            "```python",
            "print('hello')",
            "```",
            "Explanation.",
        ]
    )

    chunks = splitter.split(text)

    assert len(chunks) == 1
    assert "```python" in chunks[0].content
    assert "__CODE_BLOCK_" not in chunks[0].content
    assert chunks[0].metadata["splitter"] == "methods"


@pytest.mark.parametrize(
    ("content", "expected_type", "expected_confidence"),
    [
        ("```python\nprint('x')\n```", "code", 0.90),
        ("| a | b |\n|---|---|", "table", 0.65),
        ("- bullet item", "list", 0.60),
        ("> quoted line", "blockquote", 0.65),
        ("Abstract\nThis summarizes the paper.", "abstract", 0.80),
        ("# Header\nbody", "h1", 0.95),
        ("Plain prose only.", "content", 0.33),
    ],
)
def test_analyze_chunk_content_detects_markdown_and_academic_patterns(
    content: str,
    expected_type: str,
    expected_confidence: float,
) -> None:
    """Chunk content analysis should classify common markdown patterns."""
    splitter = MarkdownSplitter()

    chunk_type, confidence = splitter._analyze_chunk_content(content, 0.33)

    assert chunk_type == expected_type
    assert confidence == expected_confidence


def test_clean_section_name_normalizes_symbols_and_empty_results() -> None:
    """Section names should be normalized into stable identifiers."""
    splitter = MarkdownSplitter()

    assert splitter._clean_section_name(" Results & Discussion! ") == "results discussion"
    assert splitter._clean_section_name("!!!") == "untitled"


def test_extract_section_from_content_reads_markdown_headers() -> None:
    """Section extraction should only trigger when content starts with a header."""
    splitter = MarkdownSplitter()

    assert splitter._extract_section_from_content("## Related Work\nBody") == "related work"
    assert splitter._extract_section_from_content("Body\n## Related Work") is None


def test_add_and_remove_pattern_updates_registry() -> None:
    """Custom pattern registration should support add/remove cycles."""
    splitter = MarkdownSplitter()

    splitter.add_pattern("custom", r"^NOTE:", 0.55)
    assert "custom" in splitter.patterns
    assert splitter.confidence_scores["custom"] == 0.55

    assert splitter.remove_pattern("custom") is True
    assert "custom" not in splitter.patterns
    assert splitter.remove_pattern("missing") is False


def test_add_pattern_rejects_invalid_regex() -> None:
    """Invalid custom patterns should raise a ValueError."""
    splitter = MarkdownSplitter()

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        splitter.add_pattern("broken", "[", 0.5)
