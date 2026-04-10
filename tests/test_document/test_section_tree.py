"""Tests for runtime section-tree helpers and hierarchical paths."""

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.structure.splitters import MarkdownSplitter
from sciread.document.structure.splitters import RegexSectionSplitter
from sciread.document.structure.splitters import SemanticSplitter
from sciread.document.structure.tree import build_section_tree


def test_markdown_splitter_builds_heading_stack_paths() -> None:
    """Markdown headings should become hierarchical section paths."""
    text = "# Introduction\nIntro body.\n\n## Setup\nSetup body.\n\n# Results\nResults body."

    chunks = MarkdownSplitter(min_chunk_size=1).split(text)

    assert [chunk.section_path for chunk in chunks] == [
        ["introduction"],
        ["introduction", "setup"],
        ["results"],
    ]


def test_numbered_sections_build_best_effort_paths_for_semantic_splitter() -> None:
    """Semantic splitter should infer hierarchical paths from numbered headings."""
    text = "1 Introduction\nIntro body.\n\n1.1 Setup\nSetup body.\n\n2 Results\nResults body."

    chunks = SemanticSplitter(min_chunk_size=1, enable_markdown_patterns=False).split(text)

    assert [chunk.section_path for chunk in chunks] == [
        ["1 introduction"],
        ["1 introduction", "1.1 setup"],
        ["2 results"],
    ]


def test_numbered_sections_fallback_to_single_level_for_regex_splitter() -> None:
    """Regex splitter should still emit a section path when only one numbered heading is known."""
    text = "3.2 Ablation Study\nAblation details."

    chunks = RegexSectionSplitter(min_chunk_size=1).split(text)

    assert chunks[0].section_path == ["3", "3.2 ablation study"] or chunks[0].section_path == ["3.2 ablation study"]


def test_section_tree_matches_parent_and_child_paths() -> None:
    """Runtime section trees should preserve parent-child relationships."""
    doc = Document.from_text("placeholder", auto_split=False)
    doc._set_chunks(
        [
            Chunk(content="intro", section_path=["1 introduction"]),
            Chunk(content="setup", section_path=["1 introduction", "1.1 setup"]),
            Chunk(content="result", section_path=["2 results"]),
        ]
    )

    section_tree = build_section_tree(doc)

    assert section_tree.find("1 introduction") is not None
    assert section_tree.find("1 introduction > 1.1 setup") is not None
    rendered = section_tree.render(depth=3)
    assert "1 introduction" in rendered
    assert "1.1 setup" in rendered
