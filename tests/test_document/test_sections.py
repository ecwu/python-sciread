"""Tests for section matching helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

from sciread.document import Document
from sciread.document.models import Chunk
from sciread.document.structure.sections import get_closest_section_name
from sciread.document.structure.sections import match_section_pattern
from sciread.document.structure.sections import prefix_similarity
from sciread.document.structure.sections import word_similarity


def test_match_section_pattern_maps_common_academic_aliases() -> None:
    """Common aliases should resolve to the original section name."""
    normalized_names = ["overview", "methodology", "results"]
    original_names = ["Overview", "Methodology", "Results"]

    assert match_section_pattern("intro", normalized_names, original_names) == "Overview"
    assert match_section_pattern("method", normalized_names, original_names) == "Methodology"
    assert match_section_pattern("unknown", normalized_names, original_names) is None


def test_word_similarity_and_prefix_similarity_handle_matches_and_bad_inputs() -> None:
    """Similarity helpers should handle both normal and invalid inputs safely."""
    assert word_similarity("graph neural networks", "graph networks") == 2 / 3
    assert word_similarity(object(), "graph") == 0.0  # type: ignore[arg-type]
    assert prefix_similarity("introduction", "intro") == 1.0
    assert prefix_similarity(object(), "intro") == 0.0  # type: ignore[arg-type]


def test_get_closest_section_name_supports_case_sensitive_and_fuzzy_matching() -> None:
    """Section lookup should cover exact, case-sensitive, and fuzzy resolution."""
    doc = Document.from_text("placeholder", auto_split=False)
    doc._set_chunks(
        [
            Chunk(content="a", chunk_name="Introduction"),
            Chunk(content="b", chunk_name="Methodology"),
            Chunk(content="c", chunk_name="Results"),
        ]
    )

    assert get_closest_section_name(doc, "Methodology", case_sensitive=True) == "Methodology"
    assert get_closest_section_name(doc, "intro", threshold=0.7) == "Introduction"
    assert get_closest_section_name(doc, "findings", threshold=0.7) == "Results"


def test_get_closest_section_name_can_use_embeddings() -> None:
    """Embedding-based section lookup should return the highest-similarity match above threshold."""
    doc = Document.from_text("placeholder", auto_split=False)
    doc._set_chunks(
        [
            Chunk(content="a", chunk_name="Alpha"),
            Chunk(content="b", chunk_name="Beta"),
        ]
    )
    embedding_client = Mock()
    embedding_client.get_embeddings.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    doc._runtime.embedding_client = embedding_client

    match = get_closest_section_name(doc, "gamma", use_embedding=True, threshold=0.8)

    assert match == "Alpha"
    embedding_client.get_embeddings.assert_called_once_with(["gamma", "alpha", "beta"], batch_size=3)


def test_get_closest_section_name_returns_none_for_empty_available_names() -> None:
    """Missing section names should produce no match."""
    doc = Document.from_text("placeholder", auto_split=False)

    assert get_closest_section_name(doc, "intro", available_names=[]) is None


def test_get_closest_section_name_logs_and_returns_none_on_unexpected_error() -> None:
    """Unexpected lookup failures should be logged and suppressed."""
    logger = Mock()
    broken_document = SimpleNamespace(logger=logger, get_section_names=Mock(side_effect=RuntimeError("boom")))

    result = get_closest_section_name(broken_document, "intro")  # type: ignore[arg-type]

    assert result is None
    logger.error.assert_called_once()
