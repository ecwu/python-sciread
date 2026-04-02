"""Tests for shared text processing utilities."""

from sciread.agent.shared.text_utils import clean_academic_text
from sciread.agent.shared.text_utils import extract_document_metadata
from sciread.agent.shared.text_utils import remove_references


def test_remove_references_returns_input_for_blank_text() -> None:
    """Blank text should be returned unchanged."""
    assert remove_references("") == ""
    assert remove_references("   ") == "   "


def test_remove_references_trims_long_reference_sections() -> None:
    """Long reference tails should be removed."""
    text = "Introduction\nMain body\nReferences\n" + ("citation\n" * 400)

    result = remove_references(text)

    assert result == "Introduction\nMain body"


def test_remove_references_keeps_short_reference_sections() -> None:
    """Short trailing reference sections should be preserved."""
    text = "Introduction\nMain body\nReferences\n[1] Short citation"

    result = remove_references(text)

    assert result == text


def test_remove_references_supports_custom_keywords() -> None:
    """Custom cutoff keywords should be honored."""
    text = "Body\nSUPPLEMENT\n" + ("appendix\n" * 400)

    result = remove_references(text, reference_keywords=["supplement"])

    assert result == "Body"


def test_clean_academic_text_normalizes_extraction_artifacts() -> None:
    """Text cleaning should normalize compact extraction artifacts."""
    text = "camelCase model2Version 3beta"

    result = clean_academic_text(text)

    assert result == "camel Case model 2 Version 3 beta"


def test_clean_academic_text_drops_page_numbers_and_urls() -> None:
    """Standalone page-number and URL lines should be removed."""
    assert clean_academic_text("7") == ""
    assert clean_academic_text("https://example.com/paper") == ""


def test_extract_document_metadata_finds_title_authors_and_abstract() -> None:
    """Metadata extraction should skip affiliations and collect the abstract."""
    text = "\n".join(
        [
            "Department of Computer Science",
            "GNNs for Molecules",
            "Alice Smith, Bob Jones",
            "Abstract",
            "We study graph neural networks for molecules.",
            "We report strong results.",
            "Keywords: chemistry",
            "Introduction",
            "Later section",
        ]
    )

    metadata = extract_document_metadata(text)

    assert metadata == {
        "title": "GNNs for Molecules",
        "authors": "Alice Smith, Bob Jones",
        "abstract": "We study graph neural networks for molecules. We report strong results.",
    }


def test_extract_document_metadata_returns_empty_dict_when_no_fields_found() -> None:
    """Metadata extraction should degrade gracefully on sparse text."""
    metadata = extract_document_metadata("\n\nLab\n\n")

    assert metadata == {}
