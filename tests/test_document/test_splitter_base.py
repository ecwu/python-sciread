"""Tests for the abstract splitter base helpers."""

import re

import pytest

from sciread.document.models import Chunk
from sciread.document.structure.splitters.base import BaseSplitter


class DummySplitter(BaseSplitter):
    """Minimal concrete splitter for testing shared helper methods."""

    @property
    def splitter_name(self) -> str:
        return "DummySplitter"

    def split(self, text: str) -> list[Chunk]:
        return [Chunk(content=self._validate_text(text))]


def test_validate_text_strips_and_normalizes_line_endings() -> None:
    """Validation should trim whitespace and normalize CRLF variants."""
    splitter = DummySplitter()

    result = splitter._validate_text("  first\r\nsecond\rthird  ")

    assert result == "first\nsecond\nthird"


@pytest.mark.parametrize(
    ("value", "expected_error"),
    [
        (123, TypeError),
        ("   ", ValueError),
    ],
)
def test_validate_text_rejects_invalid_inputs(value: object, expected_error: type[Exception]) -> None:
    """Validation should reject non-strings and empty strings."""
    splitter = DummySplitter()

    with pytest.raises(expected_error):
        splitter._validate_text(value)  # type: ignore[arg-type]


def test_calculate_char_range_returns_end_position() -> None:
    """Character range calculation should include the full chunk length."""
    splitter = DummySplitter()

    assert splitter._calculate_char_range(5, "hello") == (5, 10)


def test_extract_and_restore_code_blocks_handles_fenced_and_indented_code() -> None:
    """Code block helpers should preserve both fenced and indented code."""
    splitter = DummySplitter()
    text = "Before\n```python\nprint('hi')\n```\n\n    indented line\nAfter"
    patterns = {
        "fenced_code": re.compile(r"^```[\w]*\n.*?\n```", re.MULTILINE | re.DOTALL),
        "indented_code": re.compile(r"^(?:\t| {4}).+(?:\n(?:\t| {4}).+)*", re.MULTILINE),
    }

    replaced_text, code_blocks = splitter._extract_code_blocks(text, patterns)
    chunks = [
        Chunk(content=replaced_text),
    ]

    assert "__CODE_BLOCK_0__" in replaced_text
    assert "__CODE_BLOCK_1__" in replaced_text
    assert [block["type"] for block in code_blocks] == ["fenced_code", "indented_code"]

    restored_chunks = splitter._restore_code_blocks(chunks, code_blocks)

    assert restored_chunks[0].content == text
