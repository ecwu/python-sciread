"""Compatibility wrapper for document splitters."""

from ...document_structure.splitters import BaseSplitter
from ...document_structure.splitters import MarkdownSplitter
from ...document_structure.splitters import RegexSectionSplitter
from ...document_structure.splitters import SemanticSplitter

__all__ = [
    "BaseSplitter",
    "MarkdownSplitter",
    "RegexSectionSplitter",
    "SemanticSplitter",
]
