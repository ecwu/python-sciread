"""Compatibility wrapper for section resolution helpers."""

from ..document_structure.sections import get_closest_section_name
from ..document_structure.sections import match_section_pattern
from ..document_structure.sections import prefix_similarity
from ..document_structure.sections import word_similarity

__all__ = [
    "get_closest_section_name",
    "match_section_pattern",
    "prefix_similarity",
    "word_similarity",
]
