"""Compatibility wrapper for document renderers."""

from ..document_structure.renderers import SHORT_SECTION_THRESHOLD
from ..document_structure.renderers import choose_best_section_match
from ..document_structure.renderers import clean_section_content
from ..document_structure.renderers import collect_sections
from ..document_structure.renderers import format_for_human
from ..document_structure.renderers import format_for_llm
from ..document_structure.renderers import format_section_choices
from ..document_structure.renderers import get_section_length_map
from ..document_structure.renderers import get_section_overview
from ..document_structure.renderers import get_sections_content
from ..document_structure.renderers import get_sections_with_confidence
from ..document_structure.renderers import is_likely_heading_only
from ..document_structure.renderers import remove_references_section
from ..document_structure.renderers import resolve_section_names

__all__ = [
    "SHORT_SECTION_THRESHOLD",
    "choose_best_section_match",
    "clean_section_content",
    "collect_sections",
    "format_for_human",
    "format_for_llm",
    "format_section_choices",
    "get_section_length_map",
    "get_section_overview",
    "get_sections_content",
    "get_sections_with_confidence",
    "is_likely_heading_only",
    "remove_references_section",
    "resolve_section_names",
]
