"""Compatibility wrapper for document renderers."""

from ..document_structure.renderers import clean_section_content
from ..document_structure.renderers import collect_sections
from ..document_structure.renderers import format_for_human
from ..document_structure.renderers import format_for_llm
from ..document_structure.renderers import get_section_overview
from ..document_structure.renderers import get_sections_content
from ..document_structure.renderers import get_sections_with_confidence
from ..document_structure.renderers import remove_references_section
from ..document_structure.renderers import resolve_section_names

__all__ = [
    "clean_section_content",
    "collect_sections",
    "format_for_human",
    "format_for_llm",
    "get_section_overview",
    "get_sections_content",
    "get_sections_with_confidence",
    "remove_references_section",
    "resolve_section_names",
]
