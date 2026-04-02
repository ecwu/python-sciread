"""Document structure helpers, splitters, and section utilities."""

from .renderers import choose_best_section_match
from .renderers import clean_section_content
from .renderers import collect_sections
from .renderers import format_for_human
from .renderers import format_for_llm
from .renderers import format_section_choices
from .renderers import get_section_length_map
from .renderers import get_section_overview
from .renderers import get_sections_content
from .renderers import get_sections_with_confidence
from .renderers import is_likely_heading_only
from .renderers import remove_references_section
from .renderers import resolve_section_names
from .sections import get_closest_section_name
from .sections import match_section_pattern
from .sections import prefix_similarity
from .sections import word_similarity
from .splitters import BaseSplitter
from .splitters import MarkdownSplitter
from .splitters import RegexSectionSplitter
from .splitters import SemanticSplitter

__all__ = [
    "BaseSplitter",
    "MarkdownSplitter",
    "RegexSectionSplitter",
    "SemanticSplitter",
    "choose_best_section_match",
    "clean_section_content",
    "collect_sections",
    "format_for_human",
    "format_for_llm",
    "format_section_choices",
    "get_closest_section_name",
    "get_section_length_map",
    "get_section_overview",
    "get_sections_content",
    "get_sections_with_confidence",
    "is_likely_heading_only",
    "match_section_pattern",
    "prefix_similarity",
    "remove_references_section",
    "resolve_section_names",
    "word_similarity",
]
