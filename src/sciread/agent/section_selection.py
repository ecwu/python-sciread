"""Shared helpers for section selection across agents."""

from __future__ import annotations

from ..document_structure import Document
from ..document_structure.renderers import get_sections_content

SHORT_SECTION_THRESHOLD = 80


def get_section_length_map(document: Document, section_names: list[str] | None = None) -> dict[str, int]:
    """Return clean-text lengths for requested sections in document order."""
    names = section_names or document.get_section_names()
    section_lengths = dict.fromkeys(names, 0)

    for section_name, content in get_sections_content(document, section_names=names, clean_text=True):
        section_lengths[section_name] = len(content.strip())

    return section_lengths


def is_likely_heading_only(section_length: int, threshold: int = SHORT_SECTION_THRESHOLD) -> bool:
    """Return whether a section is likely just a heading or transition sentence."""
    return section_length <= threshold


def format_section_choices(
    section_names: list[str],
    section_lengths: dict[str, int],
    *,
    numbered: bool = False,
    threshold: int = SHORT_SECTION_THRESHOLD,
) -> str:
    """Format section choices with clean-text length annotations."""
    lines: list[str] = []
    for index, section_name in enumerate(section_names, start=1):
        prefix = f"{index}. " if numbered else "- "
        section_length = section_lengths.get(section_name, 0)
        short_hint = " | 可能仅标题" if is_likely_heading_only(section_length, threshold) else ""
        lines.append(f"{prefix}{section_name} | {section_length} chars{short_hint}")
    return "\n".join(lines)


def _match_score(target: str, section_name: str) -> int:
    """Compute a simple lexical match score for target-to-section alignment."""
    target_lower = target.lower()
    section_lower = section_name.lower()

    if target_lower == section_lower:
        return 100
    if target_lower in section_lower:
        return 80
    if section_lower in target_lower:
        return 60

    target_words = [word for word in target_lower.replace("_", " ").split() if word]
    return sum(10 for word in target_words if word in section_lower)


def choose_best_section_match(
    target: str,
    available_sections: list[str],
    section_lengths: dict[str, int],
    *,
    threshold: int = SHORT_SECTION_THRESHOLD,
) -> str | None:
    """Choose the best section match, preferring substantial content over heading-only sections."""
    scored_candidates: list[tuple[int, int, str]] = []

    for section_name in available_sections:
        match_score = _match_score(target, section_name)
        if match_score <= 0:
            continue

        section_length = section_lengths.get(section_name, 0)
        if is_likely_heading_only(section_length, threshold):
            match_score -= 25

        scored_candidates.append((match_score, section_length, section_name))

    if not scored_candidates:
        return None

    return max(scored_candidates, key=lambda item: (item[0], item[1], -available_sections.index(item[2])))[2]
