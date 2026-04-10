"""Helpers for normalized section-path construction."""

from __future__ import annotations

import re

NUMBERED_SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")


def clean_section_name(title: str) -> str:
    """Normalize a section title for matching and path storage."""
    cleaned = title.lower()
    cleaned = re.sub(r"[^\w\s.-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "untitled"


def parse_numbered_section_header(line: str) -> tuple[str, str] | None:
    """Return the section number and normalized title from a numbered heading."""
    match = NUMBERED_SECTION_PATTERN.match(line.strip())
    if not match:
        return None

    section_number = match.group(1)
    title = clean_section_name(match.group(2))
    return section_number, title


def build_numbered_section_path(
    section_number: str,
    title: str,
    known_titles: dict[str, str],
) -> list[str]:
    """Build a best-effort hierarchical path for a numbered section."""
    parts = section_number.split(".")
    prefixes = [".".join(parts[: index + 1]) for index in range(len(parts))]

    path: list[str] = []
    for prefix in prefixes[:-1]:
        ancestor_title = known_titles.get(prefix)
        if ancestor_title:
            path.append(f"{prefix} {ancestor_title}")
        else:
            path.append(prefix)

    label = f"{section_number} {title}".strip()
    if path and path[-1] == label:
        return path
    path.append(label)
    return path


def get_parent_section_id(section_path: list[str]) -> str | None:
    """Return the parent-section identifier for a normalized path."""
    if not section_path:
        return None
    if len(section_path) == 1:
        return section_path[0]
    return " > ".join(section_path[:-1])
