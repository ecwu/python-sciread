"""Rendering and section-content helpers for Document consumers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.models import Chunk


SHORT_SECTION_THRESHOLD = 80


def resolve_section_names(
    document: Document,
    section_names: list[str] | None = None,
    max_sections: int | None = None,
) -> list[str]:
    """Resolve section names to use, honoring ordering and limits."""
    names = section_names or document.get_section_names()
    if max_sections is not None:
        names = names[:max_sections]
    return names


def clean_section_content(content: str) -> str:
    """Clean and normalize section content."""
    try:
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
        content = re.sub(r"\b(\w+)-\s*\n\s*(\w+)\b", r"\1\2", content)
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace("'", "'").replace("'", "'")
        content = re.sub(r" +", " ", content)
        return content.strip()
    except Exception:
        return content


def remove_references_section(text: str) -> str:
    """Remove references and bibliography sections from text."""
    try:
        ref_patterns = [
            r"\n\s*(?:references|bibliography|citations|works\s+cited)\s*\n",
            r"\n\s*references\s*$",
            r"\n\s*bibliography\s*$",
            r"\n\s*citations\s*$",
        ]

        earliest_pos = len(text)
        for pattern in ref_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                earliest_pos = min(earliest_pos, matches[0].start())

        if earliest_pos < len(text):
            return text[:earliest_pos].strip()

        return text
    except Exception:
        return text


def collect_sections(
    document: Document,
    section_names: list[str] | None = None,
    max_sections: int | None = None,
    clean_text: bool = False,
    max_chars_per_section: int | None = None,
) -> list[dict]:
    """Collect section data (name, content, chunks, stats) in document order."""
    names = resolve_section_names(document, section_names, max_sections)
    chunks = document.chunks
    if not names or not chunks:
        return []

    allowed_names = set(names)
    grouped_chunks: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        section_label = _section_label(chunk)
        if section_label in allowed_names:
            grouped_chunks.setdefault(section_label, []).append(chunk)

    sections: list[dict] = []
    for name in names:
        chunks = grouped_chunks.get(name, [])
        if not chunks:
            continue

        content = "\n\n".join(chunk.content for chunk in chunks)
        if clean_text:
            content = clean_section_content(content)
        if max_chars_per_section and len(content) > max_chars_per_section:
            content = content[:max_chars_per_section] + "...[truncated]"

        avg_confidence = sum(float(chunk.metadata.get("splitter_confidence", 0.0)) for chunk in chunks) / len(chunks)

        sections.append(
            {
                "name": name,
                "content": content,
                "chunks": chunks,
                "average_confidence": avg_confidence,
                "length": len(content),
            }
        )

    return sections


def _section_label(chunk: Chunk) -> str:
    """Return the normalized section label for a chunk."""
    return " > ".join(chunk.section_path) if chunk.section_path else ""


def get_sections_content(
    document: Document,
    section_names: list[str] | None = None,
    max_sections: int | None = None,
    clean_text: bool = False,
    max_chars_per_section: int | None = None,
) -> list[tuple[str, str]]:
    """Fetch section content with consistent ordering and cleaning."""
    sections = collect_sections(
        document,
        section_names=section_names,
        max_sections=max_sections,
        clean_text=clean_text,
        max_chars_per_section=max_chars_per_section,
    )
    return [(section["name"], section["content"]) for section in sections]


def get_section_length_map(
    document: Document,
    section_names: list[str] | None = None,
) -> dict[str, int]:
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


def format_for_llm(
    document: Document,
    section_names: list[str] | None = None,
    max_tokens: int | None = None,
    include_headers: bool = True,
    clean_text: bool = True,
    max_chars_per_section: int | None = None,
) -> str:
    """Get document content optimized for LLM consumption."""
    try:
        target_names = resolve_section_names(document, section_names)
        sections = collect_sections(
            document,
            section_names=target_names,
            clean_text=clean_text,
            max_chars_per_section=max_chars_per_section,
        )

        content_parts = []
        total_chars = 0
        token_limit = max_tokens * 4 if max_tokens else None

        if include_headers:
            context_parts = []
            if document.metadata.title:
                context_parts.append(f"Title: {document.metadata.title}")
            if document.metadata.author:
                context_parts.append(f"Author: {document.metadata.author}")
            if context_parts:
                content_parts.append("DOCUMENT METADATA:")
                content_parts.extend(context_parts)
                content_parts.append("")

        for section in sections:
            section_name = section["name"] or "untitled"
            content = section["content"]
            section_text = f"=== {section_name.upper()} ===\n{content}\n"
            if token_limit and total_chars + len(section_text) > token_limit:
                remaining = token_limit - total_chars - 50
                if remaining > 200:
                    partial_content = content[:remaining] + "...[truncated due to token limit]"
                    section_text = f"=== {section_name.upper()} ===\n{partial_content}\n"
                    content_parts.append(section_text)
                break

            content_parts.append(section_text)
            total_chars += len(section_text)

        return "\n".join(content_parts)
    except Exception as e:
        document.logger.error(f"Failed to get content for LLM: {e}")
        return f"Error retrieving content: {e}"
