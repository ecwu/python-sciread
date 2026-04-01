"""Section name matching and fuzzy-resolution utilities."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from ..retrieval.service import cosine_similarity

if TYPE_CHECKING:
    from .document import Document


def match_section_pattern(search_name: str, normalized_names: list[str], original_names: list[str]) -> str | None:
    """Match section using common academic paper patterns."""
    section_patterns = {
        "introduction": [
            "intro",
            "introduction",
            "background",
            "overview",
            "prelude",
            "preamble",
        ],
        "abstract": ["abstract", "summary", "executive summary", "overview"],
        "related work": [
            "related work",
            "background",
            "literature review",
            "survey",
            "previous work",
            "state of the art",
        ],
        "methodology": [
            "methodology",
            "method",
            "methods",
            "approach",
            "methodology and approach",
            "technical approach",
            "design",
        ],
        "experiments": [
            "experiment",
            "experiments",
            "experimental setup",
            "evaluation",
            "empirical evaluation",
            "study design",
            "case study",
        ],
        "results": [
            "results",
            "findings",
            "outcomes",
            "performance",
            "evaluation results",
            "experimental results",
        ],
        "discussion": ["discussion", "analysis", "interpretation", "implications"],
        "conclusion": [
            "conclusion",
            "conclusions",
            "summary",
            "future work",
            "concluding remarks",
        ],
        "references": [
            "references",
            "bibliography",
            "citations",
            "works cited",
            "bibliography and references",
        ],
        "appendix": [
            "appendix",
            "appendices",
            "supplementary material",
            "supplemental material",
            "additional information",
        ],
    }

    for _canonical_name, variations in section_patterns.items():
        if search_name in variations:
            for variation in variations:
                if variation in normalized_names:
                    return original_names[normalized_names.index(variation)]

    return None


def word_similarity(str1: str, str2: str) -> float:
    """Calculate word-level Jaccard similarity between two strings."""
    try:
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    except Exception:
        return 0.0


def prefix_similarity(str1: str, str2: str) -> float:
    """Calculate prefix similarity between two strings."""
    try:
        min_len = min(len(str1), len(str2))
        if min_len == 0:
            return 0.0

        common_prefix = 0
        for i in range(min_len):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break

        return common_prefix / min_len
    except Exception:
        return 0.0


def get_closest_section_name(
    document: Document,
    target_name: str,
    available_names: list[str] | None = None,
    case_sensitive: bool = False,
    threshold: float = 0.8,
    use_embedding: bool = False,
) -> str | None:
    """Find closest matching section name using pattern and similarity strategies."""
    try:
        if available_names is None:
            available_names = document.get_section_names()

        if not available_names:
            return None

        if not case_sensitive:
            search_name = target_name.lower()
            normalized_names = [n.lower() for n in available_names]
        else:
            search_name = target_name
            normalized_names = available_names

        best_match = None
        best_score = 0.0

        if search_name in normalized_names:
            return available_names[normalized_names.index(search_name)]

        pattern_match = match_section_pattern(search_name, normalized_names, available_names)
        if pattern_match:
            return pattern_match

        if use_embedding and hasattr(document, "_embedding_client"):
            try:
                target_embedding = document._embedding_client.get_embedding(search_name)
                if target_embedding:
                    for i, name in enumerate(normalized_names):
                        name_embedding = document._embedding_client.get_embedding(name)
                        if name_embedding:
                            similarity = cosine_similarity(target_embedding, name_embedding)
                            if similarity > best_score and similarity >= threshold:
                                best_score = similarity
                                best_match = available_names[i]
            except Exception:
                pass

        if best_match is None:
            for i, name in enumerate(normalized_names):
                sequence_sim = SequenceMatcher(None, search_name, name).ratio()
                word_sim = word_similarity(search_name, name)
                pref_sim = prefix_similarity(search_name, name)
                combined_sim = max(sequence_sim, word_sim, pref_sim)
                if combined_sim > best_score and combined_sim >= threshold:
                    best_score = combined_sim
                    best_match = available_names[i]

        return best_match

    except Exception as e:
        document.logger.error(f"Failed to find closest section name for '{target_name}': {e}")
        return None
