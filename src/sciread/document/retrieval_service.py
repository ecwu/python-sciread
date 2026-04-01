"""Compatibility wrapper for retrieval services."""

from ..retrieval.service import build_vector_index
from ..retrieval.service import cosine_similarity
from ..retrieval.service import semantic_search

__all__ = [
    "build_vector_index",
    "cosine_similarity",
    "semantic_search",
]
