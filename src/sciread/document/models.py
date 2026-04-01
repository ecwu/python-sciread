"""Compatibility wrapper for document models."""

from ..document_structure.models import Chunk
from ..document_structure.models import DocumentMetadata
from ..document_structure.models import ProcessingState

__all__ = [
    "Chunk",
    "DocumentMetadata",
    "ProcessingState",
]
