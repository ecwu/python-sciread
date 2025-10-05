"""Document processing module for sciread.

This module provides functionality for loading, processing, and managing
academic documents through a structured approach with loaders, splitters,
and document management.
"""

from .document import Document
from .models import Chunk
from .models import CoverageStats

__all__ = [
    "Chunk",
    "CoverageStats",
    "Document",
]
