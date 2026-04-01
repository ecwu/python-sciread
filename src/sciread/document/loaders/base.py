"""Compatibility wrapper for loader base types."""

from ...document_ingestion.loaders.base import BaseLoader
from ...document_ingestion.loaders.base import LoadResult

__all__ = [
    "BaseLoader",
    "LoadResult",
]
