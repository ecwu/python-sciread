"""Compatibility wrapper for document loaders."""

from ...document_ingestion.loaders import BaseLoader
from ...document_ingestion.loaders import LoadResult
from ...document_ingestion.loaders import PdfLoader
from ...document_ingestion.loaders import TxtLoader

__all__ = [
    "BaseLoader",
    "LoadResult",
    "PdfLoader",
    "TxtLoader",
]
