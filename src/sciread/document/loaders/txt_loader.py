"""Compatibility wrapper for text loader."""

from ...document_ingestion.loaders.txt_loader import BaseLoader
from ...document_ingestion.loaders.txt_loader import LoadResult
from ...document_ingestion.loaders.txt_loader import TxtLoader

__all__ = [
    "BaseLoader",
    "LoadResult",
    "TxtLoader",
]
