"""Compatibility wrapper for PDF loader."""

from ...document_ingestion.loaders.pdf_loader import BaseLoader
from ...document_ingestion.loaders.pdf_loader import LoadResult
from ...document_ingestion.loaders.pdf_loader import MineruClient
from ...document_ingestion.loaders.pdf_loader import PdfLoader
from ...document_ingestion.loaders.pdf_loader import get_config
from ...document_ingestion.loaders.pdf_loader import get_logger
from ...document_ingestion.loaders.pdf_loader import pdfplumber
from ...document_ingestion.loaders.pdf_loader import pypdf

__all__ = [
    "BaseLoader",
    "LoadResult",
    "MineruClient",
    "PdfLoader",
    "get_config",
    "get_logger",
    "pdfplumber",
    "pypdf",
]
