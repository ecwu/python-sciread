"""Shared helpers for application use cases."""

from __future__ import annotations

from pathlib import Path

from ...document_structure import Document


def ensure_file_exists(document_file_path: str) -> Path:
    """Validate the input path before a use case runs."""
    path = Path(document_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {document_file_path}")
    return path


def load_document(document_file_path: str, *, to_markdown: bool) -> Document:
    """Load and split a document with the standard project defaults."""
    ensure_file_exists(document_file_path)
    return Document.from_file(document_file_path, to_markdown=to_markdown, auto_split=True)
