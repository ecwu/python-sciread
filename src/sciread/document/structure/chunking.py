"""Chunk normalization helpers for the document subsystem."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

    from sciread.document.document import Document
    from sciread.document.models import Chunk


def calculate_file_hash(file_path: Path, logger: Logger) -> str:
    """Calculate a stable SHA-256 hash for a source file."""
    try:
        hash_sha256 = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as exc:
        logger.warning(f"Failed to calculate file hash for {file_path}: {exc}")
        return file_path.stem


def build_doc_id(document: Document) -> str:
    """Build a stable document identifier used by chunks and retrieval."""
    return document.metadata.file_hash or (
        Path(document.metadata.source_path).stem if document.metadata.source_path else "unnamed_document"
    )


def build_retrieval_text(section_path: list[str], content_plain: str) -> str:
    """Compose retrieval text used by embeddings and rerank flows."""
    if section_path:
        section_label = " > ".join(section_path)
        return f"[Section] {section_label}\n\n{content_plain}"
    return content_plain


def to_plain_text(text: str) -> str:
    """Convert markdown-like content into lexical plain text."""
    plain = text
    plain = re.sub(r"```.*?```", " ", plain, flags=re.DOTALL)
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    plain = re.sub(r"^#{1,6}\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"__([^_]+)__", r"\1", plain)
    plain = re.sub(r"\*([^*]+)\*", r"\1", plain)
    plain = re.sub(r"_([^_]+)_", r"\1", plain)
    plain = re.sub(r"^\s{0,3}>\s?", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*[-*+]\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s*\d+\.\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def enrich_chunks(document: Document, chunks: list[Chunk]) -> None:
    """Fill normalized chunk metadata and maintain neighbor links."""
    doc_id = build_doc_id(document)

    for index, chunk in enumerate(chunks):
        chunk.doc_id = chunk.doc_id or doc_id
        chunk.para_index = index
        chunk.position = index

        if not chunk.content_plain or chunk.content_plain == chunk.content:
            chunk.content_plain = to_plain_text(chunk.content)

        if not chunk.display_text:
            chunk.display_text = chunk.content

        if not chunk.retrieval_text or chunk.retrieval_text == chunk.content or chunk.retrieval_text == chunk.content_plain:
            chunk.retrieval_text = build_retrieval_text(chunk.section_path, chunk.content_plain)

        if chunk.token_count is None:
            chunk.token_count = len(chunk.content_plain.split())

        chunk.sync_page_range()
        chunk.sync_section_metadata()

        if not chunk.citation_key or chunk.citation_key == chunk.chunk_id:
            chunk.citation_key = f"{chunk.doc_id}:{chunk.position}"

        chunk.metadata["section_label"] = " > ".join(chunk.section_path) if chunk.section_path else ""

    for index, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
        chunk.next_chunk_id = chunks[index + 1].chunk_id if index < len(chunks) - 1 else None


def set_document_chunks(document: Document, chunks: list[Chunk]) -> None:
    """Attach chunks to a document and update processing state."""
    enrich_chunks(document, chunks)
    document._chunks = chunks
    document._update_chunks_by_id()
    document._split = len(chunks) > 0
    document.processing_state.update_timestamp("split")
    document.processing_state.add_note(f"Document split into {len(chunks)} chunks")
