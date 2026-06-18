"""State and chunk operations for Document instances."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .models import Chunk
from .models import DocumentMetadata
from .models import ProcessingState
from .runtime import DocumentRuntimeState

if TYPE_CHECKING:
    from .document import Document


def initialize_document(
    document: Document,
    *,
    source_path: Path | None = None,
    text: str | None = None,
    metadata: DocumentMetadata | None = None,
    processing_state: ProcessingState | None = None,
    is_markdown: bool = False,
) -> None:
    """Initialize the in-memory state for a document."""
    document.source_path = source_path
    document._raw_text = text or ""
    document.metadata = metadata or DocumentMetadata(source_path=source_path)
    if document.metadata.source_path is None:
        document.metadata.source_path = source_path
    document.processing_state = processing_state or ProcessingState()
    document._chunks = []
    document._chunks_by_id = {}
    document._split = False
    document._is_markdown = is_markdown
    document._runtime = DocumentRuntimeState()


def prepare_source_metadata(document: Document, source_path: Path | None = None) -> None:
    """Populate source-derived metadata through the builder/load path only."""
    from .structure.chunking import calculate_file_hash

    resolved_source_path = source_path or document.source_path or document.metadata.source_path
    document.source_path = resolved_source_path
    document.metadata.source_path = resolved_source_path

    if resolved_source_path and resolved_source_path.exists():
        document.metadata.file_hash = calculate_file_hash(resolved_source_path, document.logger)


def attach_document_chunks(document: Document, chunks: list[Chunk]) -> None:
    """Attach normalized chunks to a document."""
    from .structure.chunking import set_document_chunks

    set_document_chunks(document, chunks)


def update_chunk_index(document: Document) -> None:
    """Refresh the chunk-id lookup table."""
    document._chunks_by_id = {chunk.chunk_id: chunk for chunk in document._chunks}
    document._runtime.chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(document._chunks)}


def get_chunk_map(document: Document) -> dict[str, Chunk]:
    """Return the chunk lookup table, refreshing it when needed."""
    if not document._chunks_by_id or len(document._chunks_by_id) != len(document._chunks):
        update_chunk_index(document)
    return document._chunks_by_id


def get_chunks(
    document: Document,
    *,
    limit: int | None = None,
    min_length: int | None = None,
) -> list[Chunk]:
    """Filter chunks using the current document state."""
    chunks = document._chunks

    if min_length is not None:
        chunks = [chunk for chunk in chunks if len(chunk.content) >= min_length]

    if limit is not None:
        chunks = chunks[:limit]

    return chunks


def get_chunk_by_id(document: Document, chunk_id: str) -> Chunk | None:
    """Find a chunk by its identifier."""
    return get_chunk_map(document).get(chunk_id)


def get_section_parts(chunk: Chunk) -> list[str]:
    """Return normalized section path parts for a chunk."""
    section_parts = [part.strip() for part in chunk.section_path if part.strip()]
    if section_parts:
        return section_parts
    return []


def get_chunks_by_section(
    document: Document,
    section: str,
    *,
    include_subsections: bool = True,
) -> list[Chunk]:
    """Return chunks belonging to a section label or section path."""
    normalized_target = " > ".join(part.strip().lower() for part in section.split(">") if part.strip())
    if not normalized_target:
        return []

    matching_chunks: list[Chunk] = []
    for chunk in document._chunks:
        section_parts = get_section_parts(chunk)
        if not section_parts:
            continue

        normalized_parts = [part.lower() for part in section_parts]
        normalized_path = " > ".join(normalized_parts)

        if include_subsections:
            if (
                normalized_path == normalized_target
                or normalized_path.startswith(f"{normalized_target} >")
                or normalized_target in normalized_parts
            ):
                matching_chunks.append(chunk)
        elif normalized_path == normalized_target:
            matching_chunks.append(chunk)

    return matching_chunks


def get_neighbor_chunks(
    document: Document,
    chunk_id: str,
    *,
    before: int = 1,
    after: int = 1,
    include_self: bool = True,
) -> list[Chunk]:
    """Return neighboring chunks around a specific chunk."""
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")

    chunk_positions = document._runtime.chunk_positions
    if len(chunk_positions) != len(document._chunks):
        update_chunk_index(document)
        chunk_positions = document._runtime.chunk_positions

    center_index = chunk_positions.get(chunk_id)

    if center_index is None:
        return []

    start = max(0, center_index - before)
    end = min(len(document._chunks), center_index + after + 1)
    neighbors = document._chunks[start:end]

    if include_self:
        return neighbors

    return [chunk for chunk in neighbors if chunk.chunk_id != chunk_id]


def get_full_text(document: Document, separator: str = "\n\n") -> str:
    """Return full document text, preferring chunk content when present."""
    if not document._chunks:
        return document._raw_text

    if any(chunk.has_overlap for chunk in document._chunks):
        return document._raw_text

    return separator.join(chunk.content for chunk in document._chunks)


def get_section_names(document: Document) -> list[str]:
    """Return section names in document order."""
    section_names: list[str] = []
    for chunk in document._chunks:
        if chunk.section_path:
            section_name = " > ".join(chunk.section_path)
            if section_name not in section_names:
                section_names.append(section_name)
        elif chunk.metadata.get("splitter") and chunk.metadata["splitter"] != "unknown":
            generic_name = f"untitled_{chunk.metadata['splitter']}"
            if generic_name not in section_names:
                section_names.append(generic_name)
    return section_names


def get_sections_by_name(document: Document, section_names: list[str]) -> list[Chunk]:
    """Return chunks matching the requested section names."""
    requested_sections = set(section_names)
    return [chunk for chunk in document._chunks if " > ".join(chunk.section_path) in requested_sections]


def get_runtime_embedding_client(document: Document):
    """Return the cached runtime embedding client, if any."""
    return document._runtime.embedding_client


def set_runtime_embedding_client(document: Document, embedding_client) -> None:
    """Cache an embedding client for future retrieval calls."""
    document._runtime.embedding_client = embedding_client


def get_runtime_rerank_client(document: Document):
    """Return the cached runtime rerank client, if any."""
    return document._runtime.rerank_client


def set_runtime_rerank_client(document: Document, rerank_client) -> None:
    """Cache a rerank client for future retrieval calls."""
    document._runtime.rerank_client = rerank_client
