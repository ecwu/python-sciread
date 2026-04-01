"""Persistence helpers for serialized document state."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..platform.logging import get_logger
from ..retrieval.vector_index import VectorIndex
from .models import Chunk
from .models import DocumentMetadata

if TYPE_CHECKING:
    from .document import Document


def save_document(document: Document, output_path: Path) -> None:
    """Save a document and its chunk state to disk."""
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    document.logger.info(f"Saving document state to {output_path}...")

    try:
        vector_index_path = str(document.vector_index.persist_path.resolve()) if document.vector_index and document.vector_index.persist_path else None

        metadata_dict = asdict(document.metadata)
        if metadata_dict.get("source_path"):
            metadata_dict["source_path"] = str(metadata_dict["source_path"])

        if metadata_dict.get("created_at"):
            metadata_dict["created_at"] = metadata_dict["created_at"].isoformat()
        if metadata_dict.get("modified_at"):
            metadata_dict["modified_at"] = metadata_dict["modified_at"].isoformat()

        chunks_data = []
        for chunk in document._chunks:
            chunk_dict = asdict(chunk)
            chunk_dict.pop("id", None)
            chunks_data.append(chunk_dict)

        state = {
            "metadata": metadata_dict,
            "text": document.text,
            "chunks": chunks_data,
            "vector_index_path": vector_index_path,
            "is_markdown": document.is_markdown,
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=4)

        document.logger.info("Document state saved successfully.")
    except Exception as exc:
        document.logger.error(f"Failed to save document state: {exc}")
        raise RuntimeError(f"Failed to save document state: {exc}") from exc


def load_document(document_cls: type[Document], state_path: Path) -> Document:
    """Load a document instance from serialized state."""
    logger = get_logger(document_cls.__name__)
    logger.info(f"Loading document from state file: {state_path}")

    try:
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)

        metadata_dict = state["metadata"]
        if metadata_dict.get("source_path"):
            metadata_dict["source_path"] = Path(metadata_dict["source_path"])
        if metadata_dict.get("created_at"):
            metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
        if metadata_dict.get("modified_at"):
            metadata_dict["modified_at"] = datetime.fromisoformat(metadata_dict["modified_at"])

        document = document_cls(
            text=state["text"],
            metadata=DocumentMetadata(**metadata_dict),
            _is_markdown=state.get("is_markdown", False),
        )

        document._set_chunks([Chunk(**chunk_data) for chunk_data in state["chunks"]])

        vector_index_path = state.get("vector_index_path")
        if vector_index_path:
            persist_path = Path(vector_index_path)
            if persist_path.exists():
                document.vector_index = VectorIndex(collection_name=persist_path.stem, persist_path=persist_path)
                logger.info("Vector index re-linked successfully")
            else:
                logger.warning(f"Vector index path does not exist: {persist_path}")

        logger.info("Document loaded successfully.")
        return document
    except Exception as exc:
        logger.error(f"Failed to load document from {state_path}: {exc}")
        raise RuntimeError(f"Failed to load document from {state_path}: {exc}") from exc
