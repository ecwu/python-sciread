"""Main Document class for managing document content and chunks."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from ..config import get_config
from ..logging_config import get_logger
from .external_clients import OllamaClient
from .factory import DocumentFactory
from .models import Chunk
from .models import DocumentMetadata
from .models import ProcessingState
from .vector_index import VectorIndex


class Document:
    """Data container for document content and chunks."""

    def __init__(
        self,
        source_path: Optional[Path] = None,
        text: Optional[str] = None,
        metadata: Optional[DocumentMetadata] = None,
        processing_state: Optional[ProcessingState] = None,
        _is_markdown: bool = False,
    ):
        """Initialize a Document instance.

        Args:
            source_path: Path to the source file (optional).
            text: Raw text content.
            metadata: Document metadata.
            processing_state: Processing state information.
            _is_markdown: Internal flag to track if content is markdown-converted.
        """
        self.logger = get_logger(__name__)
        self.source_path = source_path
        self._raw_text = text or ""
        self.metadata = metadata or DocumentMetadata(source_path=source_path)

        # Calculate file hash if source_path is provided
        if source_path and source_path.exists():
            self.metadata.file_hash = self._calculate_file_hash(source_path)

        self.processing_state = processing_state or ProcessingState()
        self._chunks: list[Chunk] = []
        self._chunks_by_id: dict[str, Chunk] = {}
        self._split = False
        self._is_markdown = _is_markdown
        self.vector_index: Optional[VectorIndex] = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path], to_markdown: bool = False, auto_split: bool = True, **split_kwargs) -> "Document":
        """
        Create a Document from a file path.

        Args:
            file_path: Path to the file.
            to_markdown: If True, convert PDF to markdown using Mineru API.
            auto_split: Whether to automatically split the document after loading.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with loaded content and optionally split chunks.
        """
        return DocumentFactory.create_from_file(file_path, to_markdown=to_markdown)

    @classmethod
    def from_text(cls, text: str, metadata: Optional[DocumentMetadata] = None, auto_split: bool = True, **split_kwargs) -> "Document":
        """
        Create a Document from raw text.

        Args:
            text: Raw text content.
            metadata: Optional document metadata.
            auto_split: Whether to automatically split the document.
            **split_kwargs: Additional arguments for splitting.

        Returns:
            Document instance with text and optionally split chunks.
        """
        return DocumentFactory.create_from_text(text, metadata=metadata)

    @property
    def chunks(self) -> list[Chunk]:
        """Get all chunks."""
        return self._chunks.copy()

    @property
    def text(self) -> str:
        """Get the raw text content."""
        return self._raw_text

    @property
    def is_split(self) -> bool:
        """Check if document is split."""
        return self._split

    @property
    def is_markdown(self) -> bool:
        """Check if document content is markdown-converted."""
        return self._is_markdown

    def get_chunks(
        self,
        processed: Optional[bool] = None,
        chunk_name: Optional[str] = None,
        limit: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
    ) -> list[Chunk]:
        """
        Get chunks with flexible filtering criteria.

        This is the unified method for accessing chunks with any combination of filters.
        It replaces the previous multiple specialized methods (get_unprocessed_chunks,
        get_quality_chunks, filter_chunks, etc.).

        Args:
            processed: Filter by processing status (True for processed, False for unprocessed).
            chunk_name: Filter by specific chunk name (section name).
            limit: Maximum number of chunks to return.
            confidence_threshold: Minimum confidence score (0.0-1.0).
            min_length: Minimum character length for chunks.
            exclude_types: Set of chunk types to exclude.

        Returns:
            List of chunks matching the specified criteria.

        Examples:
            # Get all unprocessed chunks
            doc.get_chunks(processed=False)

            # Get high-quality chunks
            doc.get_chunks(confidence_threshold=0.7, min_length=100)

            # Get abstract chunks that haven't been processed yet
            doc.get_chunks(chunk_name="abstract", processed=False)

            # Get first 5 chunks with high confidence
            doc.get_chunks(confidence_threshold=0.8, limit=5)
        """
        chunks = self._chunks

        # Filter by processing status
        if processed is not None:
            chunks = [chunk for chunk in chunks if chunk.processed == processed]

        # Filter by chunk name
        if chunk_name is not None:
            chunks = [chunk for chunk in chunks if chunk.chunk_name == chunk_name]

        # Filter by confidence threshold
        if confidence_threshold is not None:
            chunks = [chunk for chunk in chunks if (chunk.confidence or 0.0) >= confidence_threshold]

        # Filter by minimum length
        if min_length is not None:
            chunks = [chunk for chunk in chunks if len(chunk.content) >= min_length]

        # Exclude specific chunk types
        if exclude_types:
            chunks = [chunk for chunk in chunks if chunk.chunk_name not in exclude_types]

        # Apply limit
        if limit is not None:
            chunks = chunks[:limit]

        return chunks

    def get_unprocessed_chunks(self, limit: Optional[int] = None) -> list[Chunk]:
        """Get unprocessed chunks. Convenience method for get_chunks(processed=False)."""
        return self.get_chunks(processed=False, limit=limit)

    def get_quality_chunks(
        self,
        confidence_threshold: float = 0.5,
        min_length: int = 50,
        exclude_types: Optional[set[str]] = None,
        processed: Optional[bool] = None,
    ) -> list[Chunk]:
        """Get high-quality chunks based on quality criteria. Convenience method."""
        return self.get_chunks(
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
            processed=processed,
        )

    def filter_chunks(self, **filter_kwargs) -> list[Chunk]:
        """Filter chunks using any combination of criteria. Alias for get_chunks()."""
        return self.get_chunks(**filter_kwargs)

    def mark_chunks_processed(
        self,
        confidence_threshold: Optional[float] = None,
        min_length: Optional[int] = None,
        exclude_types: Optional[set[str]] = None,
    ) -> int:
        """
        Mark chunks as processed based on filtering criteria.

        This replaces the old deactivate_low_quality_chunks method with clearer semantics.

        Args:
            confidence_threshold: Mark chunks below this threshold as processed.
            min_length: Mark chunks shorter than this as processed.
            exclude_types: Don't mark these chunk types as processed.

        Returns:
            Number of chunks marked as processed.
        """
        chunks_to_mark = self.get_chunks(
            processed=False,
            confidence_threshold=confidence_threshold,
            min_length=min_length,
            exclude_types=exclude_types,
        )

        # Mark chunks as processed
        for chunk in chunks_to_mark:
            chunk.mark_processed()

        if chunks_to_mark:
            self.processing_state.add_note(f"Marked {len(chunks_to_mark)} chunks as processed")
            self.logger.info(f"Marked {len(chunks_to_mark)} chunks as processed")

        return len(chunks_to_mark)

    def next_unprocessed(self) -> Optional[Chunk]:
        """Get the next unprocessed chunk."""
        unprocessed = self.get_unprocessed_chunks(limit=1)
        return unprocessed[0] if unprocessed else None

    def mark_all_processed(self) -> None:
        """Mark all chunks as processed."""
        for chunk in self._chunks:
            chunk.mark_processed()
        self.processing_state.update_timestamp("processed")
        self.processing_state.add_note("All chunks marked as processed")

    def mark_all_unprocessed(self) -> None:
        """Mark all chunks as unprocessed."""
        for chunk in self._chunks:
            chunk.mark_unprocessed()

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Get the full text by joining all chunks with separator."""
        if not self._chunks:
            return self._raw_text
        return separator.join(chunk.content for chunk in self._chunks)

    def get_section_names(self) -> list[str]:
        """Return ordered list of section names from all chunks.

        Section names are extracted from chunk metadata if available.
        Chunks without section names are skipped.

        Returns:
            List of section names in document order.
        """
        section_names = []
        for chunk in self._chunks:
            if chunk.chunk_name and chunk.chunk_name != "unknown":
                section_name = chunk.chunk_name
                if section_name not in section_names:  # Avoid duplicates
                    section_names.append(section_name)
            elif chunk.metadata.get("splitter") and chunk.metadata["splitter"] != "unknown":
                # For chunks without explicit section names, use a generic name based on splitter type
                splitter_type = chunk.metadata["splitter"]
                generic_name = f"untitled_{splitter_type}"
                if generic_name not in section_names:
                    section_names.append(generic_name)
        return section_names

    def get_sections_by_name(self, section_names: list[str]) -> list[Chunk]:
        """Get chunks matching specific section names.

        Args:
            section_names: List of section names to filter by.

        Returns:
            List of chunks matching the specified section names.
        """
        matching_chunks = []
        for chunk in self._chunks:
            if chunk.chunk_name in section_names:
                matching_chunks.append(chunk)
        return matching_chunks

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks."""
        return iter(self._chunks)

    def __len__(self) -> int:
        """Get the number of chunks."""
        return len(self._chunks)

    def __getitem__(self, index: Union[int, slice]) -> Union[Chunk, list[Chunk]]:
        """Get chunk by index."""
        return self._chunks[index]

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            hash_sha256 = hashlib.sha256()
            with file_path.open("rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate file hash for {file_path}: {e}")
            return file_path.stem

    def _update_chunks_by_id(self) -> None:
        """Update the _chunks_by_id dictionary to match current chunks."""
        self._chunks_by_id = {chunk.id: chunk for chunk in self._chunks}

    def _set_chunks(self, chunks: List[Chunk]) -> None:
        """Set chunks and update the _chunks_by_id dictionary."""
        self._chunks = chunks
        self._update_chunks_by_id()
        self._split = len(chunks) > 0
        self.processing_state.update_timestamp("split")
        self.processing_state.add_note(f"Document split into {len(chunks)} chunks")

    def build_vector_index(self, persist: bool = False) -> None:
        """Builds a semantic vector index from the document's chunks."""
        if not self._chunks:
            self.logger.warning("No chunks to index. Please split the document first.")
            return

        self.logger.info(f"Building vector index from {len(self._chunks)} chunks...")

        try:
            config = get_config()
            vector_config = config.vector_store
            ollama_client = OllamaClient(model=vector_config.embedding_model, cache_embeddings=vector_config.cache_embeddings)
            embeddings = ollama_client.get_embeddings([c.content for c in self._chunks], batch_size=vector_config.batch_size)

            persist_path = None
            if persist:
                store_path = Path(vector_config.path).expanduser()
                store_path.mkdir(parents=True, exist_ok=True)
                doc_id = self.metadata.file_hash or (
                    Path(self.metadata.source_path).stem if self.metadata.source_path else "unnamed_document"
                )
                persist_path = store_path / doc_id

            collection_name = self.metadata.file_hash or (
                Path(self.metadata.source_path).stem if self.metadata.source_path else "unnamed_document"
            )
            self.vector_index = VectorIndex(collection_name=collection_name, persist_path=persist_path)
            self.vector_index.add_chunks(self._chunks, embeddings)
            self.logger.info("Vector index built successfully.")

        except Exception as e:
            self.logger.error(f"Failed to build vector index: {e}")
            raise RuntimeError(f"Failed to build vector index: {e}") from e

    def semantic_search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Performs a semantic search on the document chunks."""
        if not self.vector_index:
            self.logger.warning("Vector index not found. Please run `build_vector_index()` first.")
            return []

        if not self._chunks_by_id:
            self._update_chunks_by_id()

        self.logger.info(f"Performing semantic search for: '{query}'")

        try:
            config = get_config()
            vector_config = config.vector_store
            ollama_client = OllamaClient(model=vector_config.embedding_model, cache_embeddings=vector_config.cache_embeddings)

            query_embedding = ollama_client.get_embedding(query)
            if not query_embedding:
                self.logger.error("Failed to get embedding for query")
                return []

            search_results = self.vector_index.search(query_embedding, top_k=top_k)
            found_chunks = [self._chunks_by_id[res["id"]] for res in search_results if res["id"] in self._chunks_by_id]

            self.logger.info(f"Found {len(found_chunks)} matching chunks")
            return found_chunks

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    def save(self, output_path: Path) -> None:
        """Saves the document's state and its vector index path to a JSON file."""
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        self.logger.info(f"Saving document state to {output_path}...")

        try:
            vector_index_path_str = (
                str(self.vector_index.persist_path.resolve()) if self.vector_index and self.vector_index.persist_path else None
            )

            # Convert metadata to dict, handling Path objects and None values
            metadata_dict = asdict(self.metadata)
            if metadata_dict.get("source_path"):
                metadata_dict["source_path"] = str(metadata_dict["source_path"])

            # Convert datetime objects to ISO format strings
            if metadata_dict.get("created_at"):
                metadata_dict["created_at"] = metadata_dict["created_at"].isoformat()
            if metadata_dict.get("modified_at"):
                metadata_dict["modified_at"] = metadata_dict["modified_at"].isoformat()

            # Convert chunks to dict, excluding the id field since it's init=False
            chunks_data = []
            for chunk in self._chunks:
                chunk_dict = asdict(chunk)
                # Remove id field since it's not part of __init__ parameters
                chunk_dict.pop("id", None)
                chunks_data.append(chunk_dict)

            doc_state = {
                "metadata": metadata_dict,
                "text": self._raw_text,
                "chunks": chunks_data,
                "vector_index_path": vector_index_path_str,
                "is_markdown": self._is_markdown,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_state, f, indent=4)
            self.logger.info("Document state saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save document state: {e}")
            raise RuntimeError(f"Failed to save document state: {e}") from e

    @classmethod
    def load(cls, state_path: Path) -> "Document":
        """Loads a document from a state file, re-linking its vector index."""
        logger = get_logger(cls.__name__)
        logger.info(f"Loading document from state file: {state_path}")

        try:
            with open(state_path, encoding="utf-8") as f:
                doc_state = json.load(f)

            # Reconstruct metadata with proper Path object and datetime handling
            metadata_dict = doc_state["metadata"]
            if metadata_dict.get("source_path"):
                metadata_dict["source_path"] = Path(metadata_dict["source_path"])

            # Handle datetime parsing
            from datetime import datetime

            if metadata_dict.get("created_at"):
                metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
            if metadata_dict.get("modified_at"):
                metadata_dict["modified_at"] = datetime.fromisoformat(metadata_dict["modified_at"])

            metadata = DocumentMetadata(**metadata_dict)

            # Create document instance
            doc = cls(text=doc_state["text"], metadata=metadata, _is_markdown=doc_state.get("is_markdown", False))

            # Reconstruct chunks
            doc._chunks = []
            for chunk_data in doc_state["chunks"]:
                chunk = Chunk(**chunk_data)
                doc._chunks.append(chunk)
            doc._update_chunks_by_id()

            # Re-link vector index if available
            vector_index_path_str = doc_state.get("vector_index_path")
            if vector_index_path_str:
                logger.info(f"Re-linking vector index from: {vector_index_path_str}")
                persist_path = Path(vector_index_path_str)
                if persist_path.exists():
                    collection_name = persist_path.stem
                    doc.vector_index = VectorIndex(collection_name=collection_name, persist_path=persist_path)
                    logger.info("Vector index re-linked successfully")
                else:
                    logger.warning(f"Vector index path does not exist: {persist_path}")

            logger.info("Document loaded successfully.")
            return doc

        except Exception as e:
            logger.error(f"Failed to load document from {state_path}: {e}")
            raise RuntimeError(f"Failed to load document from {state_path}: {e}") from e
