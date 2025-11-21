"""Main Document class for managing document content and chunks."""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ..config import get_config
from ..embedding_provider import get_embedding_client
from ..logging_config import get_logger
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
    def from_file(
        cls,
        file_path: Union[str, Path],
        to_markdown: bool = False,
        auto_split: bool = True,
        **split_kwargs,
    ) -> "Document":
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
    def from_text(
        cls,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
        auto_split: bool = True,
        **split_kwargs,
    ) -> "Document":
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

    def build_vector_index(self, persist: bool = False, embedding_client=None) -> None:
        """Builds a semantic vector index from the document's chunks.

        Args:
            persist: Whether to persist the index to disk
            embedding_client: Optional embedding client (OllamaClient or SiliconFlowClient).
                            If not provided, uses OllamaClient from config.
        """
        if not self._chunks:
            self.logger.warning("No chunks to index. Please split the document first.")
            return

        self.logger.info(f"Building vector index from {len(self._chunks)} chunks...")

        try:
            # Use provided embedding client or create default from config
            if embedding_client is None:
                config = get_config()
                vector_config = config.vector_store
                # Use embedding provider system to create client
                embedding_client = get_embedding_client(
                    vector_config.embedding_model,
                    cache_embeddings=vector_config.cache_embeddings,
                )

            # Store the embedding client for later use in semantic_search
            self._embedding_client = embedding_client

            batch_size = 10  # Default batch size
            if hasattr(embedding_client, "embedding_batch_size"):
                batch_size = embedding_client.embedding_batch_size

            embeddings = embedding_client.get_embeddings([c.content for c in self._chunks], batch_size=batch_size)

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

    def semantic_search(self, query: str, top_k: int = 5, return_scores: bool = False) -> Union[List[Chunk], List[tuple[Chunk, float]]]:
        """Performs a semantic search on the document chunks using cosine similarity.

        This method uses cosine similarity for ranking, which is length-invariant
        and provides better semantic matching than L2 distance. Results are ranked
        by similarity score (higher is better).

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: If True, returns tuples of (chunk, similarity_score)

        Returns:
            List of matching chunks, or list of (chunk, score) tuples if return_scores=True
            Similarity scores are in range [0, 1] where 1 is most similar
        """
        if not self.vector_index:
            self.logger.warning("Vector index not found. Please run `build_vector_index()` first.")
            return []

        if not self._chunks_by_id:
            self._update_chunks_by_id()

        self.logger.info(f"Performing semantic search for: '{query}'")

        try:
            # Use the embedding client that was used to build the index, or create a default one
            if hasattr(self, "_embedding_client") and self._embedding_client is not None:
                embedding_client = self._embedding_client
            else:
                config = get_config()
                vector_config = config.vector_store
                # Use embedding provider system to create client
                embedding_client = get_embedding_client(
                    vector_config.embedding_model,
                    cache_embeddings=vector_config.cache_embeddings,
                )

            query_embedding = embedding_client.get_embedding(query)
            if not query_embedding:
                self.logger.error("Failed to get embedding for query")
                return []

            search_results = self.vector_index.search(query_embedding, top_k=top_k)

            if return_scores:
                # Return chunks with their similarity scores
                results_with_scores = []
                for res in search_results:
                    if res["id"] in self._chunks_by_id:
                        chunk = self._chunks_by_id[res["id"]]
                        similarity = res["similarity"]
                        results_with_scores.append((chunk, similarity))
                self.logger.info(f"Found {len(results_with_scores)} matching chunks")
                return results_with_scores
            else:
                # Return just the chunks (backward compatible)
                found_chunks = [self._chunks_by_id[res["id"]] for res in search_results if res["id"] in self._chunks_by_id]
                self.logger.info(f"Found {len(found_chunks)} matching chunks")
                return found_chunks

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    # ========== Unified Section Handling Methods ==========

    def print_for_human(
        self,
        section_names: Optional[List[str]] = None,
        max_sections: Optional[int] = None,
        show_metadata: bool = True,
        show_confidence: bool = True,
        use_colors: bool = True,
    ) -> None:
        """Print formatted document sections for human reading.

        Args:
            section_names: Specific section names to display. If None, shows all sections.
            max_sections: Maximum number of sections to display.
            show_metadata: Whether to show document metadata (title, author, etc.).
            show_confidence: Whether to show confidence scores for each section.
            use_colors: Whether to use colored output (if terminal supports it).
        """
        try:
            # Use color codes if supported and requested
            if use_colors:
                colors = {
                    "header": "\033[1;36m",  # Cyan
                    "title": "\033[1;33m",  # Yellow
                    "section": "\033[1;32m",  # Green
                    "confidence": "\033[0;36m",  # Light cyan
                    "content": "\033[0m",  # Reset
                    "metadata": "\033[0;33m",  # Light yellow
                }
                reset = "\033[0m"
            else:
                colors = {k: "" for k in ["header", "title", "section", "confidence", "content", "metadata"]}
                reset = ""

            # Show document metadata
            if show_metadata:
                print(f"{colors['header']}{'='*80}{reset}")
                print(f"{colors['title']}Document: {self.metadata.title or 'Untitled'}{reset}")
                if self.metadata.author:
                    print(f"{colors['metadata']}Author: {self.metadata.author}{reset}")
                if self.metadata.source_path:
                    print(f"{colors['metadata']}Source: {self.metadata.source_path}{reset}")
                if self.metadata.page_count:
                    print(f"{colors['metadata']}Pages: {self.metadata.page_count}{reset}")
                print(f"{colors['header']}{'='*80}{reset}")
                print()

            # Get sections to display
            if section_names:
                sections_chunks = self.get_sections_by_name(section_names)
            else:
                # Get unique sections in order
                all_section_names = self.get_section_names()
                if max_sections:
                    all_section_names = all_section_names[:max_sections]
                sections_chunks = self.get_sections_by_name(all_section_names)

            # Display each section
            for i, chunk in enumerate(sections_chunks, 1):
                section_name = chunk.chunk_name or "untitled"
                content = chunk.content
                confidence = chunk.confidence or 0.0

                print(f"{colors['section']}Section {i}: {section_name}{reset}")
                if show_confidence:
                    print(f"{colors['confidence']}Confidence: {confidence:.2f} | Length: {len(content)} chars{reset}")
                print(f"{colors['header']}{'-'*60}{reset}")
                print(f"{colors['content']}{content}{reset}")
                print()

        except Exception as e:
            self.logger.error(f"Failed to print document for human: {e}")
            print(f"Error printing document: {e}")

    def get_for_llm(
        self,
        section_names: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        include_headers: bool = True,
        clean_text: bool = True,
        personality: Optional[str] = None,
        max_chars_per_section: Optional[int] = None,
    ) -> str:
        """Get document content optimized for LLM consumption.

        Args:
            section_names: Specific section names to include. If None, includes all sections.
            max_tokens: Approximate maximum number of tokens (rough estimation: 1 token ≈ 4 chars).
            include_headers: Whether to include section headers.
            clean_text: Whether to clean and normalize text content.
            personality: Optional personality type for content tailoring (DiscussionAgent).
            max_chars_per_section: Maximum characters per section to prevent overflow.

        Returns:
            Formatted document content as string.
        """
        try:
            # Get sections to include
            if section_names:
                sections_chunks = self.get_sections_by_name(section_names)
            else:
                all_section_names = self.get_section_names()
                sections_chunks = self.get_sections_by_name(all_section_names)

            # Apply personality-based filtering if specified
            if personality:
                sections_chunks = self._filter_sections_for_personality(sections_chunks, personality)

            # Build content
            content_parts = []
            total_chars = 0
            token_limit = max_tokens * 4 if max_tokens else None

            # Add document context
            if include_headers:
                context_parts = []
                if self.metadata.title:
                    context_parts.append(f"Title: {self.metadata.title}")
                if self.metadata.author:
                    context_parts.append(f"Author: {self.metadata.author}")
                if context_parts:
                    content_parts.append("DOCUMENT METADATA:")
                    content_parts.extend(context_parts)
                    content_parts.append("")

            for chunk in sections_chunks:
                section_name = chunk.chunk_name or "untitled"
                content = chunk.content

                # Clean content if requested
                if clean_text:
                    content = self._clean_section_content(content)

                # Limit section length if specified
                if max_chars_per_section and len(content) > max_chars_per_section:
                    content = content[:max_chars_per_section] + "...[truncated]"

                # Check token limit
                section_text = f"=== {section_name.upper()} ===\n{content}\n"
                if token_limit and total_chars + len(section_text) > token_limit:
                    # Add partial section if we have space
                    remaining = token_limit - total_chars - 50  # Reserve space for header
                    if remaining > 200:  # Only add if we have meaningful space
                        partial_content = content[:remaining] + "...[truncated due to token limit]"
                        section_text = f"=== {section_name.upper()} ===\n{partial_content}\n"
                        content_parts.append(section_text)
                    break

                content_parts.append(section_text)
                total_chars += len(section_text)

            return "\n".join(content_parts)

        except Exception as e:
            self.logger.error(f"Failed to get content for LLM: {e}")
            return f"Error retrieving content: {e}"

    def get_section_by_number(self, index: int, include_content: bool = True, max_chars: Optional[int] = None) -> Optional[Tuple[str, str]]:
        """Get section by numerical index.

        Args:
            index: Zero-based index of the section.
            include_content: Whether to include section content.
            max_chars: Maximum characters to return for content.

        Returns:
            Tuple of (section_name, content) if found, None otherwise.
        """
        try:
            section_names = self.get_section_names()
            if index < 0 or index >= len(section_names):
                return None

            section_name = section_names[index]
            if not include_content:
                return (section_name, "")

            # Get chunks for this section
            section_chunks = self.get_sections_by_name([section_name])
            if not section_chunks:
                return (section_name, "")

            # Combine content from all chunks in this section
            content = "\n\n".join(chunk.content for chunk in section_chunks)

            # Apply character limit if specified
            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "...[truncated]"

            return (section_name, content)

        except Exception as e:
            self.logger.error(f"Failed to get section by number {index}: {e}")
            return None

    def get_section_by_name(
        self,
        name: str,
        fuzzy: bool = True,
        case_sensitive: bool = False,
        threshold: float = 0.8,
    ) -> Optional[Tuple[str, str]]:
        """Get section by name with optional fuzzy matching.

        Args:
            name: Section name to search for.
            fuzzy: Whether to use fuzzy matching if exact match not found.
            case_sensitive: Whether matching should be case sensitive.
            threshold: Similarity threshold for fuzzy matching (0.0-1.0).

        Returns:
            Tuple of (section_name, content) if found, None otherwise.
        """
        try:
            section_names = self.get_section_names()

            # Normalize for case sensitivity
            if not case_sensitive:
                search_name = name.lower()
                normalized_names = [n.lower() for n in section_names]
            else:
                search_name = name
                normalized_names = section_names

            # Try exact match first
            if search_name in normalized_names:
                actual_name = section_names[normalized_names.index(search_name)]
                return self.get_section_by_number(section_names.index(actual_name))

            # Try fuzzy matching if requested
            if fuzzy:
                closest_match = self.get_closest_section_name(name, case_sensitive=case_sensitive, threshold=threshold)
                if closest_match:
                    return self.get_section_by_number(section_names.index(closest_match))

            return None

        except Exception as e:
            self.logger.error(f"Failed to get section by name '{name}': {e}")
            return None

    def get_closest_section_name(
        self,
        target_name: str,
        available_names: Optional[List[str]] = None,
        case_sensitive: bool = False,
        threshold: float = 0.8,
        use_embedding: bool = False,
    ) -> Optional[str]:
        """Find closest matching section name using various similarity measures.

        Args:
            target_name: Name to search for.
            available_names: List of section names to search in. If None, uses document's section names.
            case_sensitive: Whether matching should be case sensitive.
            threshold: Minimum similarity threshold (0.0-1.0).
            use_embedding: Whether to use semantic similarity via embeddings (if available).

        Returns:
            Closest matching section name if above threshold, None otherwise.
        """
        try:
            if available_names is None:
                available_names = self.get_section_names()

            if not available_names:
                return None

            # Normalize for case sensitivity
            if not case_sensitive:
                search_name = target_name.lower()
                normalized_names = [n.lower() for n in available_names]
            else:
                search_name = target_name
                normalized_names = available_names

            best_match = None
            best_score = 0.0

            # Step 1: Try exact match first
            if search_name in normalized_names:
                return available_names[normalized_names.index(search_name)]

            # Step 2: Try pattern-based matching
            pattern_match = self._match_section_pattern(search_name, normalized_names, available_names)
            if pattern_match:
                return pattern_match

            # Step 3: Try semantic similarity first if requested and available
            if use_embedding and hasattr(self, '_embedding_client'):
                try:
                    target_embedding = self._embedding_client.get_embedding(search_name)
                    if target_embedding:
                        for i, name in enumerate(normalized_names):
                            name_embedding = self._embedding_client.get_embedding(name)
                            if name_embedding:
                                # Calculate cosine similarity
                                similarity = self._cosine_similarity(target_embedding, name_embedding)
                                if similarity > best_score and similarity >= threshold:
                                    best_score = similarity
                                    best_match = available_names[i]
                except Exception:
                    # Fall back to string similarity if embedding fails
                    pass

            # Step 4: Use enhanced string similarity with multiple strategies
            if best_match is None:
                for i, name in enumerate(normalized_names):
                    # Multiple similarity measures
                    sequence_sim = SequenceMatcher(None, search_name, name).ratio()
                    word_sim = self._word_similarity(search_name, name)
                    prefix_sim = self._prefix_similarity(search_name, name)

                    # Use the best similarity score
                    combined_sim = max(sequence_sim, word_sim, prefix_sim)

                    if combined_sim > best_score and combined_sim >= threshold:
                        best_score = combined_sim
                        best_match = available_names[i]

            return best_match

        except Exception as e:
            self.logger.error(f"Failed to find closest section name for '{target_name}': {e}")
            return None

    def _match_section_pattern(self, search_name: str, normalized_names: List[str], original_names: List[str]) -> Optional[str]:
        """Match section using common academic paper patterns."""
        # Common academic section patterns and their variations
        section_patterns = {
            # Introduction variations
            'introduction': ['intro', 'introduction', 'background', 'overview', 'prelude', 'preamble'],

            # Abstract variations
            'abstract': ['abstract', 'summary', 'executive summary', 'overview'],

            # Related work variations
            'related work': ['related work', 'background', 'literature review', 'survey', 'previous work', 'state of the art'],

            # Methodology variations
            'methodology': ['methodology', 'method', 'methods', 'approach', 'methodology and approach', 'technical approach', 'design'],

            # Experiments variations
            'experiments': ['experiment', 'experiments', 'experimental setup', 'evaluation', 'empirical evaluation', 'study design', 'case study'],

            # Results variations
            'results': ['results', 'findings', 'outcomes', 'performance', 'evaluation results', 'experimental results'],

            # Discussion variations
            'discussion': ['discussion', 'analysis', 'interpretation', 'implications'],

            # Conclusion variations
            'conclusion': ['conclusion', 'conclusions', 'summary', 'future work', 'concluding remarks'],

            # References/Bibliography variations
            'references': ['references', 'bibliography', 'citations', 'works cited', 'bibliography and references'],

            # Appendix variations
            'appendix': ['appendix', 'appendices', 'supplementary material', 'supplemental material', 'additional information'],
        }

        # Check each pattern
        for canonical_name, variations in section_patterns.items():
            if search_name in variations:
                # Now look for any of these variations in the available names
                for variation in variations:
                    if variation in normalized_names:
                        return original_names[normalized_names.index(variation)]

        return None

    def _word_similarity(self, str1: str, str2: str) -> float:
        """Calculate word-level similarity between two strings."""
        try:
            words1 = set(str1.split())
            words2 = set(str2.split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0
        except Exception:
            return 0.0

    def _prefix_similarity(self, str1: str, str2: str) -> float:
        """Calculate prefix similarity between two strings."""
        try:
            min_len = min(len(str1), len(str2))
            if min_len == 0:
                return 0.0

            common_prefix = 0
            for i in range(min_len):
                if str1[i] == str2[i]:
                    common_prefix += 1
                else:
                    break

            return common_prefix / min_len
        except Exception:
            return 0.0

    def get_section_overview(self, include_stats: bool = True, include_quality: bool = True) -> dict:
        """Get comprehensive overview of all sections.

        Args:
            include_stats: Whether to include section statistics.
            include_quality: Whether to include quality metrics.

        Returns:
            Dictionary with section information.
        """
        try:
            overview = {
                "document_title": self.metadata.title or "Untitled",
                "total_sections": 0,
                "sections": [],
                "total_chunks": len(self._chunks),
                "document_type": "markdown" if self.is_markdown else "text",
            }

            section_names = self.get_section_names()
            overview["total_sections"] = len(section_names)

            for i, section_name in enumerate(section_names):
                section_chunks = self.get_sections_by_name([section_name])
                total_chars = sum(len(chunk.content) for chunk in section_chunks)
                avg_confidence = sum(chunk.confidence or 0.0 for chunk in section_chunks) / len(section_chunks) if section_chunks else 0.0

                section_info = {
                    "index": i,
                    "name": section_name,
                    "chunk_count": len(section_chunks),
                    "character_count": total_chars,
                }

                if include_quality:
                    section_info.update({
                        "average_confidence": avg_confidence,
                        "high_quality_chunks": sum(1 for chunk in section_chunks if (chunk.confidence or 0.0) >= 0.7),
                        "processed_chunks": sum(1 for chunk in section_chunks if chunk.processed),
                    })

                if include_stats:
                    # Add additional statistics if requested
                    chunk_lengths = [len(chunk.content) for chunk in section_chunks]
                    section_info.update({
                        "min_chunk_size": min(chunk_lengths) if chunk_lengths else 0,
                        "max_chunk_size": max(chunk_lengths) if chunk_lengths else 0,
                        "avg_chunk_size": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                    })

                overview["sections"].append(section_info)

            return overview

        except Exception as e:
            self.logger.error(f"Failed to get section overview: {e}")
            return {"error": str(e)}

    def get_sections_with_confidence(self, min_confidence: float = 0.5, min_length: int = 50) -> List[Tuple[str, str]]:
        """Get sections that meet minimum quality criteria.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0).
            min_length: Minimum character length per section.

        Returns:
            List of (section_name, content) tuples that meet criteria.
        """
        try:
            sections = []
            section_names = self.get_section_names()

            for section_name in section_names:
                section_chunks = self.get_sections_by_name([section_name])
                if not section_chunks:
                    continue

                # Calculate combined confidence and length for the section
                total_confidence = sum(chunk.confidence or 0.0 for chunk in section_chunks)
                avg_confidence = total_confidence / len(section_chunks)
                total_length = sum(len(chunk.content) for chunk in section_chunks)

                if avg_confidence >= min_confidence and total_length >= min_length:
                    content = "\n\n".join(chunk.content for chunk in section_chunks)
                    sections.append((section_name, content))

            return sections

        except Exception as e:
            self.logger.error(f"Failed to get sections with confidence: {e}")
            return []

    # ========== Helper Methods ==========

    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content."""
        try:
            # Remove excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

            # Fix common PDF extraction artifacts
            content = re.sub(r'\b(\w+)-\s*\n\s*(\w+)\b', r'\1\2', content)  # Fix hyphenated words

            # Normalize quotes (using string replacement instead of regex)
            content = content.replace('"', '"').replace('"', '"')
            content = content.replace("'", "'").replace("'", "'")

            # Remove excessive spaces
            content = re.sub(r' +', ' ', content)

            return content.strip()

        except Exception as e:
            self.logger.warning(f"Failed to clean section content: {e}")
            return content

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)

        except Exception:
            return 0.0

    def _filter_sections_for_personality(self, sections_chunks: List[Chunk], personality: str) -> List[Chunk]:
        """Filter sections based on personality preferences (DiscussionAgent)."""
        personality_preferences = {
            "critical_evaluator": ["methodology", "experiments", "results", "evaluation", "limitations", "analysis"],
            "innovative_insighter": ["introduction", "contributions", "novelty", "innovation", "future", "conclusion"],
            "practical_applicator": ["experiments", "applications", "implementation", "case study", "deployment", "results"],
            "theoretical_integrator": ["background", "related work", "theory", "analysis", "discussion", "conclusion"],
        }

        if personality.lower() not in personality_preferences:
            return sections_chunks

        preferred_keywords = personality_preferences[personality.lower()]
        filtered_chunks = []

        for chunk in sections_chunks:
            section_name = (chunk.chunk_name or "").lower()
            if any(keyword in section_name for keyword in preferred_keywords):
                filtered_chunks.append(chunk)

        # If no sections match preferences, return original sections
        return filtered_chunks if filtered_chunks else sections_chunks

    # ========== Agent-Specific Optimization Methods ==========

    def get_sections_for_personality(
        self,
        personality_type: str,
        max_sections: int = 5,
        max_chars_per_section: int = 3000,
        include_fallback: bool = True,
    ) -> List[Tuple[str, str]]:
        """Get sections optimized for specific DiscussionAgent personality type.

        Args:
            personality_type: Type of personality (critical_evaluator, innovative_insighter, etc.).
            max_sections: Maximum number of sections to return.
            max_chars_per_section: Maximum characters per section.
            include_fallback: Whether to include fallback sections if preferred ones aren't found.

        Returns:
            List of (section_name, content) tuples optimized for the personality.
        """
        try:
            # Get all sections
            section_names = self.get_section_names()
            if not section_names:
                return []

            # Filter sections based on personality preferences
            sections_chunks = self.get_sections_by_name(section_names)
            filtered_chunks = self._filter_sections_for_personality(sections_chunks, personality_type)

            # If no sections match and fallback is enabled, return most important sections
            if not filtered_chunks and include_fallback:
                # Priority order: abstract, introduction, methodology, results, conclusion
                priority_sections = ["abstract", "introduction", "methodology", "results", "conclusion"]
                for priority in priority_sections:
                    matching_chunks = self.get_sections_by_name([priority])
                    if matching_chunks:
                        filtered_chunks.extend(matching_chunks)
                        break

            # Limit to max_sections
            filtered_chunks = filtered_chunks[:max_sections]

            # Convert to (name, content) tuples with length limits
            result = []
            for chunk in filtered_chunks:
                section_name = chunk.chunk_name or "untitled"
                content = chunk.content

                if max_chars_per_section and len(content) > max_chars_per_section:
                    content = content[:max_chars_per_section] + "...[truncated]"

                result.append((section_name, content))

            return result

        except Exception as e:
            self.logger.error(f"Failed to get sections for personality '{personality_type}': {e}")
            return []

    def get_for_simple_agent(
        self,
        include_metadata: bool = True,
        clean_references: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get document content optimized for SimpleAgent.

        Args:
            include_metadata: Whether to include document metadata.
            clean_references: Whether to remove references section.
            max_tokens: Maximum number of tokens to include.

        Returns:
            Formatted content for SimpleAgent processing.
        """
        try:
            # Use full text approach for SimpleAgent (maintain simplicity)
            content_parts = []

            if include_metadata:
                metadata_parts = []
                if self.metadata.title:
                    metadata_parts.append(f"Title: {self.metadata.title}")
                if self.metadata.author:
                    metadata_parts.append(f"Author: {self.metadata.author}")
                if metadata_parts:
                    content_parts.append("DOCUMENT INFORMATION:")
                    content_parts.extend(metadata_parts)
                    content_parts.append("")

            # Get full text (SimpleAgent's approach)
            if self._chunks:
                full_text = self.get_full_text()
            else:
                full_text = self._raw_text

            # Clean references if requested
            if clean_references:
                full_text = self._remove_references_section(full_text)

            # Clean academic text artifacts
            full_text = self._clean_section_content(full_text)

            # Apply token limit if specified
            if max_tokens and len(full_text) > max_tokens * 4:
                full_text = full_text[:max_tokens * 4] + "...[truncated due to token limit]"

            content_parts.append(full_text)
            return "\n\n".join(content_parts)

        except Exception as e:
            self.logger.error(f"Failed to get content for SimpleAgent: {e}")
            return f"Error retrieving content: {e}"

    def get_for_coordinate_agent(
        self,
        expert_type: str,
        planned_sections: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get document content optimized for CoordinateAgent expert type.

        Args:
            expert_type: Type of expert (metadata, methodology, experiments, evaluation, etc.).
            planned_sections: Pre-planned sections for this expert type.
            max_tokens: Maximum number of tokens to include.

        Returns:
            Formatted content optimized for specific expert analysis.
        """
        try:
            # Expert-type section preferences
            expert_sections = {
                "metadata": ["abstract", "introduction", "title"],
                "methodology": ["methodology", "method", "approach", "design", "technical approach"],
                "experiments": ["experiments", "experimental setup", "evaluation", "study design", "case study"],
                "evaluation": ["results", "evaluation", "findings", "outcomes", "performance"],
                "contributions": ["introduction", "contributions", "novelty", "innovation"],
                "limitations": ["limitations", "discussion", "conclusion", "future work"],
            }

            # Determine which sections to include
            if planned_sections:
                target_sections = planned_sections
            elif expert_type in expert_sections:
                target_sections = expert_sections[expert_type]
            else:
                # Default to key sections
                target_sections = ["abstract", "introduction", "methodology", "results", "conclusion"]

            # Find matching sections
            matched_sections = []
            for target in target_sections:
                match = self.get_closest_section_name(target, threshold=0.7)
                if match and match not in matched_sections:
                    matched_sections.append(match)

            if not matched_sections:
                # Fallback to first few sections
                all_sections = self.get_section_names()
                matched_sections = all_sections[:3]

            # Get formatted content for matched sections
            return self.get_for_llm(
                section_names=matched_sections,
                max_tokens=max_tokens,
                include_headers=True,
                clean_text=True,
                max_chars_per_section=2500,  # Slightly larger for expert analysis
            )

        except Exception as e:
            self.logger.error(f"Failed to get content for CoordinateAgent expert '{expert_type}': {e}")
            return f"Error retrieving content for {expert_type}: {e}"

    def get_for_react_agent(
        self,
        current_report: str = "",
        processed_sections: Optional[List[str]] = None,
        next_section_hint: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get document content optimized for ReActAgent iterative exploration.

        Args:
            current_report: Current analysis report to provide context.
            processed_sections: List of already processed section names.
            next_section_hint: Hint for which section to explore next.
            max_tokens: Maximum number of tokens for the content.

        Returns:
            Formatted content for ReActAgent next iteration.
        """
        try:
            content_parts = []

            # Add current report context
            if current_report:
                content_parts.append("CURRENT ANALYSIS:")
                content_parts.append(current_report)
                content_parts.append("")

            # Determine which sections haven't been processed yet
            all_sections = self.get_section_names()
            unprocessed = []
            if processed_sections:
                for section in all_sections:
                    if section not in processed_sections:
                        unprocessed.append(section)
            else:
                unprocessed = all_sections

            # Select next section(s) to explore
            if next_section_hint:
                # Try to find section matching the hint
                hinted_match = self.get_closest_section_name(next_section_hint, threshold=0.7)
                if hinted_match and hinted_match in unprocessed:
                    next_sections = [hinted_match]
                else:
                    next_sections = unprocessed[:1] if unprocessed else []
            else:
                # Choose next unprocessed section
                next_sections = unprocessed[:1] if unprocessed else []

            if not next_sections:
                content_parts.append("No more unprocessed sections available.")
                return "\n\n".join(content_parts)

            # Add context about remaining sections
            content_parts.append(f"NEXT SECTION(S) TO ANALYZE:")
            content_parts.append(f"Processed: {len(processed_sections) if processed_sections else 0}")
            content_parts.append(f"Remaining: {len(unprocessed)}")
            content_parts.append("")

            # Get content for next sections
            section_content = self.get_for_llm(
                section_names=next_sections,
                max_tokens=max_tokens,
                include_headers=True,
                clean_text=True,
                max_chars_per_section=2000,  # Smaller for iterative processing
            )

            content_parts.append(section_content)
            return "\n\n".join(content_parts)

        except Exception as e:
            self.logger.error(f"Failed to get content for ReActAgent: {e}")
            return f"Error retrieving content: {e}"

    def _remove_references_section(self, text: str) -> str:
        """Remove references and bibliography sections from text."""
        try:
            # Common patterns for references sections
            ref_patterns = [
                r'\n\s*(?:references|bibliography|citations|works\s+cited)\s*\n',
                r'\n\s*references\s*$',
                r'\n\s*bibliography\s*$',
                r'\n\s*citations\s*$',
            ]

            # Find the earliest occurrence of a references section
            earliest_pos = len(text)
            for pattern in ref_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                if matches:
                    earliest_pos = min(earliest_pos, matches[0].start())

            # If found, truncate text before references
            if earliest_pos < len(text):
                return text[:earliest_pos].strip()

            return text

        except Exception:
            return text

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
            doc = cls(
                text=doc_state["text"],
                metadata=metadata,
                _is_markdown=doc_state.get("is_markdown", False),
            )

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
