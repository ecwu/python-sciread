"""Retrieval operations for Document semantic indexing/search."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sciread.document.retrieval.vector_index import VectorIndex
from sciread.document.state import get_chunk_map
from sciread.document.state import get_runtime_embedding_client
from sciread.document.state import set_runtime_embedding_client
from sciread.embedding_provider import get_embedding_client
from sciread.platform.config import get_config

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.models import Chunk


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
    except Exception:
        return 0.0


def build_vector_index(
    document: Document,
    persist: bool = False,
    embedding_client=None,
    get_config_fn=get_config,
    get_embedding_client_fn=get_embedding_client,
    vector_index_cls=VectorIndex,
) -> None:
    """Build a semantic vector index from document chunks."""
    chunks = document.chunks
    if not chunks:
        document.logger.warning("No chunks to index. Please split the document first.")
        return

    document.logger.info(f"Building vector index from {len(chunks)} chunks...")

    try:
        vector_config = None
        if embedding_client is None:
            config = get_config_fn()
            vector_config = config.vector_store
            embedding_client = get_embedding_client_fn(
                vector_config.embedding_model,
                cache_embeddings=vector_config.cache_embeddings,
            )

        set_runtime_embedding_client(document, embedding_client)

        batch_size = 10
        if hasattr(embedding_client, "embedding_batch_size"):
            batch_size = embedding_client.embedding_batch_size

        embeddings = embedding_client.get_embeddings(
            [chunk.retrieval_text or chunk.content for chunk in chunks],
            batch_size=batch_size,
        )

        persist_path = None
        if persist:
            if vector_config is None:
                config = get_config_fn()
                vector_config = config.vector_store
            store_path = Path(vector_config.path).expanduser()
            store_path.mkdir(parents=True, exist_ok=True)
            persist_path = store_path / document._build_doc_id()

        collection_name = document._build_doc_id()
        document.vector_index = vector_index_cls(
            collection_name=collection_name,
            persist_path=persist_path,
        )
        document.vector_index.add_chunks(chunks, embeddings)
        document.logger.info("Vector index built successfully.")

    except Exception as e:
        document.logger.error(f"Failed to build vector index: {e}")
        raise RuntimeError(f"Failed to build vector index: {e}") from e


def semantic_search(
    document: Document,
    query: str,
    top_k: int = 5,
    return_scores: bool = False,
    get_config_fn=get_config,
    get_embedding_client_fn=get_embedding_client,
) -> list[Chunk] | list[tuple[Chunk, float]]:
    """Perform semantic search on document chunks."""
    if not document.vector_index:
        document.logger.warning("Vector index not found. Please run `build_vector_index()` first.")
        return []

    document.logger.info(f"Performing semantic search for: '{query}'")

    try:
        embedding_client = get_runtime_embedding_client(document)
        if embedding_client is not None:
            active_embedding_client = embedding_client
        else:
            config = get_config_fn()
            vector_config = config.vector_store
            active_embedding_client = get_embedding_client_fn(
                vector_config.embedding_model,
                cache_embeddings=vector_config.cache_embeddings,
            )
            set_runtime_embedding_client(document, active_embedding_client)

        query_embedding = active_embedding_client.get_embedding(query)
        if not query_embedding:
            document.logger.error("Failed to get embedding for query")
            return []

        search_results = document.vector_index.search(query_embedding, top_k=top_k)
        chunk_map = get_chunk_map(document)

        if return_scores:
            results_with_scores = []
            for res in search_results:
                if res["id"] in chunk_map:
                    chunk = chunk_map[res["id"]]
                    similarity = res["similarity"]
                    results_with_scores.append((chunk, similarity))
            document.logger.info(f"Found {len(results_with_scores)} matching chunks")
            return results_with_scores

        found_chunks = [chunk_map[res["id"]] for res in search_results if res["id"] in chunk_map]
        document.logger.info(f"Found {len(found_chunks)} matching chunks")
        return found_chunks

    except Exception as e:
        document.logger.error(f"Semantic search failed: {e}")
        return []
