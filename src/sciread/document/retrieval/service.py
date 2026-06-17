"""Retrieval operations for Document semantic indexing/search."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sciread.document.retrieval.vector_index import VectorIndex
from sciread.document.state import get_chunk_map
from sciread.document.state import get_runtime_embedding_client
from sciread.document.state import get_runtime_rerank_client
from sciread.document.state import set_runtime_embedding_client
from sciread.document.state import set_runtime_rerank_client
from sciread.embedding_provider import get_embedding_client
from sciread.embedding_provider.base import cosine_similarity as compute_cosine_similarity
from sciread.platform.config import get_config
from sciread.rerank_provider import get_rerank_client

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.models import Chunk


cosine_similarity = compute_cosine_similarity


def _resolve_batch_size(embedding_client, default: int = 10) -> int:
    """Read an optional provider batch size without trusting arbitrary attributes."""
    batch_size = getattr(embedding_client, "embedding_batch_size", None)
    if isinstance(batch_size, int) and batch_size > 0:
        return batch_size
    return default


def _resolve_runtime_embedding_client(
    document: Document,
    *,
    explicit_client=None,
    get_config_fn=get_config,
    get_embedding_client_fn=get_embedding_client,
):
    """Return the active runtime embedding client, creating and caching it when needed."""
    if explicit_client is not None:
        set_runtime_embedding_client(document, explicit_client)
        return explicit_client

    runtime_client = get_runtime_embedding_client(document)
    if runtime_client is not None:
        return runtime_client

    config = get_config_fn()
    vector_config = config.vector_store
    runtime_client = get_embedding_client_fn(
        vector_config.embedding_model,
        cache_embeddings=vector_config.cache_embeddings,
    )
    set_runtime_embedding_client(document, runtime_client)
    return runtime_client


def _resolve_runtime_rerank_client(
    document: Document,
    *,
    explicit_client=None,
    get_config_fn=get_config,
    get_rerank_client_fn=get_rerank_client,
):
    """Return the active runtime rerank client, creating and caching it when needed."""
    if explicit_client is not None:
        set_runtime_rerank_client(document, explicit_client)
        return explicit_client

    runtime_client = get_runtime_rerank_client(document)
    if runtime_client is not None:
        return runtime_client

    config = get_config_fn()
    vector_config = config.vector_store
    runtime_client = get_rerank_client_fn(vector_config.rerank_model)
    set_runtime_rerank_client(document, runtime_client)
    return runtime_client


def build_vector_index(
    document: Document,
    persist: bool = False,
    embedding_client=None,
    get_config_fn=get_config,
    get_embedding_client_fn=get_embedding_client,
    vector_index_cls=VectorIndex,
) -> None:
    """Build a semantic vector index from document chunks."""
    chunks = [chunk for chunk in document.chunks if chunk.retrievable]
    if not chunks:
        document.logger.warning("No retrievable chunks to index. Please split the document first.")
        return

    document.logger.info(f"Building vector index from {len(chunks)} chunks...")

    try:
        vector_config = None
        if embedding_client is None:
            config = get_config_fn()
            vector_config = config.vector_store
        embedding_client = _resolve_runtime_embedding_client(
            document,
            explicit_client=embedding_client,
            get_config_fn=get_config_fn,
            get_embedding_client_fn=get_embedding_client_fn,
        )

        embeddings = embedding_client.get_embeddings(
            [chunk.retrieval_text or chunk.content for chunk in chunks],
            batch_size=_resolve_batch_size(embedding_client),
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
            reset_collection=True,
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

    normalized_query = query.strip()
    if not normalized_query:
        return []

    document.logger.info(f"Performing semantic search for: '{query}'")

    try:
        active_embedding_client = _resolve_runtime_embedding_client(
            document,
            get_config_fn=get_config_fn,
            get_embedding_client_fn=get_embedding_client_fn,
        )

        query_embedding = active_embedding_client.get_embedding(normalized_query)
        if not query_embedding:
            document.logger.error("Failed to get embedding for query")
            return []

        search_results = document.vector_index.search(query_embedding, top_k=top_k)
        chunk_map = get_chunk_map(document)

        if return_scores:
            results_with_scores = []
            for res in search_results:
                chunk = chunk_map.get(res["id"])
                if chunk is None or not chunk.retrievable:
                    continue
                similarity = res["similarity"]
                results_with_scores.append((chunk, similarity))
            document.logger.info(f"Found {len(results_with_scores)} matching chunks")
            return results_with_scores

        found_chunks = [chunk for res in search_results if (chunk := chunk_map.get(res["id"])) is not None and chunk.retrievable]
        document.logger.info(f"Found {len(found_chunks)} matching chunks")
        return found_chunks

    except Exception as e:
        document.logger.error(f"Semantic search failed: {e}")
        return []


def rerank_search(
    document: Document,
    query: str,
    top_k: int = 5,
    candidate_top_k: int | None = None,
    return_scores: bool = False,
    rerank_client=None,
    get_config_fn=get_config,
    get_embedding_client_fn=get_embedding_client,
    get_rerank_client_fn=get_rerank_client,
) -> list[Chunk] | list[tuple[Chunk, float]]:
    """Rerank semantic search candidates with a rerank provider."""
    normalized_query = query.strip()
    if top_k <= 0 or not normalized_query:
        return []

    try:
        config = get_config_fn()
        candidate_multiplier = max(1, getattr(config.vector_store, "rerank_candidate_multiplier", 4))
        effective_candidate_top_k = candidate_top_k or max(top_k * candidate_multiplier, top_k)

        semantic_candidates = semantic_search(
            document,
            normalized_query,
            top_k=effective_candidate_top_k,
            return_scores=True,
            get_config_fn=lambda: config,
            get_embedding_client_fn=get_embedding_client_fn,
        )
        if not semantic_candidates:
            return []

        candidate_chunks = [chunk for chunk, _score in semantic_candidates]
        candidate_documents = [chunk.retrieval_text or chunk.content_plain or chunk.content for chunk in candidate_chunks]
        active_rerank_client = _resolve_runtime_rerank_client(
            document,
            explicit_client=rerank_client,
            get_config_fn=lambda: config,
            get_rerank_client_fn=get_rerank_client_fn,
        )
        reranked_results = active_rerank_client.rerank(normalized_query, candidate_documents, top_n=top_k)
        if not reranked_results:
            fallback_results = semantic_candidates[:top_k]
            return fallback_results if return_scores else [chunk for chunk, _score in fallback_results]

        results_with_scores = []
        seen_chunk_ids = set()
        for result in reranked_results:
            chunk = candidate_chunks[result.index]
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            results_with_scores.append((chunk, result.relevance_score))
            if len(results_with_scores) >= top_k:
                break

        if return_scores:
            return results_with_scores
        return [chunk for chunk, _score in results_with_scores]

    except Exception as e:
        document.logger.error(f"Rerank search failed: {e}")
        return []
