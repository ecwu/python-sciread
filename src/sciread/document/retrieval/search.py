"""Unified multi-strategy retrieval helpers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from sciread.document.retrieval.models import RetrievedChunk
from sciread.document.state import get_chunk_map
from sciread.document.structure.tree import build_section_tree
from sciread.document.structure.tree import iter_descendant_chunks
from sciread.document.structure.tree import normalize_section_path

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.models import Chunk


QUERY_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_.-]+")
SUPPORTED_RETRIEVERS = ("lexical", "semantic", "tree", "hybrid")


def retrieve_chunks(
    document: Document,
    query: str,
    *,
    strategy: str = "hybrid",
    top_k: int = 5,
    neighbor_window: int = 1,
    section_scope: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve chunks with one of the supported strategies."""
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in SUPPORTED_RETRIEVERS:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    if normalized_strategy == "lexical":
        return lexical_search(
            document,
            query,
            top_k=top_k,
            neighbor_window=neighbor_window,
            section_scope=section_scope,
        )
    if normalized_strategy == "semantic":
        return semantic_chunk_search(
            document,
            query,
            top_k=top_k,
            neighbor_window=neighbor_window,
            section_scope=section_scope,
        )
    if normalized_strategy == "tree":
        return tree_search(
            document,
            query,
            top_k=top_k,
            neighbor_window=neighbor_window,
            section_scope=section_scope,
        )
    return hybrid_search(
        document,
        query,
        top_k=top_k,
        neighbor_window=neighbor_window,
        section_scope=section_scope,
    )


def lexical_search(
    document: Document,
    query: str,
    *,
    top_k: int,
    neighbor_window: int,
    section_scope: str | None,
    include_context: bool = True,
) -> list[RetrievedChunk]:
    """Perform heuristic lexical retrieval over chunk text and section metadata."""
    query_text = query.strip().lower()
    tokens = _tokenize_query(query)
    candidates = _filter_chunks_by_scope(document, section_scope)
    scored_results: list[RetrievedChunk] = []

    for chunk in candidates:
        search_text = f"{chunk.retrieval_text}\n{chunk.content_plain}".lower()
        section_label = " > ".join(chunk.section_path).lower()
        section_name = (chunk.chunk_name or "").lower()
        matched_terms: list[str] = []
        score = 0.0

        if query_text and query_text in search_text:
            score += 8.0
            matched_terms.append(query_text)

        for token in tokens:
            if token in search_text:
                score += 2.0
                matched_terms.append(token)
            if token in section_label:
                score += 3.0
            if token == section_name or token == section_label:
                score += 4.0

        if section_scope and _chunk_in_scope(chunk, section_scope):
            score += 2.0

        score += _position_bonus(chunk.position)
        if score <= 0:
            continue

        scored_results.append(
            RetrievedChunk(
                chunk=chunk,
                score=score,
                strategy="lexical",
                matched_terms=sorted(set(matched_terms)),
                section_path=chunk.section_path.copy(),
            )
        )

    return _finalize_results(document, scored_results, top_k=top_k, neighbor_window=neighbor_window, include_context=include_context)


def semantic_chunk_search(
    document: Document,
    query: str,
    *,
    top_k: int,
    neighbor_window: int,
    section_scope: str | None,
    include_context: bool = True,
) -> list[RetrievedChunk]:
    """Perform semantic chunk retrieval, lazily building the vector index."""
    try:
        if document.vector_index is None:
            document.build_vector_index()
    except Exception as exc:
        raise RuntimeError(f"Semantic retrieval unavailable: failed to build vector index ({exc})") from exc

    try:
        raw_results = document.semantic_search(query, top_k=max(top_k * 2, top_k), return_scores=True)
    except Exception as exc:
        raise RuntimeError(f"Semantic retrieval failed: {exc}") from exc

    if not raw_results:
        return []

    normalized_results: list[RetrievedChunk] = []
    for chunk, similarity in raw_results:
        if section_scope and not _chunk_in_scope(chunk, section_scope):
            continue
        normalized_results.append(
            RetrievedChunk(
                chunk=chunk,
                score=float(similarity),
                strategy="semantic",
                matched_terms=[],
                section_path=chunk.section_path.copy(),
            )
        )

    if not normalized_results:
        return []
    return _finalize_results(
        document,
        normalized_results,
        top_k=top_k,
        neighbor_window=neighbor_window,
        include_context=include_context,
    )


def tree_search(
    document: Document,
    query: str,
    *,
    top_k: int,
    neighbor_window: int,
    section_scope: str | None,
    include_context: bool = True,
) -> list[RetrievedChunk]:
    """Search the runtime section tree before selecting descendant chunks."""
    section_tree = build_section_tree(document)
    scope_prefix = normalize_section_path(section_scope) if section_scope else None
    query_text = query.strip().lower()
    tokens = _tokenize_query(query)
    chunk_map = get_chunk_map(document)
    chunk_scores: dict[str, float] = defaultdict(float)
    matched_terms_map: dict[str, set[str]] = defaultdict(set)

    for node in section_tree.nodes_by_path.values():
        node_path_text = node.path_text.lower()
        if scope_prefix and not (node_path_text == scope_prefix or node_path_text.startswith(f"{scope_prefix} >")):
            continue

        node_score = _score_tree_node(query_text, tokens, node_path_text)
        if node_score <= 0:
            continue

        for chunk in iter_descendant_chunks(node, chunk_map):
            chunk_scores[chunk.chunk_id] = max(chunk_scores[chunk.chunk_id], node_score + _position_bonus(chunk.position))
            for token in tokens:
                if token and token in node_path_text:
                    matched_terms_map[chunk.chunk_id].add(token)

    results: list[RetrievedChunk] = []
    for chunk_id, score in chunk_scores.items():
        chunk = chunk_map[chunk_id]
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=score,
                strategy="tree",
                matched_terms=sorted(matched_terms_map[chunk_id]),
                section_path=chunk.section_path.copy(),
            )
        )

    return _finalize_results(document, results, top_k=top_k, neighbor_window=neighbor_window, include_context=include_context)


def hybrid_search(
    document: Document,
    query: str,
    *,
    top_k: int,
    neighbor_window: int,
    section_scope: str | None,
    include_context: bool = True,
) -> list[RetrievedChunk]:
    """Combine lexical, semantic, and tree retrieval with reciprocal-rank fusion."""
    results_by_strategy: dict[str, list[RetrievedChunk]] = {
        "lexical": lexical_search(
            document,
            query,
            top_k=max(top_k * 2, top_k),
            neighbor_window=neighbor_window,
            section_scope=section_scope,
            include_context=False,
        ),
        "tree": tree_search(
            document,
            query,
            top_k=max(top_k * 2, top_k),
            neighbor_window=neighbor_window,
            section_scope=section_scope,
            include_context=False,
        ),
    }
    results_by_strategy["semantic"] = semantic_chunk_search(
        document,
        query,
        top_k=max(top_k * 2, top_k),
        neighbor_window=neighbor_window,
        section_scope=section_scope,
        include_context=False,
    )

    fused_scores: dict[str, float] = defaultdict(float)
    result_payloads: dict[str, RetrievedChunk] = {}

    for _strategy_name, results in results_by_strategy.items():
        for rank, result in enumerate(results, start=1):
            fused_scores[result.chunk.chunk_id] += 1.0 / (50 + rank)
            existing = result_payloads.get(result.chunk.chunk_id)
            if existing is None:
                result_payloads[result.chunk.chunk_id] = RetrievedChunk(
                    chunk=result.chunk,
                    score=result.score,
                    strategy="hybrid",
                    matched_terms=result.matched_terms.copy(),
                    section_path=result.section_path.copy(),
                )
            else:
                existing.matched_terms = sorted(set(existing.matched_terms).union(result.matched_terms))

    fused_results: list[RetrievedChunk] = []
    for chunk_id, fused_score in fused_scores.items():
        payload = result_payloads[chunk_id]
        payload.score = fused_score
        fused_results.append(payload)

    return _finalize_results(
        document,
        fused_results,
        top_k=top_k,
        neighbor_window=neighbor_window,
        include_context=include_context,
    )


def format_retrieval_results(results: list[RetrievedChunk], query: str, strategy: str) -> str:
    """Render retrieved chunks into a compact text bundle for tool outputs."""
    if not results:
        return f"No retrieval results found for query '{query}' using strategy '{strategy}'."

    lines = [f"Retrieval strategy: {strategy}", f"Query: {query}", ""]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"[{index}] section={result.section_path_text or 'unknown'} | citation={result.chunk.citation_key} | score={result.score:.3f}"
        )
        if result.matched_terms:
            lines.append(f"matched_terms={', '.join(result.matched_terms)}")
        lines.append(result.expanded_context.strip())
        lines.append("")

    return "\n".join(lines).strip()


def _filter_chunks_by_scope(document: Document, section_scope: str | None) -> list[Chunk]:
    """Return the relevant chunk set after optional section scoping."""
    if not section_scope:
        return [chunk for chunk in document.chunks if chunk.retrievable]
    return [chunk for chunk in document.get_chunks_by_section(section_scope) if chunk.retrievable]


def _chunk_in_scope(chunk: Chunk, section_scope: str) -> bool:
    """Return whether a chunk belongs to the requested scope."""
    scope = normalize_section_path(section_scope)
    chunk_path = " > ".join(chunk.section_path).lower()
    return chunk_path == scope or chunk_path.startswith(f"{scope} >")


def _tokenize_query(query: str) -> list[str]:
    """Tokenize a query into lowercase lexical terms."""
    return [match.group(0).lower() for match in QUERY_TOKEN_PATTERN.finditer(query)]


def _position_bonus(position: int) -> float:
    """Prefer earlier chunks slightly without dominating other scores."""
    return 1.0 / math.sqrt(position + 1)


def _build_expanded_context(document: Document, chunk: Chunk, neighbor_window: int) -> str:
    """Build a context window around the retrieved chunk."""
    neighbors = document.get_neighbor_chunks(
        chunk.chunk_id,
        before=max(0, neighbor_window),
        after=max(0, neighbor_window),
        include_self=True,
    )
    if not neighbors:
        neighbors = [chunk]

    parts: list[str] = []
    for neighbor in neighbors:
        section_label = " > ".join(neighbor.section_path) if neighbor.section_path else (neighbor.chunk_name or "unknown")
        parts.append(f"[{neighbor.citation_key}] {section_label}\n{(neighbor.content_plain or neighbor.content).strip()}")
    return "\n\n".join(parts)


def _score_tree_node(query_text: str, tokens: list[str], node_path_text: str) -> float:
    """Score one tree node against the query."""
    score = 0.0
    if query_text and query_text in node_path_text:
        score += 8.0
    for token in tokens:
        if token in node_path_text:
            score += 3.0
    return score


def _limit_results(results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Sort, deduplicate, and trim retrieval results."""
    deduplicated: dict[str, RetrievedChunk] = {}
    for result in sorted(results, key=lambda item: (-item.score, item.chunk.position)):
        existing = deduplicated.get(result.chunk.chunk_id)
        if existing is None or result.score > existing.score:
            deduplicated[result.chunk.chunk_id] = result
    return list(deduplicated.values())[:top_k]


def _finalize_results(
    document: Document,
    results: list[RetrievedChunk],
    *,
    top_k: int,
    neighbor_window: int,
    include_context: bool,
) -> list[RetrievedChunk]:
    """Trim results first, then build expanded context only for the final payload."""
    limited_results = _limit_results(results, top_k)
    if not include_context:
        return limited_results

    for result in limited_results:
        result.expanded_context = _build_expanded_context(document, result.chunk, neighbor_window)
    return limited_results
