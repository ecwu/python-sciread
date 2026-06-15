"""Evidence-level retrieval built on top of chunk retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sciread.document.models import Chunk
from sciread.document.retrieval.models import Evidence

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.retrieval.models import RetrievedChunk


class EvidenceRetriever:
    """Return citation-ready evidence blocks instead of raw chunk results."""

    def __init__(
        self,
        document: Document,
        *,
        strategy: str = "semantic",
        neighbor_window: int = 1,
        max_context_tokens: int = 1000,
    ) -> None:
        self.document = document
        self.strategy = strategy
        self.neighbor_window = max(0, neighbor_window)
        self.max_context_tokens = max(1, max_context_tokens)

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        expand_context: bool = True,
        section_filter: list[str] | None = None,
    ) -> list[Evidence]:
        """Retrieve agent-facing evidence blocks for a query."""
        if not query.strip() or top_k <= 0:
            return []

        raw_top_k = top_k if not section_filter else max(top_k * 3, top_k)
        retrieved_chunks = self.document.retrieve_chunks(
            query=query,
            strategy=self.strategy,
            top_k=raw_top_k,
            neighbor_window=0,
            section_scope=None,
        )

        filtered_results = [
            result
            for result in retrieved_chunks
            if not section_filter or _chunk_matches_section_filter(result.chunk, section_filter)
        ][:top_k]

        return [
            self._build_evidence(
                result,
                rank=rank,
                expand_context=expand_context,
            )
            for rank, result in enumerate(filtered_results, start=1)
        ]

    def _build_evidence(self, result: RetrievedChunk, *, rank: int, expand_context: bool) -> Evidence:
        """Convert one retrieved chunk into a citation-ready evidence block."""
        chunk = result.chunk
        evidence_chunks = [chunk]

        if expand_context:
            evidence_chunks = self._expand_neighbors(chunk)

        text, included_chunks = _render_chunks_with_token_budget(evidence_chunks, chunk.chunk_id, self.max_context_tokens)
        display_text = text if expand_context else (chunk.display_text or chunk.content)
        expanded_from = [item.chunk_id for item in included_chunks]
        if expanded_from == [chunk.chunk_id]:
            expanded_from_value = None
        else:
            expanded_from_value = expanded_from

        return Evidence(
            evidence_id=f"E{rank}",
            chunk_id=chunk.chunk_id,
            citation_key=chunk.citation_key,
            section_path=chunk.section_path.copy(),
            section_label=_section_label(chunk),
            text=text,
            display_text=display_text,
            score=result.score,
            rank=rank,
            page_range=_merge_page_ranges(included_chunks),
            expanded_from=expanded_from_value,
        )

    def _expand_neighbors(self, chunk: Chunk) -> list[Chunk]:
        """Expand around a hit using same-section neighbors only."""
        chunks = [chunk]
        current = chunk
        for _ in range(self.neighbor_window):
            if not current.prev_chunk_id:
                break
            previous = self.document.get_chunk_by_id(current.prev_chunk_id)
            if previous is None or previous.section_path != chunk.section_path:
                break
            chunks.insert(0, previous)
            current = previous

        current = chunk
        for _ in range(self.neighbor_window):
            if not current.next_chunk_id:
                break
            next_chunk = self.document.get_chunk_by_id(current.next_chunk_id)
            if next_chunk is None or next_chunk.section_path != chunk.section_path:
                break
            chunks.append(next_chunk)
            current = next_chunk

        return chunks


def format_evidence_results(results: list[Evidence], query: str, strategy: str) -> str:
    """Render evidence results without exposing chunk internals."""
    if not results:
        return f"No evidence found for query '{query}' using strategy '{strategy}'."

    lines = [f"Evidence strategy: {strategy}", f"Query: {query}", ""]
    for evidence in results:
        lines.append(
            f"[{evidence.rank}] id={evidence.evidence_id} | section={evidence.section_label or 'unknown'} | "
            f"citation={evidence.citation_key} | score={evidence.score:.3f}"
        )
        lines.append(evidence.text)
        lines.append("")
    return "\n".join(lines).strip()


def _chunk_matches_section_filter(chunk: Chunk, section_filter: list[str]) -> bool:
    """Return whether a chunk belongs to any requested section path."""
    section_label = _section_label(chunk).lower()
    section_parts = [part.lower() for part in chunk.section_path]

    for section in section_filter:
        normalized_section = " > ".join(part.strip().lower() for part in section.split(">") if part.strip())
        if not normalized_section:
            continue
        if section_label == normalized_section or section_label.startswith(f"{normalized_section} >"):
            return True
        if normalized_section in section_parts:
            return True
    return False


def _render_chunks_with_token_budget(chunks: list[Chunk], anchor_chunk_id: str, max_tokens: int) -> tuple[str, list[Chunk]]:
    """Join chunk display text while keeping the hit chunk inside the budget."""
    anchor_index = next((index for index, chunk in enumerate(chunks) if chunk.chunk_id == anchor_chunk_id), 0)
    anchor_chunk = chunks[anchor_index]
    anchor_text = anchor_chunk.display_text or anchor_chunk.content
    anchor_tokens = _estimate_tokens(anchor_text)

    if anchor_tokens >= max_tokens:
        return _trim_to_token_budget(anchor_text, max_tokens), [anchor_chunk]

    included_by_id = {anchor_chunk.chunk_id}
    used_tokens = anchor_tokens

    for neighbor_index in _neighbor_indexes(anchor_index, len(chunks)):
        neighbor = chunks[neighbor_index]
        neighbor_text = neighbor.display_text or neighbor.content
        neighbor_tokens = _estimate_tokens(neighbor_text)
        if used_tokens + neighbor_tokens > max_tokens:
            continue
        included_by_id.add(neighbor.chunk_id)
        used_tokens += neighbor_tokens

    included_chunks = [chunk for chunk in chunks if chunk.chunk_id in included_by_id]
    rendered_parts = [chunk.display_text or chunk.content for chunk in included_chunks]
    return "\n\n".join(part for part in rendered_parts if part).strip(), included_chunks


def _neighbor_indexes(anchor_index: int, chunk_count: int) -> list[int]:
    """Return neighbor indexes by proximity while alternating before and after."""
    indexes: list[int] = []
    offset = 1
    while anchor_index - offset >= 0 or anchor_index + offset < chunk_count:
        before_index = anchor_index - offset
        after_index = anchor_index + offset
        if before_index >= 0:
            indexes.append(before_index)
        if after_index < chunk_count:
            indexes.append(after_index)
        offset += 1
    return indexes


def _estimate_tokens(text: str) -> int:
    """Use a simple deterministic token approximation for retrieval budgeting."""
    return len(text.split())


def _trim_to_token_budget(text: str, max_tokens: int) -> str:
    """Trim text to the approximate token budget."""
    if max_tokens <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def _merge_page_ranges(chunks: list[Chunk]) -> tuple[int, int] | None:
    """Merge available page ranges from included evidence chunks."""
    ranges = [chunk.page_range for chunk in chunks if chunk.page_range is not None]
    if not ranges:
        return None
    return (min(page_range[0] for page_range in ranges), max(page_range[1] for page_range in ranges))


def _section_label(chunk: Chunk) -> str:
    """Return a readable section label for one chunk."""
    if chunk.section_path:
        return " > ".join(chunk.section_path)
    return chunk.chunk_name if chunk.chunk_name != "unknown" else ""
