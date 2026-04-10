"""Runtime section-tree helpers derived from document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from .paths import clean_section_name

if TYPE_CHECKING:
    from sciread.document.document import Document
    from sciread.document.models import Chunk


@dataclass(slots=True)
class SectionTreeNode:
    """One runtime node in the derived section tree."""

    name: str
    path: list[str]
    chunk_ids: list[str] = field(default_factory=list)
    children: list[SectionTreeNode] = field(default_factory=list)
    content_length: int = 0
    retrievable: bool = True

    @property
    def path_text(self) -> str:
        """Return the node path as a user-facing label."""
        return " > ".join(self.path)


@dataclass(slots=True)
class SectionTree:
    """Runtime section tree built from the document chunks."""

    roots: list[SectionTreeNode]
    nodes_by_path: dict[str, SectionTreeNode]

    def find(self, path: str | None) -> SectionTreeNode | None:
        """Find a node by normalized path."""
        if not path:
            return None
        normalized_path = normalize_section_path(path)
        return self.nodes_by_path.get(normalized_path)

    def render(self, path: str | None = None, depth: int = 2) -> str:
        """Render the tree or subtree as plain text."""
        target = self.find(path) if path else None
        roots = target.children if target else self.roots
        lines: list[str] = []

        if target:
            lines.append(f"[Root] {target.path_text} | chars={target.content_length} | chunks={len(target.chunk_ids)}")

        for root in roots:
            _render_tree_lines(root, lines, current_depth=1, max_depth=depth)

        return "\n".join(lines) if lines else "No section tree available."


def normalize_section_path(path: str) -> str:
    """Normalize a section path for matching."""
    return " > ".join(clean_section_name(part) for part in path.split(">") if part.strip())


def build_section_tree(document: Document) -> SectionTree:
    """Build a runtime section tree from the current document chunks."""
    nodes_by_path: dict[str, SectionTreeNode] = {}
    roots: list[SectionTreeNode] = []

    for chunk in document.chunks:
        section_path = [part for part in chunk.section_path if part.strip()]
        if not section_path:
            continue

        parent_node: SectionTreeNode | None = None
        for depth in range(len(section_path)):
            current_path = section_path[: depth + 1]
            path_key = " > ".join(current_path)
            node = nodes_by_path.get(path_key)
            if node is None:
                node = SectionTreeNode(
                    name=current_path[-1],
                    path=current_path.copy(),
                    retrievable=chunk.retrievable,
                )
                nodes_by_path[path_key] = node
                if parent_node is None:
                    roots.append(node)
                else:
                    parent_node.children.append(node)

            node.content_length += len(chunk.content_plain or chunk.content)
            if chunk.chunk_id not in node.chunk_ids:
                node.chunk_ids.append(chunk.chunk_id)
            node.retrievable = node.retrievable or chunk.retrievable
            parent_node = node

    return SectionTree(roots=roots, nodes_by_path=nodes_by_path)


def iter_descendant_chunks(node: SectionTreeNode, chunk_map: dict[str, Chunk]) -> list[Chunk]:
    """Return descendant chunks in insertion order without duplicates."""
    ordered_chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    def visit(current: SectionTreeNode) -> None:
        for chunk_id in current.chunk_ids:
            if chunk_id in seen_chunk_ids or chunk_id not in chunk_map:
                continue
            seen_chunk_ids.add(chunk_id)
            ordered_chunks.append(chunk_map[chunk_id])
        for child in current.children:
            visit(child)

    visit(node)
    return ordered_chunks


def _render_tree_lines(node: SectionTreeNode, lines: list[str], current_depth: int, max_depth: int) -> None:
    """Append rendered lines for one subtree."""
    indent = "  " * max(0, current_depth - 1)
    lines.append(f"{indent}- {node.name} | chars={node.content_length} | chunks={len(node.chunk_ids)}")
    if current_depth >= max_depth:
        return
    for child in node.children:
        _render_tree_lines(child, lines, current_depth=current_depth + 1, max_depth=max_depth)
