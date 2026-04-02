"""Private runtime state for Document instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .retrieval.vector_index import VectorIndex


@dataclass
class DocumentRuntimeState:
    """Holds non-serialized runtime-only dependencies for a document."""

    embedding_client: Any | None = None
    vector_index: VectorIndex | None = None
