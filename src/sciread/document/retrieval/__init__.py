"""Document retrieval services and vector indexing."""

from .evidence import EvidenceRetriever
from .evidence import format_evidence_results
from .models import Evidence
from .models import RetrievedChunk
from .search import SUPPORTED_RETRIEVERS
from .search import format_retrieval_results
from .search import hybrid_search
from .search import lexical_search
from .search import rerank_chunk_search
from .search import retrieve_chunks
from .search import semantic_chunk_search
from .search import tree_search
from .service import build_vector_index
from .service import cosine_similarity
from .service import rerank_search
from .service import semantic_search
from .vector_index import VectorIndex

__all__ = [
    "SUPPORTED_RETRIEVERS",
    "Evidence",
    "EvidenceRetriever",
    "RetrievedChunk",
    "VectorIndex",
    "build_vector_index",
    "cosine_similarity",
    "format_evidence_results",
    "format_retrieval_results",
    "hybrid_search",
    "lexical_search",
    "rerank_chunk_search",
    "rerank_search",
    "retrieve_chunks",
    "semantic_chunk_search",
    "semantic_search",
    "tree_search",
]
