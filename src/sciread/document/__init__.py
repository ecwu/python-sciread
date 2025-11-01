"""Document processing module for sciread.

This module provides functionality for loading, processing, and managing
academic documents through a structured approach with loaders, splitters,
and document management.
"""

# Import embedding clients from embedding_provider module
from ..embedding_provider import OllamaClient
from ..embedding_provider import SiliconFlowClient
from .document import Document
from .document_builder import DocumentBuilder
from .external_clients import MineruClient
from .factory import DocumentFactory
from .models import Chunk
from .splitters.consecutive_flow import ConsecutiveFlowSplitter
from .splitters.cumulative_flow import CumulativeFlowSplitter
from .splitters.semantic_splitter import SemanticSplitter

__all__ = [
    "Chunk",
    "ConsecutiveFlowSplitter",
    "CumulativeFlowSplitter",
    "Document",
    "DocumentBuilder",
    "DocumentFactory",
    "MineruClient",
    "OllamaClient",
    "SemanticSplitter",
    "SiliconFlowClient",
]
