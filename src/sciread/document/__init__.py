"""Document processing module for sciread.

This module provides functionality for loading, processing, and managing
academic documents through a structured approach with loaders, splitters,
and document management.
"""

from .document import Document
from .document_builder import DocumentBuilder
from .document_builder import DocumentFactory
from .external_clients import MineruClient
from .external_clients import OllamaClient
from .models import Chunk
from .models import CoverageStats
from .splitters.semantic_splitter import SemanticSplitter
from .splitters.topic_flow import TopicFlowSplitter

__all__ = [
    "Chunk",
    "CoverageStats",
    "Document",
    "DocumentBuilder",
    "DocumentFactory",
    "MineruClient",
    "OllamaClient",
    "SemanticSplitter",
    "TopicFlowSplitter",
]
