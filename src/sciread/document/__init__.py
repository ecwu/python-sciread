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
from .splitters.semantic_splitter import SemanticSplitter
from .splitters.topic_flow import TopicFlowSplitter

__all__ = [
    "Chunk",
    "Document",
    "DocumentBuilder",
    "DocumentFactory",
    "MineruClient",
    "OllamaClient",
    "SemanticSplitter",
    "TopicFlowSplitter",
]
