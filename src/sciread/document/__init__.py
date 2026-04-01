"""Public compatibility facade for sciread's document APIs."""

from ..document_ingestion import MineruClient
from ..document_structure import Chunk
from ..document_structure import Document
from ..document_structure import DocumentBuilder
from ..document_structure import DocumentFactory
from ..document_structure.splitters import SemanticSplitter
from ..embedding_provider import OllamaClient
from ..embedding_provider import SiliconFlowClient

__all__ = [
    "Chunk",
    "Document",
    "DocumentBuilder",
    "DocumentFactory",
    "MineruClient",
    "OllamaClient",
    "SemanticSplitter",
    "SiliconFlowClient",
]
