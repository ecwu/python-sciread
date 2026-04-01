from .application import comprehensive_analysis
from .application import compute
from .application import discussion_analysis
from .application import main
from .application import run_react_analysis
from .embedding_provider import EmbeddingFactory
from .embedding_provider import OllamaClient
from .embedding_provider import SiliconFlowClient
from .embedding_provider import get_embedding_client
from .llm_provider import get_model
from .platform.logging import get_logger
from .platform.logging import setup_logging

__version__ = "0.0.0"

__all__ = [
    "EmbeddingFactory",
    "OllamaClient",
    "SiliconFlowClient",
    "comprehensive_analysis",
    "compute",
    "discussion_analysis",
    "get_embedding_client",
    "get_logger",
    "get_model",
    "main",
    "run_react_analysis",
    "setup_logging",
]
