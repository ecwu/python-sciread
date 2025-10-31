from .core import comprehensive_analysis
from .core import compute
from .core import main
from .core import run_rag_react_analysis
from .core import run_react_analysis
from .embedding_provider import EmbeddingFactory
from .embedding_provider import get_embedding_client
from .embedding_provider import OllamaClient
from .embedding_provider import SiliconFlowClient
from .llm_provider import get_model
from .logging_config import get_logger
from .logging_config import setup_logging

__version__ = "0.0.0"

__all__ = [
    "EmbeddingFactory",
    "OllamaClient",
    "SiliconFlowClient",
    "comprehensive_analysis",
    "compute",
    "get_embedding_client",
    "get_logger",
    "get_model",
    "main",
    "run_rag_react_analysis",
    "run_react_analysis",
    "setup_logging",
]
