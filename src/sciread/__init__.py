from .application import compute
from .application import run_coordinate_analysis
from .application import run_discussion_analysis
from .application import run_react_analysis
from .application import run_simple_analysis
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
    "compute",
    "get_embedding_client",
    "get_logger",
    "get_model",
    "run_coordinate_analysis",
    "run_discussion_analysis",
    "run_react_analysis",
    "run_simple_analysis",
    "setup_logging",
]
