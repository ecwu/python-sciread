from .application import run_coordinate_analysis
from .application import run_discussion_analysis
from .application import run_react_analysis
from .application import run_search_react_analysis
from .application import run_simple_analysis
from .platform.logging import get_logger
from .platform.logging import setup_logging
from .providers.embedding import EmbeddingFactory
from .providers.embedding import LMStudioClient
from .providers.embedding import OllamaClient
from .providers.embedding import SiliconFlowClient
from .providers.embedding import get_embedding_client
from .providers.llm import get_model
from .providers.rerank import RerankFactory
from .providers.rerank import SiliconFlowRerankClient
from .providers.rerank import get_rerank_client

__version__ = "1.1.0"

__all__ = [
    "EmbeddingFactory",
    "LMStudioClient",
    "OllamaClient",
    "RerankFactory",
    "SiliconFlowClient",
    "SiliconFlowRerankClient",
    "get_embedding_client",
    "get_logger",
    "get_model",
    "get_rerank_client",
    "run_coordinate_analysis",
    "run_discussion_analysis",
    "run_react_analysis",
    "run_search_react_analysis",
    "run_simple_analysis",
    "setup_logging",
]
