from .core import compute, main, run_main
from .llm_provider import get_model
from .logging_config import get_logger
from .logging_config import setup_logging

__version__ = "0.0.0"

__all__ = [
    "compute",
    "main",
    "run_main",
    "get_logger",
    "get_model",
    "setup_logging",
]
