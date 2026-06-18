"""Simple analysis subsystem."""

from .agent import SimpleAgent
from .agent import analyze_file_with_simple
from .agent import analyze_file_with_simple_sync
from .models import SimpleAnalysisResult
from .prompts import DEFAULT_SYSTEM_PROMPT
from .prompts import DEFAULT_TASK_PROMPT
from .prompts import build_analysis_prompt

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TASK_PROMPT",
    "SimpleAgent",
    "SimpleAnalysisResult",
    "analyze_file_with_simple",
    "analyze_file_with_simple_sync",
    "build_analysis_prompt",
]
