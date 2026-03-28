"""Data models for the agent system.

This module contains all Pydantic models used by the various agents
for structured inputs and outputs.
"""

from .coordinate_models import AnalysisPlan
from .coordinate_models import ComprehensiveAnalysisResult
from .coordinate_models import ExperimentResult
from .coordinate_models import FutureDirectionsResult
from .coordinate_models import MetadataExtractionResult
from .coordinate_models import MethodologyResult
from .coordinate_models import PreviousMethodsResult
from .coordinate_models import ResearchQuestionsResult
from .react_models import ReActAgentInput
from .react_models import ReActAgentOutput
from .simple_models import SimpleAnalysisResult
from .react_models import ReActAgentOutput
from .simple_models import SimpleAnalysisResult

__all__ = [
    "MetadataExtractionResult",
    "MethodologyResult",
    "PreviousMethodsResult",
    "Question",
    # ReActAgent models
    "ReActAgentInput",
    "ReActAgentOutput",
    "ResearchQuestionsResult",
    "Response",
    # SimpleAgent models
    "SimpleAnalysisResult",
    # Task models
    "SimpleAnalysisResult",
]
