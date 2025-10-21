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
from .discussion_models import AgentInsight
from .discussion_models import AgentPersonality
from .discussion_models import DiscussionPhase
from .discussion_models import DiscussionResult
from .discussion_models import DiscussionState
from .discussion_models import Question
from .discussion_models import Response
from .react_models import ReActAgentInput
from .react_models import ReActAgentOutput
from .simple_models import SimpleAnalysisResult
from .task_models import Task
from .task_models import TaskPriority
from .task_models import TaskQueue
from .task_models import TaskResult
from .task_models import TaskStatus
from .task_models import TaskType

__all__ = [
    # Discussion models
    "AgentPersonality",
    "DiscussionState",
    "DiscussionPhase",
    "AgentInsight",
    "Question",
    "Response",
    "DiscussionResult",
    # Task models
    "Task",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    "TaskResult",
    "TaskQueue",
    # CoordinateAgent models
    "MetadataExtractionResult",
    "PreviousMethodsResult",
    "ResearchQuestionsResult",
    "MethodologyResult",
    "ExperimentResult",
    "FutureDirectionsResult",
    "AnalysisPlan",
    "ComprehensiveAnalysisResult",
    # SimpleAgent models
    "SimpleAnalysisResult",
    # ReActAgent models
    "ReActAgentInput",
    "ReActAgentOutput",
]
