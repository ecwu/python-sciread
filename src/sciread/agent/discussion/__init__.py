"""Discussion agent module for multi-agent document analysis.

This module provides a multi-agent discussion system where different AI personalities
collaborate to analyze academic papers, ask questions, and build consensus.
"""

from .agent import DiscussionAgent
from .consensus import ConsensusBuilder
from .models import AgentInsight
from .models import AgentPersonality
from .models import ConsensusPoint
from .models import DiscussionPhase
from .models import DiscussionResult
from .models import DiscussionState
from .models import DivergentView
from .models import Question
from .models import Response
from .personalities import PersonalityAgent
from .personalities import create_personality_agent
from .task_models import Task
from .task_models import TaskPriority
from .task_models import TaskQueue
from .task_models import TaskResult
from .task_models import TaskStatus
from .task_models import TaskType
from .task_queue import TaskQueueManager

__all__ = [
    "AgentInsight",
    "AgentPersonality",
    "ConsensusBuilder",
    "ConsensusPoint",
    "DiscussionAgent",
    "DiscussionPhase",
    "DiscussionResult",
    "DiscussionState",
    "DivergentView",
    "PersonalityAgent",
    "Question",
    "Response",
    "Task",
    "TaskPriority",
    "TaskQueue",
    "TaskQueueManager",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "create_personality_agent",
]
