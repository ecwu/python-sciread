"""Data models for the discussion agent system."""

from .discussion_models import (
    AgentPersonality,
    DiscussionState,
    DiscussionPhase,
    AgentInsight,
    Question,
    Response,
    DiscussionResult,
)

from .task_models import (
    Task,
    TaskType,
    TaskPriority,
    TaskStatus,
    TaskResult,
    TaskQueue,
)

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
]