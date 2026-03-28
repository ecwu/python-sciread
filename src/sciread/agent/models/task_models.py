"""Task management models for multi-agent discussion system."""

from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .discussion_models import AgentInsight
from .discussion_models import AgentPersonality
from .discussion_models import Question
from .discussion_models import Response


class TaskType(StrEnum):
    """Different types of tasks in the discussion system."""

    ANALYZE_DOCUMENT = "analyze_document"
    GENERATE_INSIGHTS = "generate_insights"
    ASK_QUESTION = "ask_question"
    ANSWER_QUESTION = "answer_question"
    EVALUATE_RESPONSE = "evaluate_response"
    BUILD_CONSENSUS = "build_consensus"
    SYNTHESIZE_REPORT = "synthesize_report"
    MONITOR_CONVERGENCE = "monitor_convergence"


class TaskPriority(StrEnum):
    """Priority levels for tasks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(StrEnum):
    """Status of a task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a task to be executed by an agent."""

    task_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")

    # Task assignment
    assigned_to: AgentPersonality | None = Field(None, description="Agent assigned to this task")
    created_by: AgentPersonality | None = Field(None, description="Agent that created this task")

    # Task parameters
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context for task")

    # Dependencies and relationships
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this task depends on")
    related_tasks: list[str] = Field(default_factory=list, description="Related task IDs")

    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")
    assigned_at: datetime | None = Field(None, description="Task assignment time")
    started_at: datetime | None = Field(None, description="Task start time")
    completed_at: datetime | None = Field(None, description="Task completion time")
    deadline: datetime | None = Field(None, description="Task deadline")

    # Execution details
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_count: int = Field(default=0, description="Current retry count")
    timeout_seconds: int | None = Field(None, description="Task timeout in seconds")

    # Results
    result: "TaskResult | None" = Field(None, description="Task execution result")
    error_message: str | None = Field(None, description="Error message if task failed")

    model_config = ConfigDict(use_enum_values=True)


class TaskResult(BaseModel):
    """Represents the result of a task execution."""

    task_id: str = Field(..., description="ID of the task this result belongs to")
    success: bool = Field(..., description="Whether the task was successful")
    execution_time: float = Field(..., description="Time taken to execute the task in seconds")

    # Task-specific results
    insights: list[AgentInsight] = Field(default_factory=list, description="Generated insights")
    questions: list[Question] = Field(default_factory=list, description="Generated questions")
    responses: list[Response] = Field(default_factory=list, description="Generated responses")
    analysis_result: str | None = Field(None, description="General analysis result")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the result")
    notes: list[str] = Field(default_factory=list, description="Additional notes about the result")
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")

    model_config = ConfigDict(use_enum_values=True)


class TaskQueue(BaseModel):
    """Manages a queue of tasks for the discussion system."""

    name: str = Field(..., description="Name of this task queue")
    description: str | None = Field(None, description="Description of this queue")

    # Task storage
    pending_tasks: list[Task] = Field(default_factory=list, description="Tasks waiting to be executed")
    active_tasks: list[Task] = Field(default_factory=list, description="Tasks currently being executed")
    completed_tasks: list[Task] = Field(default_factory=list, description="Completed tasks")
    failed_tasks: list[Task] = Field(default_factory=list, description="Failed tasks")

    # Queue metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Queue creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    total_tasks_created: int = Field(default=0, description="Total number of tasks created")
    total_tasks_completed: int = Field(default=0, description="Total number of tasks completed")

    # Configuration
    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent tasks")
    default_priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Default task priority")
    auto_retry_failed_tasks: bool = Field(default=True, description="Whether to automatically retry failed tasks")

    model_config = ConfigDict(use_enum_values=True)

    def add_task(self, task: Task) -> str:
        """Add a new task to the queue."""
        task.created_at = datetime.now(UTC)
        self.pending_tasks.append(task)
        self.total_tasks_created += 1
        self.last_activity = datetime.now(UTC)
        return task.task_id

    def get_next_task(self, agent_personality: AgentPersonality) -> Task | None:
        """Get the next available task for a specific agent."""
        # Sort pending tasks by priority and creation time
        sorted_tasks = sorted(
            self.pending_tasks,
            key=lambda t: (_priority_to_value(t.priority), t.created_at),
            reverse=True,
        )

        for task in sorted_tasks:
            if task.assigned_to is None or task.assigned_to == agent_personality:
                # Check if dependencies are satisfied
                if self._are_dependencies_satisfied(task):
                    return task

        return None

    def assign_task(self, task_id: str, agent: AgentPersonality) -> bool:
        """Assign a task to an agent."""
        for i, task in enumerate(self.pending_tasks):
            if task.task_id == task_id:
                task.assigned_to = agent
                task.status = TaskStatus.ASSIGNED
                task.assigned_at = datetime.now(UTC)

                # Move to active tasks
                self.active_tasks.append(task)
                self.pending_tasks.pop(i)
                self.last_activity = datetime.now(UTC)
                return True

        return False

    def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """Mark a task as completed with its result."""
        for i, task in enumerate(self.active_tasks):
            if task.task_id == task_id:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(UTC)

                # Move to completed tasks
                self.completed_tasks.append(task)
                self.active_tasks.pop(i)
                self.total_tasks_completed += 1
                self.last_activity = datetime.now(UTC)
                return True

        return False

    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark a task as failed."""
        for i, task in enumerate(self.active_tasks):
            if task.task_id == task_id:
                task.status = TaskStatus.FAILED
                task.error_message = error_message

                # Check if we should retry
                if self.auto_retry_failed_tasks and task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_to = None
                    task.assigned_at = None
                    task.started_at = None

                    # Move back to pending
                    self.pending_tasks.append(task)
                    self.active_tasks.pop(i)
                else:
                    # Move to failed tasks
                    self.failed_tasks.append(task)
                    self.active_tasks.pop(i)

                self.last_activity = datetime.now(UTC)
                return True

        return False

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task by ID from any queue bucket."""
        for task_list in [
            self.pending_tasks,
            self.active_tasks,
            self.completed_tasks,
            self.failed_tasks,
        ]:
            for task in task_list:
                if task.task_id == task_id:
                    return task
        return None

    def get_task_status(self, task_id: str) -> TaskStatus | None:
        """Get the status of a specific task."""
        for task_list in [
            self.pending_tasks,
            self.active_tasks,
            self.completed_tasks,
            self.failed_tasks,
        ]:
            for task in task_list:
                if task.task_id == task_id:
                    return task.status
        return None

    def get_agent_workload(self, agent: AgentPersonality) -> int:
        """Get the number of tasks currently assigned to an agent."""
        return len([task for task in self.active_tasks if task.assigned_to == agent])

    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all dependencies for a task are satisfied."""
        for dep_id in task.depends_on:
            dep_status = self.get_task_status(dep_id)
            if dep_status != TaskStatus.COMPLETED:
                return False
        return True


def _priority_to_value(priority: TaskPriority) -> int:
    """Convert priority to numerical value for sorting."""
    priority_values = {
        TaskPriority.LOW: 1,
        TaskPriority.MEDIUM: 2,
        TaskPriority.HIGH: 3,
        TaskPriority.CRITICAL: 4,
    }
    return priority_values.get(priority, 2)
