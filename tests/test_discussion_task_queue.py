"""Tests for discussion task models and queue management."""

import asyncio
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest

from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.task_models import Task
from sciread.agent.discussion.task_models import TaskPriority
from sciread.agent.discussion.task_models import TaskQueue
from sciread.agent.discussion.task_models import TaskResult
from sciread.agent.discussion.task_models import TaskStatus
from sciread.agent.discussion.task_models import TaskType
from sciread.agent.discussion.task_models import _priority_to_value
from sciread.agent.discussion.task_queue import TaskQueueManager


def _make_task(
    *,
    task_id: str,
    task_type: TaskType = TaskType.GENERATE_INSIGHTS,
    priority: TaskPriority = TaskPriority.MEDIUM,
    assigned_to: AgentPersonality | None = None,
    depends_on: list[str] | None = None,
    max_retries: int = 3,
) -> Task:
    return Task(
        task_id=task_id,
        task_type=task_type,
        priority=priority,
        assigned_to=assigned_to,
        depends_on=depends_on or [],
        max_retries=max_retries,
    )


def test_task_queue_orders_by_priority_and_dependency_state() -> None:
    """Queues should surface the highest-priority task whose dependencies are complete."""
    queue = TaskQueue(name="main")
    dependency = _make_task(task_id="dep", task_type=TaskType.ANALYZE_DOCUMENT)
    dependency.status = TaskStatus.COMPLETED

    blocked = _make_task(
        task_id="blocked",
        priority=TaskPriority.CRITICAL,
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
        depends_on=["missing"],
    )
    medium = _make_task(
        task_id="medium",
        priority=TaskPriority.MEDIUM,
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
    )
    ready = _make_task(
        task_id="ready",
        priority=TaskPriority.HIGH,
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
        depends_on=["dep"],
    )

    blocked.created_at = datetime.now(UTC)
    medium.created_at = datetime.now(UTC) - timedelta(seconds=2)
    ready.created_at = datetime.now(UTC) - timedelta(seconds=1)

    queue.completed_tasks.append(dependency)
    queue.pending_tasks.extend([blocked, medium, ready])

    next_task = queue.get_next_task(AgentPersonality.CRITICAL_EVALUATOR)

    assert next_task is not None
    assert next_task.task_id == "ready"
    assert _priority_to_value(TaskPriority.CRITICAL) > _priority_to_value(TaskPriority.LOW)
    assert _priority_to_value("unexpected") == 2


def test_task_queue_retries_then_moves_task_to_failed_bucket() -> None:
    """Failed tasks should be retried until they hit the configured retry ceiling."""
    queue = TaskQueue(name="main", auto_retry_failed_tasks=True)
    task = _make_task(task_id="retry-me", assigned_to=AgentPersonality.PRACTICAL_APPLICATOR, max_retries=1)
    queue.pending_tasks.append(task)

    assert queue.assign_task(task.task_id, AgentPersonality.PRACTICAL_APPLICATOR) is True
    assert queue.fail_task(task.task_id, "temporary error") is True
    assert queue.pending_tasks[0].retry_count == 1
    assert queue.pending_tasks[0].status == TaskStatus.PENDING
    assert queue.pending_tasks[0].assigned_to is None

    assert queue.assign_task(task.task_id, AgentPersonality.PRACTICAL_APPLICATOR) is True
    assert queue.fail_task(task.task_id, "permanent error") is True
    assert queue.failed_tasks[0].task_id == task.task_id
    assert queue.get_task_status(task.task_id) == TaskStatus.FAILED
    assert queue.get_task(task.task_id) is queue.failed_tasks[0]


def test_task_queue_complete_task_and_workload_tracking() -> None:
    """Completing a task should move it to the completed bucket and update counters."""
    queue = TaskQueue(name="main")
    task = _make_task(task_id="complete-me")
    queue.pending_tasks.append(task)

    assert queue.assign_task(task.task_id, AgentPersonality.THEORETICAL_INTEGRATOR) is True
    assert queue.get_agent_workload(AgentPersonality.THEORETICAL_INTEGRATOR) == 1

    result = TaskResult(
        task_id=task.task_id,
        success=True,
        execution_time=0.25,
        analysis_result="done",
        confidence=0.8,
    )
    assert queue.complete_task(task.task_id, result) is True

    completed = queue.completed_tasks[0]
    assert completed.result is result
    assert completed.status == TaskStatus.COMPLETED
    assert queue.total_tasks_completed == 1
    assert queue.get_agent_workload(AgentPersonality.THEORETICAL_INTEGRATOR) == 0


@pytest.mark.asyncio
async def test_task_queue_manager_executes_tasks_and_collects_statistics() -> None:
    """The manager should execute callbacks, record history, and expose summary stats."""
    manager = TaskQueueManager(max_concurrent_tasks=1)
    queue = manager.create_queue("discussion", description="Main queue")

    async def callback(task: Task) -> str:
        return f"processed {task.task_id}"

    manager.register_task_callback(TaskType.GENERATE_INSIGHTS, callback)
    task_id = manager.create_task(
        "discussion",
        TaskType.GENERATE_INSIGHTS,
        parameters={"document": "paper"},
        priority=TaskPriority.HIGH,
        assigned_to=AgentPersonality.INNOVATIVE_INSIGHTER,
    )

    assigned = queue.get_next_task(AgentPersonality.INNOVATIVE_INSIGHTER)
    assert assigned is not None
    assert queue.assign_task(task_id, AgentPersonality.INNOVATIVE_INSIGHTER) is True

    result = await manager.execute_task(assigned, queue)

    assert result.success is True
    assert result.analysis_result == f"processed {task_id}"
    assert len(manager.task_execution_history) == 1

    stats = manager.get_queue_statistics("discussion")
    assert stats is not None
    assert stats["completed_tasks"] == 1
    assert stats["type_counts"]["generate_insights"] == 1
    assert stats["status_counts"]["completed"] == 1

    workload = manager.get_agent_workload(AgentPersonality.INNOVATIVE_INSIGHTER)
    assert workload["completed_today"] == 1
    assert workload["active_tasks"] == 0


@pytest.mark.asyncio
async def test_task_queue_manager_background_processing_and_cleanup() -> None:
    """Background processing should execute assigned work and cleanup should drop stale tasks."""
    manager = TaskQueueManager(max_concurrent_tasks=1)
    queue = manager.create_queue("discussion")

    async def callback(task: Task) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=0.1,
            analysis_result="ok",
            confidence=0.9,
        )

    manager.register_task_callback(TaskType.GENERATE_INSIGHTS, callback)
    task_id = manager.create_task(
        "discussion",
        TaskType.GENERATE_INSIGHTS,
        parameters={},
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
    )

    await manager.start_processing()
    await manager.start_processing()
    await asyncio.sleep(0.2)
    await manager.stop_processing()

    task = queue.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.COMPLETED

    old_completed = _make_task(task_id="old-completed")
    old_completed.status = TaskStatus.COMPLETED
    old_completed.completed_at = datetime.now(UTC) - timedelta(days=10)

    old_failed = _make_task(task_id="old-failed")
    old_failed.status = TaskStatus.FAILED
    old_failed.assigned_to = AgentPersonality.CRITICAL_EVALUATOR
    old_failed.completed_at = datetime.now(UTC) - timedelta(days=10)

    queue.completed_tasks.append(old_completed)
    queue.failed_tasks.append(old_failed)
    manager.cleanup_old_tasks(days_old=7)

    assert all(saved_task.task_id != "old-completed" for saved_task in queue.completed_tasks)
    assert all(saved_task.task_id != "old-failed" for saved_task in queue.failed_tasks)


@pytest.mark.asyncio
async def test_task_queue_manager_failure_paths_and_duplicates() -> None:
    """Duplicate queues and failing callbacks should be handled predictably."""
    manager = TaskQueueManager()
    queue = manager.create_queue("discussion")

    with pytest.raises(ValueError, match="already exists"):
        manager.create_queue("discussion")

    with pytest.raises(ValueError, match="does not exist"):
        manager.add_task("missing", _make_task(task_id="missing"))

    task = _make_task(
        task_id="will-fail",
        task_type=TaskType.ASK_QUESTION,
        assigned_to=AgentPersonality.PRACTICAL_APPLICATOR,
    )
    queue.pending_tasks.append(task)
    queue.assign_task(task.task_id, AgentPersonality.PRACTICAL_APPLICATOR)

    failed_result = await manager.execute_task(task, queue)

    assert failed_result.success is False
    assert queue.pending_tasks[0].task_id == task.task_id
    assert manager.assign_task_to_agent("missing", AgentPersonality.PRACTICAL_APPLICATOR) is False
    assert manager.get_next_task_for_agent(AgentPersonality.PRACTICAL_APPLICATOR, queue_name="unknown") is None
    assert manager.get_queue_statistics("unknown") is None
