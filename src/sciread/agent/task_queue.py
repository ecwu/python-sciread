"""Task queue management system for multi-agent discussion."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Awaitable
from collections import defaultdict

from .models.discussion_models import AgentPersonality
from .models.task_models import (
    Task, TaskType, TaskPriority, TaskStatus, TaskResult, TaskQueue
)
from ..logging_config import get_logger

logger = get_logger(__name__)


class TaskQueueManager:
    """Manages multiple task queues and task execution coordination."""

    def __init__(self, max_concurrent_tasks: int = 10):
        """Initialize the task queue manager."""
        self.queues: Dict[str, TaskQueue] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_callbacks: Dict[TaskType, Callable] = {}
        self.is_running = False
        self._background_task: Optional[asyncio.Task] = None
        self.task_execution_history: List[Dict[str, Any]] = []

    def create_queue(self, name: str, description: Optional[str] = None) -> TaskQueue:
        """Create a new task queue."""
        if name in self.queues:
            raise ValueError(f"Task queue '{name}' already exists")

        queue = TaskQueue(
            name=name,
            description=description,
            max_concurrent_tasks=self.max_concurrent_tasks
        )
        self.queues[name] = queue
        logger.info(f"Created task queue '{name}'")
        return queue

    def get_queue(self, name: str) -> Optional[TaskQueue]:
        """Get a task queue by name."""
        return self.queues.get(name)

    def register_task_callback(self, task_type: TaskType, callback: Callable) -> None:
        """Register a callback function for a specific task type."""
        self.task_callbacks[task_type] = callback
        logger.info(f"Registered callback for task type '{task_type}'")

    def add_task(self, queue_name: str, task: Task) -> str:
        """Add a task to a specific queue."""
        if queue_name not in self.queues:
            raise ValueError(f"Task queue '{queue_name}' does not exist")

        queue = self.queues[queue_name]
        task_id = queue.add_task(task)
        logger.debug(f"Added task {task_id} to queue '{queue_name}'")
        return task_id

    def create_task(
        self,
        queue_name: str,
        task_type: TaskType,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        assigned_to: Optional[AgentPersonality] = None,
        created_by: Optional[AgentPersonality] = None,
        depends_on: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """Create and add a new task to a queue."""
        task = Task(
            task_type=task_type,
            priority=priority,
            assigned_to=assigned_to,
            created_by=created_by,
            parameters=parameters,
            depends_on=depends_on or [],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return self.add_task(queue_name, task)

    def get_next_task_for_agent(self, agent: AgentPersonality, queue_name: Optional[str] = None) -> Optional[Task]:
        """Get the next available task for a specific agent."""
        # If queue_name is specified, only check that queue
        if queue_name:
            if queue_name not in self.queues:
                return None
            return self.queues[queue_name].get_next_task(agent)

        # Check all queues in priority order (you can customize this)
        for name, queue in sorted(self.queues.items()):
            task = queue.get_next_task(agent)
            if task:
                return task

        return None

    def assign_task_to_agent(self, task_id: str, agent: AgentPersonality) -> bool:
        """Assign a specific task to an agent."""
        for queue in self.queues.values():
            if queue.assign_task(task_id, agent):
                logger.info(f"Assigned task {task_id} to {agent}")
                return True
        return False

    async def execute_task(self, task: Task, queue: TaskQueue) -> TaskResult:
        """Execute a single task."""
        logger.info(f"Executing task {task.task_id} of type {task.task_type}")
        start_time = datetime.now()

        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = start_time

            # Get the callback for this task type
            callback = self.task_callbacks.get(task.task_type)
            if not callback:
                raise ValueError(f"No callback registered for task type '{task.task_type}'")

            # Execute the task with timeout
            timeout = task.timeout_seconds or 300  # Default 5 minutes
            result = await asyncio.wait_for(callback(task), timeout=timeout)

            # Ensure result has proper task_id
            if isinstance(result, TaskResult):
                result.task_id = task.task_id
            else:
                # Convert non-TaskResult to TaskResult
                result = TaskResult(
                    task_id=task.task_id,
                    success=True,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    analysis_result=str(result),
                    confidence=0.8,
                )

            queue.complete_task(task.task_id, result)
            logger.info(f"Completed task {task.task_id} successfully")

            # Record execution history
            self._record_task_execution(task, result, success=True)
            return result

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(f"Task {task.task_id} failed: {error_msg}")

            # Create failure result
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=error_msg,
                confidence=0.0,
            )

            queue.fail_task(task.task_id, error_msg)

            # Record execution history
            self._record_task_execution(task, result, success=False)
            return result

    def _record_task_execution(self, task: Task, result: TaskResult, success: bool) -> None:
        """Record task execution in history."""
        self.task_execution_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "assigned_to": task.assigned_to,
            "success": success,
            "execution_time": result.execution_time,
            "confidence": result.confidence,
            "timestamp": datetime.now(),
        })

        # Keep only last 1000 executions
        if len(self.task_execution_history) > 1000:
            self.task_execution_history = self.task_execution_history[-1000:]

    async def start_processing(self) -> None:
        """Start the background task processing loop."""
        if self.is_running:
            logger.warning("Task queue manager is already running")
            return

        self.is_running = True
        self._background_task = asyncio.create_task(self._processing_loop())
        logger.info("Started task queue processing loop")

    async def stop_processing(self) -> None:
        """Stop the background task processing loop."""
        if not self.is_running:
            return

        self.is_running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped task queue processing loop")

    async def _processing_loop(self) -> None:
        """Main background processing loop."""
        logger.info("Task processing loop started")

        while self.is_running:
            try:
                # Check for tasks to execute
                tasks_executed = 0
                for queue in self.queues.values():
                    # Get available tasks that can be executed
                    available_tasks = [
                        task for task in queue.pending_tasks
                        if task.assigned_to and self._are_dependencies_satisfied(queue, task)
                    ]

                    # Execute tasks up to the concurrent limit
                    while (available_tasks and
                           len(queue.active_tasks) < queue.max_concurrent_tasks and
                           tasks_executed < 5):  # Limit per iteration to prevent blocking

                        task = available_tasks.pop(0)
                        queue.assign_task(task.task_id, task.assigned_to)

                        # Execute task asynchronously
                        asyncio.create_task(self.execute_task(task, queue))
                        tasks_executed += 1

                # Sleep briefly if no tasks were executed
                if tasks_executed == 0:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

        logger.info("Task processing loop stopped")

    def _are_dependencies_satisfied(self, queue: TaskQueue, task: Task) -> bool:
        """Check if all dependencies for a task are satisfied."""
        for dep_id in task.depends_on:
            dep_status = queue.get_task_status(dep_id)
            if dep_status != TaskStatus.COMPLETED:
                return False
        return True

    def get_queue_statistics(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific queue."""
        queue = self.get_queue(queue_name)
        if not queue:
            return None

        # Count tasks by status and type
        status_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for task in queue.pending_tasks + queue.active_tasks + queue.completed_tasks + queue.failed_tasks:
            status_counts[task.status.value] += 1
            type_counts[task.task_type.value] += 1

        # Calculate average execution time
        completed_times = [
            task.result.execution_time
            for task in queue.completed_tasks
            if task.result and task.result.execution_time
        ]
        avg_execution_time = sum(completed_times) / len(completed_times) if completed_times else 0

        return {
            "name": queue.name,
            "description": queue.description,
            "total_tasks": queue.total_tasks_created,
            "completed_tasks": queue.total_tasks_completed,
            "current_backlog": len(queue.pending_tasks),
            "active_tasks": len(queue.active_tasks),
            "failed_tasks": len(queue.failed_tasks),
            "success_rate": (queue.total_tasks_completed / max(1, queue.total_tasks_created)),
            "status_counts": dict(status_counts),
            "type_counts": dict(type_counts),
            "average_execution_time": avg_execution_time,
            "last_activity": queue.last_activity.isoformat(),
        }

    def get_agent_workload(self, agent: AgentPersonality) -> Dict[str, int]:
        """Get current workload for an agent across all queues."""
        workload = {
            "active_tasks": 0,
            "pending_assigned": 0,
            "completed_today": 0,
            "failed_today": 0,
        }

        today = datetime.now().date()
        for queue in self.queues.values():
            # Active tasks
            workload["active_tasks"] += queue.get_agent_workload(agent)

            # Pending assigned tasks
            workload["pending_assigned"] += len([
                task for task in queue.pending_tasks
                if task.assigned_to == agent
            ])

            # Completed today
            workload["completed_today"] += len([
                task for task in queue.completed_tasks
                if task.assigned_to == agent and
                task.completed_at and
                task.completed_at.date() == today
            ])

            # Failed today
            workload["failed_today"] += len([
                task for task in queue.failed_tasks
                if task.assigned_to == agent and
                task.completed_at and
                task.completed_at.date() == today
            ])

        return workload

    def cleanup_old_tasks(self, days_old: int = 7) -> None:
        """Remove old completed and failed tasks from all queues."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        total_removed = 0

        for queue in self.queues.values():
            # Filter out old tasks
            original_counts = (len(queue.completed_tasks), len(queue.failed_tasks))

            queue.completed_tasks = [
                task for task in queue.completed_tasks
                if task.completed_at and task.completed_at > cutoff_date
            ]

            queue.failed_tasks = [
                task for task in queue.failed_tasks
                if task.completed_at and task.completed_at > cutoff_date
            ]

            total_removed += (original_counts[0] - len(queue.completed_tasks)) + \
                           (original_counts[1] - len(queue.failed_tasks))

        logger.info(f"Cleaned up {total_removed} old tasks from all queues")