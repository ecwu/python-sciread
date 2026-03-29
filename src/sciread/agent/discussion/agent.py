"""Main discussion agent for multi-agent document analysis."""

import asyncio
from datetime import UTC
from datetime import datetime
from datetime import timedelta

from pydantic_ai import Agent

from ...document import Document
from ...llm_provider import get_model
from ...logging_config import get_logger
from .models import AgentPersonality
from .models import DiscussionPhase
from .models import DiscussionResult
from .models import DiscussionState
from .models import QuestionIdGenerator
from .task_models import TaskPriority
from .task_models import TaskQueue
from .task_models import TaskStatus
from .task_models import TaskType
from .task_queue import TaskQueueManager

logger = get_logger(__name__)


class DiscussionAgent:
    """Main coordinator for multi-agent discussion system."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        max_iterations: int = 5,
        convergence_threshold: float = 0.75,
        max_discussion_time_minutes: int = 30,
    ):
        """Initialize the discussion agent."""
        self.model_name = model_name
        self.model = get_model(model_name)
        self.agent = Agent(
            self.model,
            system_prompt="You are a discussion coordinator for multi-agent academic paper analysis.",
        )

        # Configuration
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_discussion_time = timedelta(minutes=max_discussion_time_minutes)

        # Task management
        self.task_manager = TaskQueueManager(max_concurrent_tasks=4)
        self.discussion_queue: TaskQueue | None = None

        # State tracking
        self.discussion_state: DiscussionState | None = None
        self.agent_insights: dict[AgentPersonality, list] = {}
        self.all_questions: list = []
        self.all_responses: list = []
        self.question_id_gen = QuestionIdGenerator()

        # Register task tools
        self._register_task_tools()

    def _register_task_tools(self):
        """Register all task tools with the task manager."""
        from .tools import answer_question_tool
        from .tools import ask_question_tool
        from .tools import evaluate_convergence_tool
        from .tools import generate_insights_tool

        self.task_manager.register_task_callback(TaskType.GENERATE_INSIGHTS, generate_insights_tool)
        self.task_manager.register_task_callback(TaskType.ASK_QUESTION, ask_question_tool)
        self.task_manager.register_task_callback(TaskType.ANSWER_QUESTION, answer_question_tool)
        self.task_manager.register_task_callback(TaskType.MONITOR_CONVERGENCE, evaluate_convergence_tool)

    async def analyze_document(self, document: Document) -> DiscussionResult:
        """Main entry point for document analysis."""
        logger.info(f"Starting multi-agent discussion analysis for document: {document.metadata.title}")

        try:
            # Initialize discussion
            await self._initialize_discussion(document)

            # Start task processing
            await self.task_manager.start_processing()

            # Run discussion phases
            await self._run_discussion_phases(document)

            # Build final result
            result = await self._build_final_result(document)

            # Stop task processing
            await self.task_manager.stop_processing()

            logger.info("Discussion analysis completed successfully")
            return result

        except Exception as e:
            logger.error(f"Discussion analysis failed: {e}")
            # Create error result
            return DiscussionResult(
                document_title=document.metadata.title or "Untitled",
                summary=f"Discussion analysis failed: {e!s}",
                key_contributions=[],
                significance="Analysis failed",
                confidence_score=0.0,
                completion_time=datetime.now(UTC),
                discussion_metadata={"error": str(e)},
            )

    async def _initialize_discussion(self, document: Document):
        """Initialize discussion state and task queue."""
        # Create discussion queue
        self.discussion_queue = self.task_manager.create_queue("main_discussion", "Main queue for multi-agent document discussion")

        # Initialize discussion state
        self.discussion_state = DiscussionState(
            current_phase=DiscussionPhase.INITIAL_ANALYSIS,
            max_iterations=self.max_iterations,
            start_time=datetime.now(UTC),
        )

        # Initialize agent insights storage
        self.agent_insights = {personality: [] for personality in AgentPersonality}
        self.all_questions = []
        self.all_responses = []
        self.question_id_gen = QuestionIdGenerator()

        logger.info("Discussion initialized")

    async def _run_discussion_phases(self, document: Document):
        """Run through all discussion phases."""
        while not self._should_terminate_discussion():
            current_phase = self.discussion_state.current_phase

            try:
                if current_phase == DiscussionPhase.INITIAL_ANALYSIS:
                    await self._run_initial_analysis_phase(document)

                elif current_phase == DiscussionPhase.QUESTIONING:
                    await self._run_questioning_phase()

                elif current_phase == DiscussionPhase.RESPONDING:
                    await self._run_responding_phase()

                elif current_phase == DiscussionPhase.CONVERGENCE:
                    await self._run_convergence_phase()

                elif current_phase == DiscussionPhase.CONSENSUS:
                    await self._run_consensus_phase(document)

                # Update state
                self.discussion_state.last_activity = datetime.now(UTC)

                # Brief pause between phases
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in phase {current_phase}: {e}")
                # Continue to next phase
                await self._advance_phase()

    async def _run_initial_analysis_phase(self, document: Document):
        """Run initial analysis phase where all agents generate insights."""
        logger.info("Starting initial analysis phase")

        # Create insight generation tasks for all agents
        tasks = []
        for personality in AgentPersonality:
            task_id = self.task_manager.create_task(
                queue_name="main_discussion",
                task_type=TaskType.GENERATE_INSIGHTS,
                parameters={
                    "personality": personality,
                    "document": document,
                    "discussion_context": {
                        "phase": "initial_analysis",
                        "iteration": self.discussion_state.iteration_count,
                        "total_insights": sum(len(insights) for insights in self.agent_insights.values()),
                    },
                },
                priority=TaskPriority.HIGH,
                assigned_to=personality,
                timeout_seconds=180,  # 3 minutes
                context={"model_name": self.model_name},
            )
            tasks.append(task_id)

        # Wait for all insight generation tasks to complete
        await self._wait_for_tasks_completion(tasks, timeout_minutes=5)

        # Collect insights from these specific tasks only
        await self._collect_insights_from_tasks(tasks)

        # Update phase
        self.discussion_state.current_phase = DiscussionPhase.QUESTIONING
        self.discussion_state.iteration_count += 1

        logger.info(f"Initial analysis phase completed. Total insights: {sum(len(insights) for insights in self.agent_insights.values())}")

    async def _run_questioning_phase(self):
        """Run questioning phase (batch mode)."""
        logger.info("Starting questioning phase (batch)")

        question_tasks = []

        # All insights from all other agents
        all_other_insights = []
        for insights in self.agent_insights.values():
            # Sort and take top 3 insights per agent for questioning
            top_insights = sorted(insights, key=lambda i: i.importance_score, reverse=True)[:3]
            all_other_insights.extend(top_insights)

        for from_personality in AgentPersonality:
            # Insights NOT from this agent
            target_insights = [i for i in all_other_insights if i.agent_id != from_personality]

            if not target_insights:
                continue

            task_id = self.task_manager.create_task(
                queue_name="main_discussion",
                task_type=TaskType.ASK_QUESTION,
                parameters={
                    "from_agent": from_personality,
                    "target_insights": target_insights,
                    "discussion_context": {
                        "phase": "questioning",
                        "iteration": self.discussion_state.iteration_count,
                        "role_qa_summary": self._build_role_qa_context(from_personality),
                    },
                },
                priority=TaskPriority.MEDIUM,
                assigned_to=from_personality,
                timeout_seconds=240,  # Batch questioning takes longer
                context={"model_name": self.model_name},
            )
            question_tasks.append(task_id)

        # Wait for question generation to complete
        await self._wait_for_tasks_completion(question_tasks, timeout_minutes=5)

        # Collect questions and assign short IDs
        await self._collect_questions_from_tasks(question_tasks)

        # Update phase
        if self.all_questions:
            self.discussion_state.current_phase = DiscussionPhase.RESPONDING
        else:
            self.discussion_state.current_phase = DiscussionPhase.CONVERGENCE

        logger.info(f"Questioning phase completed. Questions generated: {len(self.all_questions)}")

    async def _run_responding_phase(self):
        """Run responding phase (batch mode)."""
        logger.info("Starting responding phase (batch)")

        response_tasks = []

        # Group unanswered questions by to_agent
        questions_by_agent: dict[AgentPersonality, list] = {p: [] for p in AgentPersonality}

        for question in self.all_questions:
            # Check if this question has been answered
            if not any(resp.question_id == question.question_id for resp in self.all_responses):
                to_agent = question.to_agent
                if isinstance(to_agent, str):
                    to_agent = AgentPersonality(to_agent)
                questions_by_agent[to_agent].append(question)

        for agent, questions in questions_by_agent.items():
            if not questions:
                continue

            task_id = self.task_manager.create_task(
                queue_name="main_discussion",
                task_type=TaskType.ANSWER_QUESTION,
                parameters={
                    "questions": questions,
                    "my_insights": self.agent_insights.get(agent, []),
                    "discussion_context": {
                        "phase": "responding",
                        "iteration": self.discussion_state.iteration_count,
                        "role_qa_summary": self._build_role_qa_context(agent),
                    },
                },
                priority=TaskPriority.HIGH,
                assigned_to=agent,
                timeout_seconds=300,  # Batch answering takes longer
                context={"model_name": self.model_name},
            )
            response_tasks.append(task_id)

        # Wait for response generation to complete
        await self._wait_for_tasks_completion(response_tasks, timeout_minutes=7)

        # Collect responses
        await self._collect_responses_from_tasks(response_tasks)

        # Update phase
        self.discussion_state.current_phase = DiscussionPhase.CONVERGENCE

        logger.info(f"Responding phase completed. Total responses: {len(self.all_responses)}")

    async def _run_convergence_phase(self):
        """Run convergence evaluation phase."""
        logger.info("Starting convergence phase")

        # Evaluate convergence from all agents
        convergence_tasks = []

        for personality in AgentPersonality:
            task_id = self.task_manager.create_task(
                queue_name="main_discussion",
                task_type=TaskType.MONITOR_CONVERGENCE,
                parameters={
                    "personality": personality,
                    "all_insights": [insight for insights in self.agent_insights.values() for insight in insights],
                    "all_questions": self.all_questions,
                    "all_responses": self.all_responses,
                    "discussion_context": {
                        "phase": "convergence",
                        "iteration": self.discussion_state.iteration_count,
                    },
                },
                priority=TaskPriority.MEDIUM,
                assigned_to=personality,
                timeout_seconds=120,  # 2 minutes
                context={"model_name": self.model_name},
            )
            convergence_tasks.append(task_id)

        # Wait for convergence evaluation to complete
        await self._wait_for_tasks_completion(convergence_tasks, timeout_minutes=3)

        # Evaluate overall convergence from this iteration's tasks only
        convergence_score = await self._calculate_overall_convergence(convergence_tasks)

        self.discussion_state.convergence_score = convergence_score

        if convergence_score >= self.convergence_threshold:
            logger.info(f"Convergence reached: {convergence_score:.2f}")
            self.discussion_state.current_phase = DiscussionPhase.CONSENSUS
        elif self.discussion_state.iteration_count >= self.max_iterations:
            logger.info("Max iterations reached, proceeding to consensus")
            self.discussion_state.current_phase = DiscussionPhase.CONSENSUS
        else:
            logger.info(f"Convergence not reached: {convergence_score:.2f}, continuing discussion")
            self.discussion_state.current_phase = DiscussionPhase.QUESTIONING
            self.discussion_state.iteration_count += 1

    async def _run_consensus_phase(self, document: Document):
        """Run final consensus building phase."""
        logger.info("Starting consensus phase")

        # Import here to avoid circular imports
        from .consensus import ConsensusBuilder

        ConsensusBuilder(self.model_name)
        self.discussion_state.current_phase = DiscussionPhase.COMPLETED

    def _should_terminate_discussion(self) -> bool:
        """Check if discussion should terminate."""
        if not self.discussion_state:
            return True

        # Check time limit
        if datetime.now(UTC) - self.discussion_state.start_time > self.max_discussion_time:
            logger.warning("Discussion time limit reached")
            return True

        # Only terminate if phase is COMPLETED
        if self.discussion_state.current_phase == DiscussionPhase.COMPLETED:
            return True

        return False

    async def _wait_for_tasks_completion(self, task_ids: list[str], timeout_minutes: int = 5):
        """Wait for specified tasks to complete."""
        if not task_ids:
            return

        start_time = datetime.now(UTC)
        timeout = timedelta(minutes=timeout_minutes)

        while datetime.now(UTC) - start_time < timeout:
            all_completed = True
            for task_id in task_ids:
                status = self.discussion_queue.get_task_status(task_id) if self.discussion_queue else None
                if status in [
                    TaskStatus.PENDING,
                    TaskStatus.ASSIGNED,
                    TaskStatus.IN_PROGRESS,
                ]:
                    all_completed = False
                    break

            if all_completed:
                break

            await asyncio.sleep(2.0)

        # Check for any failed tasks
        for task_id in task_ids:
            status = self.discussion_queue.get_task_status(task_id) if self.discussion_queue else None
            if status == TaskStatus.FAILED:
                logger.warning(f"Task {task_id} failed")

    async def _collect_insights_from_tasks(self, task_ids: list[str]):
        """Collect insights from specified tasks only."""
        if not self.discussion_queue:
            return

        for task_id in task_ids:
            task = self.discussion_queue.get_task(task_id)
            if task and task.task_type == TaskType.GENERATE_INSIGHTS and task.result:
                personality = task.result.metadata.get("personality")
                if personality:
                    personality_enum = AgentPersonality(personality)
                    self.agent_insights[personality_enum].extend(task.result.insights)

                    # Also add to discussion state
                    self.discussion_state.insights.extend(task.result.insights)

    async def _collect_responses_from_tasks(self, task_ids: list[str]):
        """Collect responses from specified tasks only."""
        if not self.discussion_queue:
            return

        for task_id in task_ids:
            task = self.discussion_queue.get_task(task_id)
            if task and task.task_type == TaskType.ANSWER_QUESTION and task.result:
                self.all_responses.extend(task.result.responses)

                # Also add to discussion state
                self.discussion_state.responses.extend(task.result.responses)

    def _build_role_qa_context(self, personality: AgentPersonality) -> str:
        """Build per-agent Q&A summary for context."""
        pers_value = personality.value if hasattr(personality, "value") else str(personality)

        # Questions asked BY this agent
        asked_by_me = [q for q in self.all_questions if q.from_agent == pers_value]
        # Questions directed TO this agent
        directed_to_me = [q for q in self.all_questions if q.to_agent == pers_value]

        lines = []

        if asked_by_me:
            lines.append("Questions you asked:")
            for q in asked_by_me:
                resp = next((r for r in self.all_responses if r.question_id == q.question_id), None)
                to_name = str(q.to_agent).replace("_", " ").title()
                status = f"Answered (stance: {resp.stance})" if resp else "Pending"
                lines.append(f"  [{q.question_id}] \u2192 {to_name}: \"{q.content[:100]}...\" ({status})")

        if directed_to_me:
            lines.append("\nQuestions directed at you:")
            for q in directed_to_me:
                resp = next((r for r in self.all_responses if r.question_id == q.question_id), None)
                from_name = str(q.from_agent).replace("_", " ").title()
                status = "Answered" if resp else "Pending - please answer"
                lines.append(f"  [{q.question_id}] From {from_name}: \"{q.content[:100]}...\" ({status})")

        return "\n".join(lines) if lines else "No prior Q&A involving you."

    async def _collect_questions_from_tasks(self, task_ids: list[str]):
        """Collect questions and assign short IDs."""
        if not self.discussion_queue:
            return

        for task_id in task_ids:
            task = self.discussion_queue.get_task(task_id)
            if task and task.task_type == TaskType.ASK_QUESTION and task.result:
                from_agent = task.assigned_to
                if isinstance(from_agent, str):
                    from_agent = AgentPersonality(from_agent)

                for q in task.result.questions:
                    # Assign short ID
                    q.question_id = self.question_id_gen.next_id(from_agent)
                    self.all_questions.append(q)
                    self.discussion_state.questions.append(q)

    async def _calculate_overall_convergence(self, task_ids: list[str]) -> float:
        """Calculate overall convergence score from given convergence tasks."""
        if not self.discussion_queue:
            return 0.5

        scores: list[float] = []

        for task_id in task_ids:
            task = self.discussion_queue.get_task(task_id)
            if task and task.task_type == TaskType.MONITOR_CONVERGENCE and task.result:
                score = task.result.metadata.get("convergence_score", 0.5)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5

    def _get_prior_qa_for_insight(self, insight) -> list[dict]:
        """Build Q&A thread for a specific insight by matching questions and responses."""
        insight_snippet = insight.content[:50]

        related_questions = [q for q in self.all_questions if q.target_insight == insight_snippet]

        qa_pairs = []
        for q in related_questions:
            matching_response = next((r for r in self.all_responses if r.question_id == q.question_id), None)
            qa_pairs.append(
                {
                    "from_agent": (q.from_agent if isinstance(q.from_agent, str) else q.from_agent.value),
                    "to_agent": (q.to_agent if isinstance(q.to_agent, str) else q.to_agent.value),
                    "question": q.content,
                    "response": (matching_response.content if matching_response else None),
                    "response_stance": (matching_response.stance if matching_response else None),
                }
            )

        return qa_pairs

    async def _build_final_result(self, document: Document) -> DiscussionResult:
        """Build final discussion result."""
        # Import here to avoid circular imports
        from .consensus import ConsensusBuilder

        consensus_builder = ConsensusBuilder(self.model_name)
        return await consensus_builder.build_consensus_result(
            document=document,
            discussion_state=self.discussion_state,
            agent_insights=self.agent_insights,
            questions=self.all_questions,
            responses=self.all_responses,
        )

    async def _advance_phase(self):
        """Advance to next phase for error recovery."""
        phase_order = [
            DiscussionPhase.INITIAL_ANALYSIS,
            DiscussionPhase.QUESTIONING,
            DiscussionPhase.RESPONDING,
            DiscussionPhase.CONVERGENCE,
            DiscussionPhase.CONSENSUS,
            DiscussionPhase.COMPLETED,
        ]

        current_idx = phase_order.index(self.discussion_state.current_phase)
        if current_idx < len(phase_order) - 1:
            self.discussion_state.current_phase = phase_order[current_idx + 1]

    def clear_agent_cache(self):
        """Clear all cached agent instances to free memory or reset discussion."""
        from .tools import clear_agent_cache

        clear_agent_cache()
        logger.info("DiscussionAgent: Cleared all cached agent instances")

    def get_agent_cache_status(self) -> dict:
        """Get current agent cache status for debugging."""
        from .tools import get_agent_cache_status

        status = get_agent_cache_status()
        logger.info(f"DiscussionAgent: Cache status - {status}")
        return status
