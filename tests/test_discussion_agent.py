"""Offline control-flow tests for the discussion agent."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import pytest

from sciread.agent.discussion.agent import DiscussionAgent
from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import DiscussionPhase
from sciread.agent.discussion.models import DiscussionResult
from sciread.agent.discussion.models import DiscussionState
from sciread.agent.discussion.models import Question
from sciread.agent.discussion.models import QuestionIdGenerator
from sciread.agent.discussion.models import Response
from sciread.agent.discussion.task_models import Task
from sciread.agent.discussion.task_models import TaskPriority
from sciread.agent.discussion.task_models import TaskResult
from sciread.agent.discussion.task_models import TaskStatus
from sciread.agent.discussion.task_models import TaskType


class _FakeTaskManager:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.created_tasks: list[dict] = []
        self.queues: dict[str, _FakeQueue] = {}

    async def start_processing(self) -> None:
        self.started += 1

    async def stop_processing(self) -> None:
        self.stopped += 1

    def create_queue(self, name: str, description: str | None = None) -> "_FakeQueue":
        queue = _FakeQueue(name=name)
        self.queues[name] = queue
        return queue

    def create_task(self, **kwargs) -> str:
        task_id = f"task-{len(self.created_tasks) + 1}"
        task_record = {"task_id": task_id, **kwargs}
        self.created_tasks.append(task_record)

        queue_name = kwargs.get("queue_name", "main_discussion")
        queue = self.queues.get(queue_name)
        if queue is not None:
            from sciread.agent.discussion.task_models import Task
            from sciread.agent.discussion.task_models import TaskStatus

            task = Task(
                task_id=task_id,
                task_type=kwargs["task_type"],
                priority=kwargs.get("priority", TaskPriority.MEDIUM),
                assigned_to=kwargs.get("assigned_to"),
                parameters=kwargs.get("parameters", {}),
                context=kwargs.get("context", {}),
                timeout_seconds=kwargs.get("timeout_seconds"),
                status=TaskStatus.PENDING,
            )
            queue.tasks[task_id] = task

        return task_id


class _FakeQueue:
    def __init__(self, tasks: dict[str, Task] | None = None, name: str = "main_discussion") -> None:
        self.name = name
        self.tasks = tasks or {}

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def get_task_status(self, task_id: str) -> TaskStatus | None:
        task = self.tasks.get(task_id)
        return task.status if task else None


def _make_insight(
    insight_id: str,
    agent_id: AgentPersonality,
    content: str,
    importance: float = 0.8,
) -> AgentInsight:
    return AgentInsight(
        insight_id=insight_id,
        agent_id=agent_id,
        content=content,
        importance_score=importance,
        confidence=0.7,
    )


def _make_agent() -> DiscussionAgent:
    agent = DiscussionAgent.__new__(DiscussionAgent)
    agent.model_name = "mock-model"
    agent.max_iterations = 5
    agent.convergence_threshold = 0.75
    agent.max_discussion_time = timedelta(minutes=30)
    agent.task_manager = _FakeTaskManager()
    agent.discussion_queue = None
    agent.discussion_state = None
    agent.agent_insights = {}
    agent.all_questions = []
    agent.all_responses = []
    agent.question_id_gen = QuestionIdGenerator()
    return agent


@pytest.mark.asyncio
async def test_discussion_analyze_document_runs_full_offline_pipeline() -> None:
    """The top-level discussion entrypoint should orchestrate init, run, build, and stop."""
    agent = _make_agent()
    document = SimpleNamespace(metadata=SimpleNamespace(title="Paper Title"))
    calls: list[str] = []
    expected_result = DiscussionResult(
        document_title="Paper Title",
        summary="offline summary",
        key_contributions=[],
        significance="High",
        confidence_score=0.8,
        discussion_metadata={"mode": "offline"},
    )

    async def fake_initialize(_document) -> None:
        calls.append("initialize")

    async def fake_run_phases(_document) -> None:
        calls.append("run_phases")

    async def fake_build(_document):
        calls.append("build_result")
        return expected_result

    agent._initialize_discussion = fake_initialize
    agent._run_discussion_phases = fake_run_phases
    agent._build_final_result = fake_build

    result = await agent.analyze_document(document)

    assert result is expected_result
    assert calls == ["initialize", "run_phases", "build_result"]
    assert agent.task_manager.started == 1
    assert agent.task_manager.stopped == 1


@pytest.mark.asyncio
async def test_discussion_analyze_document_returns_error_result_without_real_llm() -> None:
    """Failures during orchestration should be converted into a stable DiscussionResult."""
    agent = _make_agent()
    document = SimpleNamespace(metadata=SimpleNamespace(title="Paper Title"))

    async def fake_initialize(_document) -> None:
        raise RuntimeError("phase failed")

    agent._initialize_discussion = fake_initialize

    result = await agent.analyze_document(document)

    assert result.document_title == "Paper Title"
    assert result.significance == "分析失败"
    assert result.confidence_score == 0.0
    assert result.discussion_metadata == {"error": "phase failed"}
    assert "phase failed" in result.summary
    assert agent.task_manager.started == 0
    assert agent.task_manager.stopped == 0


def test_discussion_should_terminate_for_timeout_and_completed_phase() -> None:
    """Termination logic should stop on both timeout and completed phase."""
    agent = _make_agent()
    agent.discussion_state = DiscussionState(
        current_phase=DiscussionPhase.QUESTIONING,
        start_time=datetime.now(UTC) - timedelta(hours=1),
    )

    assert agent._should_terminate_discussion() is True

    agent.discussion_state = DiscussionState(
        current_phase=DiscussionPhase.COMPLETED,
        start_time=datetime.now(UTC),
    )

    assert agent._should_terminate_discussion() is True

    agent.discussion_state = DiscussionState(
        current_phase=DiscussionPhase.QUESTIONING,
        start_time=datetime.now(UTC),
    )

    assert agent._should_terminate_discussion() is False


@pytest.mark.asyncio
async def test_run_questioning_phase_moves_to_responding_when_questions_exist() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.QUESTIONING, iteration_count=2)
    agent.agent_insights = {
        AgentPersonality.CRITICAL_EVALUATOR: [_make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Need stronger evaluation.")],
        AgentPersonality.INNOVATIVE_INSIGHTER: [
            _make_insight("INS-II-01", AgentPersonality.INNOVATIVE_INSIGHTER, "The novelty is the planner.")
        ],
        AgentPersonality.PRACTICAL_APPLICATOR: [
            _make_insight("INS-PA-01", AgentPersonality.PRACTICAL_APPLICATOR, "Deployment looks feasible.")
        ],
        AgentPersonality.THEORETICAL_INTEGRATOR: [
            _make_insight("INS-TI-01", AgentPersonality.THEORETICAL_INTEGRATOR, "The framework unifies prior work.")
        ],
    }

    async def fake_wait(_task_ids: list[str], timeout_minutes: int = 5) -> None:
        return None

    async def fake_collect(_task_ids: list[str]) -> None:
        agent.all_questions.append(
            Question(
                question_id="Q-CE-01",
                from_agent=AgentPersonality.CRITICAL_EVALUATOR,
                to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
                content="What evidence isolates the planner contribution?",
                target_insight="INS-II-01",
                target_insight_id="INS-II-01",
                question_type="challenge",
                priority=0.9,
            )
        )

    agent._wait_for_tasks_completion = fake_wait
    agent._collect_questions_from_tasks = fake_collect
    agent._render_phase_banner = lambda *args, **kwargs: None
    agent._render_question_summary = lambda *args, **kwargs: None

    await agent._run_questioning_phase()

    assert len(agent.task_manager.created_tasks) == len(AgentPersonality)
    assert agent.discussion_state.current_phase == DiscussionPhase.RESPONDING
    assert all(task["task_type"] == TaskType.ASK_QUESTION for task in agent.task_manager.created_tasks)
    assert all(task["parameters"]["discussion_context"]["phase"] == "questioning" for task in agent.task_manager.created_tasks)


@pytest.mark.asyncio
async def test_run_questioning_phase_skips_to_convergence_without_questions() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.QUESTIONING, iteration_count=1)
    agent.agent_insights = {
        AgentPersonality.CRITICAL_EVALUATOR: [_make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Evaluation is narrow.")],
        AgentPersonality.INNOVATIVE_INSIGHTER: [],
        AgentPersonality.PRACTICAL_APPLICATOR: [],
        AgentPersonality.THEORETICAL_INTEGRATOR: [],
    }

    async def fake_wait(_task_ids: list[str], timeout_minutes: int = 5) -> None:
        return None

    async def fake_collect(_task_ids: list[str]) -> None:
        return None

    agent._wait_for_tasks_completion = fake_wait
    agent._collect_questions_from_tasks = fake_collect
    agent._render_phase_banner = lambda *args, **kwargs: None
    agent._render_question_summary = lambda *args, **kwargs: None

    await agent._run_questioning_phase()

    assert agent.discussion_state.current_phase == DiscussionPhase.CONVERGENCE


@pytest.mark.asyncio
async def test_run_responding_phase_groups_only_unanswered_questions() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.RESPONDING, iteration_count=2)
    agent.agent_insights = {
        AgentPersonality.CRITICAL_EVALUATOR: [_make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Evaluation caveat.")],
        AgentPersonality.INNOVATIVE_INSIGHTER: [_make_insight("INS-II-01", AgentPersonality.INNOVATIVE_INSIGHTER, "Planner novelty.")],
        AgentPersonality.PRACTICAL_APPLICATOR: [_make_insight("INS-PA-01", AgentPersonality.PRACTICAL_APPLICATOR, "Deployment path.")],
        AgentPersonality.THEORETICAL_INTEGRATOR: [],
    }
    agent.all_questions = [
        Question(
            question_id="Q-CE-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="How isolated is the planner gain?",
            target_insight="INS-II-01",
            target_insight_id="INS-II-01",
            question_type="challenge",
            priority=0.9,
        ),
        Question(
            question_id="Q-II-01",
            from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
            content="What deployment evidence exists?",
            target_insight="INS-PA-01",
            target_insight_id="INS-PA-01",
            question_type="clarification",
            priority=0.8,
        ),
        Question(
            question_id="Q-PA-01",
            from_agent=AgentPersonality.PRACTICAL_APPLICATOR,
            to_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="Which baseline is missing?",
            target_insight="INS-CE-01",
            target_insight_id="INS-CE-01",
            question_type="extension",
            priority=0.7,
        ),
    ]
    answered = Response(
        response_id="R-CE-01",
        question_id="Q-PA-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="The public benchmark baseline is missing.",
        stance="clarify",
        confidence=0.8,
    )
    agent.all_responses = [answered]
    agent.discussion_state.responses.append(answered)

    async def fake_wait(_task_ids: list[str], timeout_minutes: int = 7) -> None:
        return None

    async def fake_collect(_task_ids: list[str]) -> None:
        agent.all_responses.append(
            Response(
                response_id="R-II-01",
                question_id="Q-CE-01",
                from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
                content="The ablation isolates planner contribution.",
                stance="clarify",
                confidence=0.86,
            )
        )

    agent._wait_for_tasks_completion = fake_wait
    agent._collect_responses_from_tasks = fake_collect
    agent._render_phase_banner = lambda *args, **kwargs: None

    await agent._run_responding_phase()

    assert len(agent.task_manager.created_tasks) == 2
    assert {task["assigned_to"] for task in agent.task_manager.created_tasks} == {
        AgentPersonality.INNOVATIVE_INSIGHTER,
        AgentPersonality.PRACTICAL_APPLICATOR,
    }
    assert agent.discussion_state.current_phase == DiscussionPhase.CONVERGENCE


@pytest.mark.asyncio
async def test_run_convergence_phase_advances_to_consensus_at_threshold() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.CONVERGENCE, iteration_count=2)
    agent.agent_insights = {
        personality: [_make_insight(f"INS-{personality.name[:2]}-01", personality, f"{personality.value} insight")]
        for personality in AgentPersonality
    }
    agent.all_questions = []
    agent.all_responses = []

    async def fake_wait(_task_ids: list[str], timeout_minutes: int = 3) -> None:
        return None

    async def fake_score(_task_ids: list[str]) -> float:
        return 0.82

    agent._wait_for_tasks_completion = fake_wait
    agent._calculate_overall_convergence = fake_score
    agent._render_phase_banner = lambda *args, **kwargs: None
    agent._render_convergence_summary = lambda *args, **kwargs: None

    await agent._run_convergence_phase()

    assert len(agent.task_manager.created_tasks) == len(AgentPersonality)
    assert agent.discussion_state.current_phase == DiscussionPhase.CONSENSUS
    assert agent.discussion_state.convergence_score == 0.82


@pytest.mark.asyncio
async def test_run_convergence_phase_loops_back_when_not_converged() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.CONVERGENCE, iteration_count=2)
    agent.agent_insights = {personality: [] for personality in AgentPersonality}
    agent.all_questions = []
    agent.all_responses = []

    async def fake_wait(_task_ids: list[str], timeout_minutes: int = 3) -> None:
        return None

    async def fake_score(_task_ids: list[str]) -> float:
        return 0.31

    agent._wait_for_tasks_completion = fake_wait
    agent._calculate_overall_convergence = fake_score
    agent._render_phase_banner = lambda *args, **kwargs: None
    agent._render_convergence_summary = lambda *args, **kwargs: None

    await agent._run_convergence_phase()

    assert agent.discussion_state.current_phase == DiscussionPhase.QUESTIONING
    assert agent.discussion_state.iteration_count == 3


@pytest.mark.asyncio
async def test_collect_questions_from_tasks_assigns_short_ids_and_updates_state() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.QUESTIONING)
    task = Task(
        task_id="task-1",
        task_type=TaskType.ASK_QUESTION,
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.COMPLETED,
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
        result=TaskResult(
            task_id="task-1",
            success=True,
            execution_time=0.1,
            questions=[
                Question(
                    question_id="PENDING",
                    from_agent=AgentPersonality.CRITICAL_EVALUATOR,
                    to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
                    content="What evidence supports the planner claim?",
                    target_insight="INS-II-01",
                    target_insight_id="INS-II-01",
                    question_type="challenge",
                    priority=0.9,
                ),
                Question(
                    question_id="PENDING",
                    from_agent=AgentPersonality.CRITICAL_EVALUATOR,
                    to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
                    content="How does this behave in deployment?",
                    target_insight="INS-PA-01",
                    target_insight_id="INS-PA-01",
                    question_type="extension",
                    priority=0.7,
                ),
            ],
            confidence=0.9,
        ),
    )
    agent.discussion_queue = _FakeQueue({"task-1": task})
    agent._log_question_batch = lambda *args, **kwargs: None

    await agent._collect_questions_from_tasks(["task-1"])

    assert [question.question_id for question in agent.all_questions] == ["Q-CE-01", "Q-CE-02"]
    assert [question.question_id for question in agent.discussion_state.questions] == ["Q-CE-01", "Q-CE-02"]


@pytest.mark.asyncio
async def test_calculate_overall_convergence_averages_task_scores() -> None:
    agent = _make_agent()
    agent.discussion_queue = _FakeQueue(
        {
            "task-1": Task(
                task_id="task-1",
                task_type=TaskType.MONITOR_CONVERGENCE,
                status=TaskStatus.COMPLETED,
                result=TaskResult(
                    task_id="task-1",
                    success=True,
                    execution_time=0.1,
                    confidence=0.8,
                    metadata={"convergence_score": 0.9},
                ),
            ),
            "task-2": Task(
                task_id="task-2",
                task_type=TaskType.MONITOR_CONVERGENCE,
                status=TaskStatus.COMPLETED,
                result=TaskResult(
                    task_id="task-2",
                    success=True,
                    execution_time=0.1,
                    confidence=0.8,
                    metadata={"convergence_score": 0.5},
                ),
            ),
        }
    )

    score = await agent._calculate_overall_convergence(["task-1", "task-2"])

    assert score == pytest.approx(0.7)


def test_discussion_role_context_and_prior_qa_track_answer_status() -> None:
    agent = _make_agent()
    tracked_insight = _make_insight("INS-II-01", AgentPersonality.INNOVATIVE_INSIGHTER, "Planner orchestration is the main novelty.")
    agent.all_questions = [
        Question(
            question_id="Q-CE-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="What ablation isolates planner value?",
            target_insight="INS-II-01",
            target_insight_id="INS-II-01",
            question_type="challenge",
            priority=0.9,
        ),
        Question(
            question_id="Q-PA-01",
            from_agent=AgentPersonality.PRACTICAL_APPLICATOR,
            to_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="Which benchmark is missing?",
            target_insight="INS-CE-01",
            target_insight_id="INS-CE-01",
            question_type="clarification",
            priority=0.6,
        ),
    ]
    agent.all_responses = [
        Response(
            response_id="R-II-01",
            question_id="Q-CE-01",
            from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="The ablation removes the planner stage.",
            stance="clarify",
            confidence=0.82,
        )
    ]

    context = agent._build_role_qa_context(AgentPersonality.CRITICAL_EVALUATOR)
    prior_qa = agent._get_prior_qa_for_insight(tracked_insight)

    assert "你提出的问题：" in context
    assert "已回答" in context
    assert "待回答" in context
    assert prior_qa == [
        {
            "from_agent": AgentPersonality.CRITICAL_EVALUATOR.value,
            "to_agent": AgentPersonality.INNOVATIVE_INSIGHTER.value,
            "question": "What ablation isolates planner value?",
            "response": "The ablation removes the planner stage.",
            "response_stance": "clarify",
        }
    ]


@pytest.mark.asyncio
async def test_advance_phase_moves_to_next_state() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.RESPONDING)

    await agent._advance_phase()

    assert agent.discussion_state.current_phase == DiscussionPhase.CONVERGENCE


@pytest.mark.asyncio
async def test_initialize_discussion_creates_state_and_queue() -> None:
    agent = _make_agent()
    document = SimpleNamespace(metadata=SimpleNamespace(title="Paper"))

    await agent._initialize_discussion(document)

    assert agent.discussion_state is not None
    assert agent.discussion_state.current_phase == DiscussionPhase.INITIAL_ANALYSIS
    assert agent.discussion_queue is not None
    assert agent.discussion_queue.name == "main_discussion"
    assert set(agent.agent_insights.keys()) == set(AgentPersonality)


@pytest.mark.asyncio
async def test_run_initial_analysis_phase_creates_insight_tasks_and_collects() -> None:
    agent = _make_agent()
    await agent._initialize_discussion(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))
    agent._render_phase_banner = lambda *args, **kwargs: None

    insight = _make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Key finding.")

    async def fake_wait(task_ids: list[str], timeout_minutes: int = 5) -> None:
        for task_id in task_ids:
            task = agent.discussion_queue.get_task(task_id)
            if task and str(task.task_type) == TaskType.GENERATE_INSIGHTS.value:
                personality = AgentPersonality(task.assigned_to)
                task.status = TaskStatus.COMPLETED
                task.result = TaskResult(
                    task_id=task_id,
                    success=True,
                    execution_time=0.1,
                    insights=[insight],
                    confidence=0.8,
                    metadata={"personality": personality.value, "selected_sections": ["abstract"]},
                )

    agent._wait_for_tasks_completion = fake_wait

    await agent._run_initial_analysis_phase(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))

    assert len(agent.task_manager.created_tasks) == len(AgentPersonality)
    assert all(task["task_type"] == TaskType.GENERATE_INSIGHTS for task in agent.task_manager.created_tasks)
    assert agent.discussion_state.current_phase == DiscussionPhase.QUESTIONING
    assert len(agent.discussion_state.insights) == len(AgentPersonality)


@pytest.mark.asyncio
async def test_wait_for_tasks_completion_handles_failed_tasks() -> None:
    agent = _make_agent()
    await agent._initialize_discussion(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))

    task_id = agent.task_manager.create_task(
        queue_name="main_discussion",
        task_type=TaskType.GENERATE_INSIGHTS,
        parameters={},
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
    )
    task = agent.discussion_queue.get_task(task_id)
    task.status = TaskStatus.FAILED

    async def fake_wait(task_ids: list[str], timeout_minutes: int = 5) -> None:
        return None

    agent._wait_for_tasks_completion = fake_wait
    await agent._wait_for_tasks_completion([task_id], timeout_minutes=0)

    assert task.status == TaskStatus.FAILED


def test_apply_revised_insights_updates_target_insight() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.RESPONDING)
    original = _make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Original claim.")
    agent.agent_insights[AgentPersonality.CRITICAL_EVALUATOR] = [original]
    agent.discussion_state.insights.append(original)

    question = Question(
        question_id="Q-01",
        from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        to_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="Can you clarify?",
        target_insight="INS-CE-01",
        target_insight_id="INS-CE-01",
        question_type="clarification",
        priority=0.8,
    )
    agent.all_questions.append(question)

    response = Response(
        response_id="R-01",
        question_id="Q-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="Revised claim.",
        stance="clarify",
        revised_insight="Revised claim with more detail.",
        confidence=0.9,
    )

    agent._apply_revised_insights([response])

    assert original.content == "Revised claim with more detail."
    assert original.confidence == 0.9
    assert any("Q-01" in note for note in original.supporting_evidence)


def test_find_insight_for_question_by_id_and_snippet() -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.RESPONDING)
    insight = _make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Target content here.")
    agent.agent_insights[AgentPersonality.CRITICAL_EVALUATOR] = [insight]

    by_id = SimpleNamespace(target_insight_id="INS-CE-01", target_insight="")
    assert agent._find_insight_for_question(by_id) is insight

    by_snippet = SimpleNamespace(target_insight_id=None, target_insight="Target content here.")
    assert agent._find_insight_for_question(by_snippet) is insight

    missing = SimpleNamespace(target_insight_id=None, target_insight="missing")
    assert agent._find_insight_for_question(missing) is None


def test_update_insight_from_response_skips_unchanged_content() -> None:
    agent = _make_agent()
    insight = _make_insight("INS-CE-01", AgentPersonality.CRITICAL_EVALUATOR, "Same content.")
    response = Response(
        response_id="R-01",
        question_id="Q-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="ok",
        stance="agree",
        revised_insight="Same content.",
        confidence=0.8,
    )

    agent._update_insight_from_response(insight, "Same content.", response)

    assert len(insight.supporting_evidence) == 0


@pytest.mark.asyncio
async def test_build_final_result_delegates_to_consensus_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent()
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.CONSENSUS)
    expected_result = DiscussionResult(
        document_title="Paper",
        summary="summary",
        significance="High",
        confidence_score=0.8,
    )

    class FakeConsensusBuilder:
        def __init__(self, model_name: str):
            pass

        async def build_consensus_result(self, **kwargs):
            return expected_result

    monkeypatch.setattr(
        "sciread.agent.discussion.consensus.ConsensusBuilder",
        FakeConsensusBuilder,
    )

    result = await agent._build_final_result(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))
    assert result is expected_result


@pytest.mark.asyncio
async def test_run_discussion_phases_advances_through_pipeline() -> None:
    agent = _make_agent()
    await agent._initialize_discussion(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))
    agent._render_phase_banner = lambda *args, **kwargs: None
    phase_log: list[str] = []

    async def fake_initial(_document) -> None:
        phase_log.append("initial")
        agent.discussion_state.current_phase = DiscussionPhase.QUESTIONING

    async def fake_questioning() -> None:
        phase_log.append("questioning")
        agent.discussion_state.current_phase = DiscussionPhase.RESPONDING

    async def fake_responding() -> None:
        phase_log.append("responding")
        agent.discussion_state.current_phase = DiscussionPhase.CONVERGENCE

    async def fake_convergence() -> None:
        phase_log.append("convergence")
        agent.discussion_state.current_phase = DiscussionPhase.CONSENSUS

    async def fake_consensus(_document) -> None:
        phase_log.append("consensus")
        agent.discussion_state.current_phase = DiscussionPhase.COMPLETED

    agent._run_initial_analysis_phase = fake_initial
    agent._run_questioning_phase = fake_questioning
    agent._run_responding_phase = fake_responding
    agent._run_convergence_phase = fake_convergence
    agent._run_consensus_phase = fake_consensus

    await agent._run_discussion_phases(SimpleNamespace(metadata=SimpleNamespace(title="Paper")))

    assert phase_log == ["initial", "questioning", "responding", "convergence", "consensus"]
    assert agent.discussion_state.current_phase == DiscussionPhase.COMPLETED


def test_clear_agent_cache_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent()
    cache_state = {"cleared": False}

    def fake_clear():
        cache_state["cleared"] = True

    def fake_status():
        return {"size": 0}

    monkeypatch.setattr("sciread.agent.discussion.tools.clear_agent_cache", fake_clear)
    monkeypatch.setattr("sciread.agent.discussion.tools.get_agent_cache_status", fake_status)

    agent.clear_agent_cache()
    assert cache_state["cleared"] is True

    status = agent.get_agent_cache_status()
    assert status == {"size": 0}
