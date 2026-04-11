"""Tests for discussion task tools without real LLM calls."""

from types import SimpleNamespace

import pytest

from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import Question
from sciread.agent.discussion.models import Response
from sciread.agent.discussion.task_models import Task
from sciread.agent.discussion.task_models import TaskType
from sciread.agent.discussion.tools import task_tools


def _build_insight() -> AgentInsight:
    return AgentInsight(
        insight_id="INS-CE-01",
        agent_id=AgentPersonality.CRITICAL_EVALUATOR,
        content="The sample size is limited.",
        importance_score=0.8,
        confidence=0.6,
        supporting_evidence=["n=20"],
        related_sections=["Methods"],
    )


def _build_question() -> Question:
    return Question(
        question_id="Q-CE-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="How strong is the novelty claim?",
        target_insight="INS-CE-01",
        question_type="challenge",
        priority=0.7,
    )


def _build_response() -> Response:
    return Response(
        response_id="R-01",
        question_id="Q-CE-01",
        from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="The combination of methods is new.",
        stance="clarify",
        confidence=0.75,
    )


class FakePersonalityAgent:
    """Small async test double for task tools."""

    def __init__(self) -> None:
        self.last_selected_sections = ["Methods", "Results"]

    async def generate_insights(self, _document: object, _discussion_context: dict[str, object]) -> list[AgentInsight]:
        return [_build_insight()]

    async def ask_questions_batch(self, _target_insights: list[object], _discussion_context: dict[str, object]) -> list[Question]:
        return [_build_question()]

    async def answer_questions_batch(
        self,
        _questions: list[Question],
        _my_insights: list[AgentInsight],
        _discussion_context: dict[str, object],
    ) -> list[Response]:
        return [_build_response()]

    async def evaluate_convergence(
        self,
        _all_insights: list[AgentInsight],
        _all_questions: list[Question],
        _all_responses: list[Response],
        _discussion_context: dict[str, object],
    ) -> dict[str, object]:
        return {
            "convergence_score": 0.82,
            "continue_discussion": False,
            "key_issues": ["Need more external validation"],
            "recommendations": ["Draft the final summary"],
        }


def test_get_cached_agent_reuses_instances_and_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent cache should be keyed by personality and model name."""
    created: list[tuple[AgentPersonality, str]] = []

    def fake_create(personality: AgentPersonality, model_name: str) -> FakePersonalityAgent:
        created.append((personality, model_name))
        return FakePersonalityAgent()

    task_tools.clear_agent_cache()
    monkeypatch.setattr(task_tools, "create_personality_agent", fake_create)

    first = task_tools.get_cached_agent(AgentPersonality.CRITICAL_EVALUATOR, "model-a")
    second = task_tools.get_cached_agent(AgentPersonality.CRITICAL_EVALUATOR, "model-a")
    third = task_tools.get_cached_agent(AgentPersonality.CRITICAL_EVALUATOR, "model-b")

    assert first is second
    assert third is not first
    assert created == [
        (AgentPersonality.CRITICAL_EVALUATOR, "model-a"),
        (AgentPersonality.CRITICAL_EVALUATOR, "model-b"),
    ]
    assert task_tools.get_agent_cache_status()["cache_size"] == 2

    task_tools.clear_agent_cache()
    assert task_tools.get_agent_cache_status()["cache_size"] == 0

    with pytest.raises(ValueError, match="Personality cannot be None"):
        task_tools.get_cached_agent(None)


@pytest.mark.asyncio
async def test_generate_insights_tool_builds_success_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insight generation should convert strings to enums and summarize output metadata."""
    monkeypatch.setattr(task_tools, "get_cached_agent", lambda *_args, **_kwargs: FakePersonalityAgent())

    task = Task(
        task_id="generate-1",
        task_type=TaskType.GENERATE_INSIGHTS,
        parameters={
            "personality": AgentPersonality.CRITICAL_EVALUATOR.value,
            "document": SimpleNamespace(title="paper"),
            "discussion_context": {"phase": "initial"},
        },
        context={"model_name": "mock-model"},
    )

    result = await task_tools.generate_insights_tool(task)

    assert result.success is True
    assert len(result.insights) == 1
    assert result.confidence == pytest.approx(0.6)
    assert result.metadata["personality"] == AgentPersonality.CRITICAL_EVALUATOR.value
    assert result.metadata["selected_sections"] == ["Methods", "Results"]


@pytest.mark.asyncio
async def test_generate_insights_tool_returns_failure_result_for_missing_parameters() -> None:
    """Missing required inputs should not escape as uncaught exceptions."""
    task = Task(
        task_id="generate-2",
        task_type=TaskType.GENERATE_INSIGHTS,
        parameters={"document": object()},
    )

    result = await task_tools.generate_insights_tool(task)

    assert result.success is False
    assert result.confidence == 0.0
    assert result.metadata["personality"] == "unknown"


@pytest.mark.asyncio
async def test_question_and_answer_tools_support_batch_and_single_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Question and answer tools should support both batch inputs and legacy single-item inputs."""
    monkeypatch.setattr(task_tools, "get_cached_agent", lambda *_args, **_kwargs: FakePersonalityAgent())
    insight = _build_insight()
    question = _build_question()

    ask_task = Task(
        task_id="ask-1",
        task_type=TaskType.ASK_QUESTION,
        parameters={
            "from_agent": AgentPersonality.CRITICAL_EVALUATOR.value,
            "target_insight": insight,
        },
    )
    ask_result = await task_tools.ask_question_tool(ask_task)

    assert ask_result.success is True
    assert len(ask_result.questions) == 1
    assert ask_result.metadata["from_agent"] == AgentPersonality.CRITICAL_EVALUATOR.value

    answer_task = Task(
        task_id="answer-1",
        task_type=TaskType.ANSWER_QUESTION,
        parameters={
            "question": question,
            "my_insights": [insight],
        },
    )
    answer_result = await task_tools.answer_question_tool(answer_task)

    assert answer_result.success is True
    assert len(answer_result.responses) == 1
    assert answer_result.metadata["personality"] == AgentPersonality.INNOVATIVE_INSIGHTER.value


@pytest.mark.asyncio
async def test_question_and_answer_tools_return_failure_results_on_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid task payloads should become failure TaskResults."""
    monkeypatch.setattr(task_tools, "get_cached_agent", lambda *_args, **_kwargs: FakePersonalityAgent())

    ask_result = await task_tools.ask_question_tool(Task(task_id="ask-2", task_type=TaskType.ASK_QUESTION, parameters={}))
    answer_result = await task_tools.answer_question_tool(Task(task_id="answer-2", task_type=TaskType.ANSWER_QUESTION, parameters={}))

    assert ask_result.success is False
    assert ask_result.metadata["from_agent"] == "unknown"
    assert answer_result.success is False
    assert answer_result.metadata["personality"] == "unknown"


@pytest.mark.asyncio
async def test_evaluate_convergence_tool_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convergence evaluation should expose the model result and registry lookups."""
    monkeypatch.setattr(task_tools, "get_cached_agent", lambda *_args, **_kwargs: FakePersonalityAgent())

    task = Task(
        task_id="eval-1",
        task_type=TaskType.MONITOR_CONVERGENCE,
        parameters={"personality": AgentPersonality.THEORETICAL_INTEGRATOR.value},
    )
    result = await task_tools.evaluate_convergence_tool(task)

    assert result.success is True
    assert result.confidence == pytest.approx(0.82)
    assert "收敛评分：0.82" == result.analysis_result
    assert result.metadata["continue_discussion"] is False
    assert task_tools.get_task_tool(TaskType.MONITOR_CONVERGENCE) is task_tools.evaluate_convergence_tool
    assert task_tools.get_task_tool(TaskType.SYNTHESIZE_REPORT) is None

    failed = await task_tools.evaluate_convergence_tool(Task(task_id="eval-2", task_type=TaskType.MONITOR_CONVERGENCE, parameters={}))
    assert failed.success is False
    assert failed.metadata["personality"] == "unknown"
