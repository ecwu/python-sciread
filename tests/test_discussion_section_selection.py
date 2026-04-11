"""Tests for discussion-agent section selection heuristics."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import Question
from sciread.agent.discussion.models import Response
from sciread.agent.discussion.personalities import PersonalityAgent
from sciread.platform.logging import get_logger


def _make_personality_agent(personality: AgentPersonality) -> PersonalityAgent:
    """Create a lightweight PersonalityAgent instance for unit tests."""
    agent = PersonalityAgent.__new__(PersonalityAgent)
    agent.personality = personality
    agent.logger = get_logger(__name__)
    agent.message_history = []
    agent._display_name = PersonalityAgent._display_name.__get__(agent, PersonalityAgent)
    return agent


def test_discussion_default_sections_prefer_richer_child_sections() -> None:
    """Fallback selection should favor substantive child sections over heading-only parents."""
    agent = _make_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)

    available_sections = [
        "Abstract",
        "3. Method",
        "3.1 Proposed Method",
        "4. Experiments",
    ]
    section_lengths = {
        "Abstract": 240,
        "3. Method": 18,
        "3.1 Proposed Method": 820,
        "4. Experiments": 730,
    }

    selected = agent._get_default_sections(available_sections, section_lengths)

    assert "3.1 Proposed Method" in selected
    assert "3. Method" not in selected


def test_parse_insights_uses_selected_sections_as_related_sections() -> None:
    """Parsed insights should preserve the sections the personality agent actually chose to read."""
    agent = _make_personality_agent(AgentPersonality.THEORETICAL_INTEGRATOR)

    class DummyDocument:
        def get_section_names(self) -> list[str]:
            return ["Abstract", "Introduction", "Conclusion"]

    insights = agent._parse_insights_response(
        "Insight: The method improves reasoning consistency.\nImportance: 0.82\nConfidence: 0.77\nEvidence: Supported by the evaluation section.",
        DummyDocument(),
        ["3.1 Proposed Method", "4. Experiments"],
    )

    assert len(insights) == 1
    assert insights[0].related_sections == ["3.1 Proposed Method", "4. Experiments"]


@pytest.mark.asyncio
async def test_discussion_section_selector_accepts_model_output_with_length_annotations() -> None:
    """Section parsing should tolerate model outputs that copy the prompt annotations."""
    agent = _make_personality_agent(AgentPersonality.INNOVATIVE_INSIGHTER)

    async def _fake_run_with_history(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(output="1. 3.1 Proposed Method | 820 chars\n2. 5. Future Work | 410 chars")

    agent._run_with_history = _fake_run_with_history

    selected = await agent._select_sections_to_read(
        title="Test Paper",
        abstract="Paper abstract",
        available_sections=["3. Method", "3.1 Proposed Method", "5. Future Work"],
        section_lengths={"3. Method": 18, "3.1 Proposed Method": 820, "5. Future Work": 410},
    )

    assert selected == ["3.1 Proposed Method", "5. Future Work"]


def test_parse_batch_questions_response_keeps_only_real_questions() -> None:
    agent = _make_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)
    target_insights = [
        AgentInsight(
            insight_id="INS-II-01",
            agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="The planner is the core novelty.",
            importance_score=0.9,
            confidence=0.8,
        ),
        AgentInsight(
            insight_id="INS-PA-01",
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
            content="Deployment is feasible.",
            importance_score=0.8,
            confidence=0.7,
        ),
    ]

    questions = agent._parse_batch_questions_response(
        """
---
Question about [INS-II-01]:
Decision: ask
Reason: 需要更多证据
Question: 哪个消融实验真正隔离了规划器贡献？
Priority: 0.9
Type: challenge
---
Question about [INS-PA-01]:
Decision: skip
Reason: 当前没有新增疑问
Question: None
Priority: 0.0
Type: none
---
Question about [INS-XX-99]:
Decision: ask
Reason: 无法定位
Question: 这条不该被保留
Priority: 0.8
Type: clarification
---
""",
        target_insights,
    )

    assert len(questions) == 1
    assert questions[0].target_insight_id == "INS-II-01"
    assert questions[0].content == "哪个消融实验真正隔离了规划器贡献？"


def test_parse_batch_responses_response_ignores_unknown_question_ids() -> None:
    agent = _make_personality_agent(AgentPersonality.INNOVATIVE_INSIGHTER)
    questions = [
        Question(
            question_id="Q-CE-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="What ablation isolates the planner?",
            target_insight="INS-II-01",
            target_insight_id="INS-II-01",
            question_type="challenge",
            priority=0.9,
        )
    ]

    responses = agent._parse_batch_responses_response(
        """
---
Answer to [Q-CE-01]:
Response: 消融实验移除了规划器阶段，因此能隔离它的贡献。
Stance: clarify
Revised Insight: None
Confidence: 0.86
---
Answer to [Q-PA-99]:
Response: 这条不应被保留。
Stance: disagree
Revised Insight: None
Confidence: 0.2
---
""",
        questions,
    )

    assert len(responses) == 1
    assert responses[0].question_id == "Q-CE-01"
    assert responses[0].stance == "clarify"


@pytest.mark.asyncio
async def test_discussion_section_selector_falls_back_when_model_returns_too_few_matches() -> None:
    agent = _make_personality_agent(AgentPersonality.PRACTICAL_APPLICATOR)

    async def _fake_run_with_history(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(output="Appendix A")

    agent._run_with_history = _fake_run_with_history

    selected = await agent._select_sections_to_read(
        title="Test Paper",
        abstract="Paper abstract",
        available_sections=["Abstract", "Introduction", "Experiments", "Applications"],
        section_lengths={"Abstract": 220, "Introduction": 600, "Experiments": 740, "Applications": 510},
    )

    assert selected == ["Abstract", "Introduction", "Applications", "Experiments"]


def test_get_section_content_backfills_missing_requested_sections_with_defaults() -> None:
    agent = _make_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)
    document = SimpleNamespace(get_section_names=lambda: ["Abstract", "Method", "Experiments"])

    with (
        patch(
            "sciread.agent.discussion.personalities.get_sections_content",
            side_effect=[
                [("Method", "Method content")],
                [("Abstract", "Abstract content"), ("Experiments", "Experiment content")],
            ],
        ) as mock_get_sections_content,
        patch(
            "sciread.agent.discussion.personalities.get_section_length_map",
            return_value={"Abstract": 220, "Method": 680, "Experiments": 750},
        ),
    ):
        content = agent._get_section_content(document, ["Method", "Missing Section"])

    assert content == {
        "Method": "Method content",
        "Abstract": "Abstract content",
        "Experiments": "Experiment content",
    }
    assert mock_get_sections_content.call_count == 2


def test_personality_qa_thread_summary_and_answer_counts() -> None:
    agent = _make_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)
    questions = [
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
            question_id="Q-CE-02",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
            content="How realistic is the deployment setting?",
            target_insight="INS-PA-01",
            target_insight_id="INS-PA-01",
            question_type="extension",
            priority=0.8,
        ),
    ]
    responses = [
        Response(
            response_id="R-II-01",
            question_id="Q-CE-01",
            from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="The ablation removes the planner stage.",
            stance="clarify",
            confidence=0.84,
        )
    ]

    summary = agent._build_qa_thread_summary(questions, responses)
    counts = agent._count_my_answered_questions(questions, responses)

    assert "问（批判性评估者 → 创新洞察者）" in summary
    assert "答（立场：clarify）" in summary
    assert "答：（待回复）" in summary
    assert counts == {"total": 2, "answered": 1}


def test_parse_convergence_evaluation_reads_score_and_continue_flag() -> None:
    agent = _make_personality_agent(AgentPersonality.THEORETICAL_INTEGRATOR)

    evaluation = agent._parse_convergence_evaluation(
        """
Convergence Score: 0.78
Continue Discussion: no
Key Issues Remaining: 主要分歧已消解
Recommendations: 进入总结阶段
"""
    )

    assert evaluation["convergence_score"] == 0.78
    assert evaluation["continue_discussion"] is False
    assert evaluation["key_issues"] == ["主要分歧已消解", "Recommendations: 进入总结阶段"]
    assert evaluation["recommendations"] == ["进入总结阶段"]


@pytest.mark.asyncio
async def test_ask_question_returns_none_for_skip_decision() -> None:
    agent = _make_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)
    insight = AgentInsight(
        insight_id="INS-II-01",
        agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="The planner is the core novelty.",
        importance_score=0.9,
        confidence=0.8,
        supporting_evidence=["Ablation evidence"],
    )

    async def _fake_run_with_history(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            output="""
Decision: skip
Reason: 当前没有新增疑问
Question: None
Priority: 0.0
Type: none
"""
        )

    agent._run_with_history = _fake_run_with_history

    result = await agent.ask_question(
        target_insight=insight,
        target_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        discussion_context={"phase": "questioning", "iteration": 2},
    )

    assert result == {
        "decision": "skip",
        "reason": "当前没有新增疑问",
        "question": None,
        "priority": 0.0,
        "type": "none",
    }


def test_find_relevant_insights_for_question_uses_overlap_and_evidence() -> None:
    agent = _make_personality_agent(AgentPersonality.PRACTICAL_APPLICATOR)
    insights = [
        AgentInsight(
            insight_id="INS-PA-01",
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
            content="Deployment latency overhead remains manageable under staged planner execution.",
            importance_score=0.8,
            confidence=0.75,
            supporting_evidence=["Latency overhead remains under 50ms in the deployment experiment."],
        ),
        AgentInsight(
            insight_id="INS-PA-02",
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
            content="The framework improves collaboration quality.",
            importance_score=0.6,
            confidence=0.7,
            supporting_evidence=["Generic collaboration evidence."],
        ),
    ]
    question = Question(
        question_id="Q-II-01",
        from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="How does deployment latency overhead behave under staged planner execution?",
        target_insight="INS-PA-01",
        target_insight_id="INS-PA-01",
        question_type="clarification",
        priority=0.8,
    )

    relevant = agent._find_relevant_insights_for_question(insights, question)

    assert [insight.insight_id for insight in relevant] == ["INS-PA-01"]


@pytest.mark.asyncio
async def test_answer_question_parses_structured_response_without_real_llm() -> None:
    agent = _make_personality_agent(AgentPersonality.PRACTICAL_APPLICATOR)
    question = Question(
        question_id="Q-II-01",
        from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="How does deployment latency overhead behave?",
        target_insight="INS-PA-01",
        target_insight_id="INS-PA-01",
        question_type="clarification",
        priority=0.8,
    )
    my_insights = [
        AgentInsight(
            insight_id="INS-PA-01",
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
            content="Deployment latency overhead remains manageable under staged planner execution.",
            importance_score=0.8,
            confidence=0.75,
            supporting_evidence=["Latency overhead remains under 50ms in the deployment experiment."],
        )
    ]

    async def _fake_run_with_history(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            output="""
Response: 延迟主要来自规划器阶段，但实验显示仍处于可接受范围。
Stance: clarify
Revised Insight: 部署延迟主要由规划器阶段带来，但在实验设置中仍可接受。
Confidence: 0.88
"""
        )

    agent._run_with_history = _fake_run_with_history

    response = await agent.answer_question(
        question=question,
        my_insights=my_insights,
        discussion_context={"phase": "responding"},
    )

    assert response is not None
    assert response.question_id == "Q-II-01"
    assert response.stance == "clarify"
    assert response.revised_insight == "部署延迟主要由规划器阶段带来，但在实验设置中仍可接受。"
    assert response.confidence == 0.88


@pytest.mark.asyncio
async def test_evaluate_convergence_parses_model_output_without_real_llm() -> None:
    agent = _make_personality_agent(AgentPersonality.THEORETICAL_INTEGRATOR)
    insight = AgentInsight(
        insight_id="INS-TI-01",
        agent_id=AgentPersonality.THEORETICAL_INTEGRATOR,
        content="The framework reconciles prior coordination theories.",
        importance_score=0.8,
        confidence=0.78,
    )
    question = Question(
        question_id="Q-TI-01",
        from_agent=AgentPersonality.THEORETICAL_INTEGRATOR,
        to_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="What theoretical assumption remains unsupported?",
        target_insight="INS-CE-01",
        target_insight_id="INS-CE-01",
        question_type="challenge",
        priority=0.7,
    )
    response = Response(
        response_id="R-CE-01",
        question_id="Q-TI-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        content="The independence assumption is unsupported.",
        stance="clarify",
        confidence=0.82,
    )

    async def _fake_run_with_history(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            output="""
Convergence Score: 0.83
Continue Discussion: no
Key Issues Remaining: 仅剩措辞统一
Recommendations: 进入共识总结
"""
        )

    agent._run_with_history = _fake_run_with_history

    evaluation = await agent.evaluate_convergence(
        all_insights=[insight],
        all_questions=[question],
        all_responses=[response],
        discussion_context={"iteration": 3},
    )

    assert evaluation["convergence_score"] == 0.83
    assert evaluation["continue_discussion"] is False
