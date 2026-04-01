from unittest.mock import Mock
from unittest.mock import patch

from sciread.agent.discussion.agent import DiscussionAgent
from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import DiscussionPhase
from sciread.agent.discussion.models import DiscussionState
from sciread.agent.discussion.models import Question
from sciread.agent.discussion.models import Response


@patch("sciread.agent.discussion.agent.Agent")
@patch("sciread.agent.discussion.agent.get_model")
def test_format_question_log_entry_includes_content(mock_get_model, mock_agent):
    mock_get_model.return_value = Mock()
    mock_agent.return_value = Mock()
    agent = DiscussionAgent()
    question = Question(
        question_id="Q-CE-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="What evidence shows the agent swarm result is not just prompt tuning?",
        target_insight="INS-II-01",
        question_type="challenge",
        priority=0.91,
    )

    entry = agent._format_question_log_entry(question)

    assert "Q-CE-01" in entry
    assert "批判性评估者 -> 创新洞察者" in entry
    assert "challenge" in entry
    assert "What evidence shows the agent swarm result is not just prompt tuning?" in entry


@patch("sciread.agent.discussion.agent.Agent")
@patch("sciread.agent.discussion.agent.get_model")
def test_format_response_log_entry_includes_stance_and_question_content(mock_get_model, mock_agent):
    mock_get_model.return_value = Mock()
    mock_agent.return_value = Mock()
    agent = DiscussionAgent()
    question = Question(
        question_id="Q-II-02",
        from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="How would this hold up under deployment latency constraints?",
        target_insight="INS-PA-02",
        question_type="extension",
        priority=0.7,
    )
    response = Response(
        response_id="R-PA-01",
        question_id="Q-II-02",
        from_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="The paper only partially addresses latency; the strongest evidence is limited to controlled evaluation settings.",
        stance="clarify",
        confidence=0.82,
    )

    entry = agent._format_response_log_entry(response, question)

    assert "Q-II-02" in entry
    assert "实践应用者 -> 创新洞察者" in entry
    assert "clarify" in entry
    assert "Q: How would this hold up under deployment latency constraints?" in entry
    assert "A: The paper only partially addresses latency" in entry


@patch("sciread.agent.discussion.agent.Agent")
@patch("sciread.agent.discussion.agent.get_model")
def test_apply_revised_insights_updates_exact_target_insight(mock_get_model, mock_agent):
    mock_get_model.return_value = Mock()
    mock_agent.return_value = Mock()
    agent = DiscussionAgent()

    target_insight = AgentInsight(
        insight_id="INS-PA-02",
        agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
        content="Original deployment insight.",
        importance_score=0.82,
        confidence=0.71,
        supporting_evidence=["Initial evidence"],
    )
    untouched_insight = AgentInsight(
        insight_id="INS-PA-03",
        agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
        content="Separate operational insight.",
        importance_score=0.7,
        confidence=0.68,
    )

    agent.agent_insights = {
        AgentPersonality.CRITICAL_EVALUATOR: [],
        AgentPersonality.INNOVATIVE_INSIGHTER: [],
        AgentPersonality.PRACTICAL_APPLICATOR: [target_insight, untouched_insight],
        AgentPersonality.THEORETICAL_INTEGRATOR: [],
    }
    agent.discussion_state = DiscussionState(current_phase=DiscussionPhase.RESPONDING)
    agent.discussion_state.insights.extend([target_insight, untouched_insight])
    agent.all_questions = [
        Question(
            question_id="Q-II-02",
            from_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
            to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
            content="How does this survive real deployment latency?",
            target_insight="INS-PA-02",
            target_insight_id="INS-PA-02",
            question_type="challenge",
            priority=0.9,
        )
    ]

    response = Response(
        response_id="R-PA-01",
        question_id="Q-II-02",
        from_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="The original claim needs qualification.",
        stance="modify",
        revised_insight="Deployment feasibility depends on latency-sensitive orchestration overhead, so the claim should be limited to controlled environments.",
        confidence=0.86,
    )

    agent._apply_revised_insights([response])

    assert target_insight.content.startswith("Deployment feasibility depends on latency-sensitive orchestration overhead")
    assert target_insight.confidence == 0.86
    assert untouched_insight.content == "Separate operational insight."
    assert any(note.startswith("在 Q-II-02 之后修订：Original deployment insight.") for note in target_insight.supporting_evidence)
