from unittest.mock import Mock
from unittest.mock import patch

from sciread.agent.discussion.agent import DiscussionAgent
from sciread.agent.discussion.models import AgentPersonality
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
    assert "Critical Evaluator -> Innovative Insighter" in entry
    assert "challenge" in entry
    assert (
        "What evidence shows the agent swarm result is not just prompt tuning?" in entry
    )


@patch("sciread.agent.discussion.agent.Agent")
@patch("sciread.agent.discussion.agent.get_model")
def test_format_response_log_entry_includes_stance_and_question_content(
    mock_get_model, mock_agent
):
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
    assert "Practical Applicator -> Innovative Insighter" in entry
    assert "clarify" in entry
    assert "Q: How would this hold up under deployment latency constraints?" in entry
    assert "A: The paper only partially addresses latency" in entry
