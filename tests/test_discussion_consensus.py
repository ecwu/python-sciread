import asyncio
from types import SimpleNamespace

import pytest

from sciread.agent.discussion.consensus import ConsensusBuilder
from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import ConsensusPoint
from sciread.agent.discussion.models import DiscussionPhase
from sciread.agent.discussion.models import DiscussionState
from sciread.agent.discussion.models import Question
from sciread.agent.discussion.models import Response
from sciread.platform.logging import get_logger


def _make_builder() -> ConsensusBuilder:
    builder = ConsensusBuilder.__new__(ConsensusBuilder)
    builder.logger = get_logger(__name__)
    return builder


def test_extract_key_contributions_filters_critical_and_meta_statements():
    builder = _make_builder()

    top_insights = [
        AgentInsight(
            agent_id=AgentPersonality.CRITICAL_EVALUATOR,
            content='The evaluation of the novel "Agent Swarm" framework relies heavily on an "In-house Swarm Bench" benchmark, which lacks public validation, raising significant concerns about the objectivity of the reported gains.',
            importance_score=0.95,
            confidence=0.93,
        ),
        AgentInsight(
            agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
            content='The "Agent Swarm" framework introduces a novel theoretical model for multi-agent systems that explicitly decouples high-level coordination from low-level execution to solve complex tasks more robustly.',
            importance_score=0.92,
            confidence=0.9,
        ),
        AgentInsight(
            agent_id=AgentPersonality.THEORETICAL_INTEGRATOR,
            content='Analysis of Theoretical Contributions: The paper\'s core thesis of "joint optimization" from pretrained specialists is framed as a conceptual synthesis rather than a standalone contribution. Key contributions is omitted.',
            importance_score=0.88,
            confidence=0.84,
        ),
    ]

    consensus_points = [
        ConsensusPoint(
            topic="Methodology",
            content="The paper presents a modular orchestration architecture for coordinating specialized subagents with explicit role separation.",
            supporting_agents=[
                AgentPersonality.INNOVATIVE_INSIGHTER,
                AgentPersonality.PRACTICAL_APPLICATOR,
            ],
            strength=0.86,
            evidence=[],
        )
    ]

    contributions = asyncio.run(
        builder._extract_key_contributions(document=None, top_insights=top_insights, consensus_points=consensus_points)
    )

    assert len(contributions) == 2
    assert any("introduces a novel theoretical model" in contribution for contribution in contributions)
    assert any("presents a modular orchestration architecture" in contribution for contribution in contributions)
    assert all("lacks public validation" not in contribution for contribution in contributions)
    assert all("Key contributions is omitted" not in contribution for contribution in contributions)


def test_extract_key_contributions_uses_clean_fallback_when_no_explicit_contribution_keyword():
    builder = _make_builder()

    top_insights = [
        AgentInsight(
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
            content="The system enables role-specialized subagents to handle long-horizon workflows with clearer coordination boundaries.",
            importance_score=0.8,
            confidence=0.85,
        ),
        AgentInsight(
            agent_id=AgentPersonality.CRITICAL_EVALUATOR,
            content="The evaluation setup has several limitations that reduce confidence in the headline benchmark gains.",
            importance_score=0.79,
            confidence=0.8,
        ),
    ]

    contributions = asyncio.run(builder._extract_key_contributions(document=None, top_insights=top_insights, consensus_points=[]))

    assert contributions == [
        "The system enables role-specialized subagents to handle long-horizon workflows with clearer coordination boundaries."
    ]


def test_extract_key_contributions_preserves_full_sentence_without_ellipsis():
    builder = _make_builder()

    long_contribution = (
        'The "Agent Swarm" framework introduces a novel theoretical model for multi-agent systems that explicitly decouples '
        "high-level coordination from low-level execution, defines stable interfaces between orchestrators and subagents, "
        "and provides a general design pattern for scaling complex task decomposition without collapsing all reasoning into a single agent."
    )

    contributions = asyncio.run(
        builder._extract_key_contributions(
            document=None,
            top_insights=[
                AgentInsight(
                    agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
                    content=long_contribution,
                    importance_score=0.9,
                    confidence=0.9,
                )
            ],
            consensus_points=[],
        )
    )

    assert contributions == [long_contribution]
    assert not contributions[0].endswith("...")


def test_extract_top_insights_orders_by_importance_and_confidence() -> None:
    builder = _make_builder()
    low = AgentInsight(
        agent_id=AgentPersonality.CRITICAL_EVALUATOR,
        content="Low",
        importance_score=0.4,
        confidence=0.4,
    )
    high = AgentInsight(
        agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="High",
        importance_score=0.9,
        confidence=0.8,
    )

    result = builder._extract_top_insights(
        {
            AgentPersonality.CRITICAL_EVALUATOR: [low],
            AgentPersonality.INNOVATIVE_INSIGHTER: [high],
        }
    )

    assert [insight.content for insight in result] == ["High", "Low"]


def test_identify_divergent_views_uses_target_insight_id() -> None:
    builder = _make_builder()
    challenged = AgentInsight(
        insight_id="INS-PA-01",
        agent_id=AgentPersonality.PRACTICAL_APPLICATOR,
        content="The method is practical for deployment.",
        importance_score=0.8,
        confidence=0.7,
    )
    question = Question(
        question_id="Q-CE-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        to_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="Does it really hold under deployment latency constraints?",
        target_insight="The method is practical for deployment.",
        target_insight_id="INS-PA-01",
        question_type="challenge",
        priority=0.9,
    )
    response = Response(
        response_id="R-PA-01",
        question_id="Q-CE-01",
        from_agent=AgentPersonality.PRACTICAL_APPLICATOR,
        content="The claim only holds in controlled environments.",
        stance="modify",
        confidence=0.75,
    )

    views = asyncio.run(
        builder._identify_divergent_views(
            {AgentPersonality.PRACTICAL_APPLICATOR: [challenged]},
            [question],
            [response],
        )
    )

    assert len(views) == 1
    assert views[0].holding_agent == AgentPersonality.PRACTICAL_APPLICATOR
    assert views[0].content == challenged.content
    assert views[0].reasoning == "The claim only holds in controlled environments."


def test_generate_summary_and_significance_parses_labeled_output() -> None:
    builder = _make_builder()

    class FakeAgent:
        async def run(self, prompt: str):
            return SimpleNamespace(output="SUMMARY:\n中文摘要。\n\nSIGNIFICANCE:\n整体意义。")

    builder.agent = FakeAgent()
    summary, significance = asyncio.run(
        builder._generate_summary_and_significance(
            document=SimpleNamespace(metadata=SimpleNamespace(title="Paper")),
            top_insights=[],
            consensus_points=[],
            divergent_views=[],
        )
    )

    assert summary == "中文摘要。"
    assert significance == "整体意义。"


def test_build_consensus_result_assembles_metadata_without_real_llm() -> None:
    builder = _make_builder()
    builder._extract_top_insights = lambda agent_insights: [
        AgentInsight(
            agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
            content="Top insight",
            importance_score=0.9,
            confidence=0.85,
        )
    ]

    async def fake_consensus_points(agent_insights, questions, responses):
        return [
            ConsensusPoint(
                topic="方法",
                content="Agents agree on the method.",
                supporting_agents=[AgentPersonality.INNOVATIVE_INSIGHTER],
                strength=0.8,
            )
        ]

    async def fake_divergent_views(agent_insights, questions, responses):
        return []

    async def fake_summary(document, top_insights, consensus_points, divergent_views):
        return "summary", "significance"

    async def fake_contributions(document, top_insights, consensus_points):
        return ["Contribution"]

    builder._identify_consensus_points = fake_consensus_points
    builder._identify_divergent_views = fake_divergent_views
    builder._generate_summary_and_significance = fake_summary
    builder._extract_key_contributions = fake_contributions

    result = asyncio.run(
        builder.build_consensus_result(
            document=SimpleNamespace(metadata=SimpleNamespace(title="Paper")),
            discussion_state=DiscussionState(current_phase=DiscussionPhase.CONSENSUS, iteration_count=2, convergence_score=0.8),
            agent_insights={AgentPersonality.INNOVATIVE_INSIGHTER: []},
            questions=[],
            responses=[],
        )
    )

    assert result.document_title == "Paper"
    assert result.summary == "summary"
    assert result.significance == "significance"
    assert result.key_contributions == ["Contribution"]
    assert result.discussion_metadata["total_iterations"] == 2
    assert result.discussion_metadata["convergence_score"] == 0.8


def test_extract_top_insights_respects_limit() -> None:
    """Only the top-N insights should be returned."""
    builder = _make_builder()
    insights = [
        AgentInsight(
            agent_id=AgentPersonality.INNOVATIVE_INSIGHTER, content=f"Insight {i}", importance_score=0.9 - i * 0.05, confidence=0.8
        )
        for i in range(15)
    ]
    result = builder._extract_top_insights({AgentPersonality.INNOVATIVE_INSIGHTER: insights})
    assert len(result) == 10
    assert result[0].content == "Insight 0"


def test_group_insights_by_topic() -> None:
    """Topic grouping should bucket insights by keyword domain."""
    builder = _make_builder()
    insights = {
        AgentPersonality.INNOVATIVE_INSIGHTER: [
            AgentInsight(
                agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
                content="The method improves accuracy.",
                importance_score=0.8,
                confidence=0.8,
            ),
            AgentInsight(
                agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
                content="A novel framework is proposed.",
                importance_score=0.8,
                confidence=0.8,
            ),
        ],
        AgentPersonality.CRITICAL_EVALUATOR: [
            AgentInsight(
                agent_id=AgentPersonality.CRITICAL_EVALUATOR,
                content="The result lacks reproducibility.",
                importance_score=0.8,
                confidence=0.8,
            ),
        ],
    }
    groups = builder._group_insights_by_topic(insights)
    assert "方法" in groups
    assert "理论贡献" in groups
    assert "结果" in groups


def test_extract_topic_from_insight() -> None:
    """Topic extraction should map content keywords to canonical topics."""
    builder = _make_builder()
    assert builder._extract_topic_from_insight("The method is scalable.") == "方法"
    assert builder._extract_topic_from_insight("Results show gains.") == "结果"
    assert builder._extract_topic_from_insight("A limitation is noise.") == "局限性"
    assert builder._extract_topic_from_insight("Something else entirely.") == "综合分析"


@pytest.mark.asyncio
async def test_evaluate_topic_consensus_requires_multiple_insights() -> None:
    """Consensus evaluation should require at least two insights."""
    builder = _make_builder()
    single = AgentInsight(agent_id=AgentPersonality.INNOVATIVE_INSIGHTER, content="Only one.", importance_score=0.8, confidence=0.8)
    result = await builder._evaluate_topic_consensus("方法", [single], [], [])
    assert result["has_consensus"] is False


@pytest.mark.asyncio
async def test_evaluate_topic_consensus_detects_conflict_majority() -> None:
    """If most insights are challenged, consensus should be rejected."""
    builder = _make_builder()
    insights = [
        AgentInsight(agent_id=AgentPersonality.INNOVATIVE_INSIGHTER, content="Method A is best.", importance_score=0.8, confidence=0.8),
        AgentInsight(agent_id=AgentPersonality.PRACTICAL_APPLICATOR, content="Method B is best.", importance_score=0.8, confidence=0.8),
    ]
    responses = [
        Response(
            response_id="R-01",
            question_id="Q-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="I disagree with the method claim.",
            stance="disagree",
            confidence=0.9,
        ),
        Response(
            response_id="R-02",
            question_id="Q-02",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="The method is still questionable.",
            stance="challenge",
            confidence=0.9,
        ),
    ]
    result = await builder._evaluate_topic_consensus("method", insights, [], responses)
    assert result["has_consensus"] is False


@pytest.mark.asyncio
async def test_evaluate_topic_consensus_builds_point() -> None:
    """Two supporting insights without strong challenges should form a consensus point."""
    builder = _make_builder()
    insights = [
        AgentInsight(agent_id=AgentPersonality.INNOVATIVE_INSIGHTER, content="Method A is best.", importance_score=0.8, confidence=0.9),
        AgentInsight(
            agent_id=AgentPersonality.PRACTICAL_APPLICATOR, content="Method A scales well.", importance_score=0.8, confidence=0.85
        ),
    ]
    result = await builder._evaluate_topic_consensus("方法", insights, [], [])
    assert result["has_consensus"] is True
    assert set(result["supporting_agents"]) == {
        AgentPersonality.INNOVATIVE_INSIGHTER.value,
        AgentPersonality.PRACTICAL_APPLICATOR.value,
    }
    assert result["strength"] == pytest.approx(0.875)


def test_calculate_overall_confidence() -> None:
    """Overall confidence should blend insight confidence and consensus strength."""
    builder = _make_builder()
    insights = [
        AgentInsight(agent_id=AgentPersonality.INNOVATIVE_INSIGHTER, content="A", importance_score=0.8, confidence=0.8),
        AgentInsight(agent_id=AgentPersonality.PRACTICAL_APPLICATOR, content="B", importance_score=0.8, confidence=0.6),
    ]
    consensus_points = [
        ConsensusPoint(topic="方法", content="Agree.", supporting_agents=[AgentPersonality.INNOVATIVE_INSIGHTER], strength=0.8)
    ]
    score = builder._calculate_overall_confidence(insights, consensus_points)
    expected = (0.7 * 0.7) + (0.3 * (0.8 / 2 / 2))
    assert score == pytest.approx(expected)


def test_calculate_overall_confidence_without_insights() -> None:
    builder = _make_builder()
    assert builder._calculate_overall_confidence([], []) == 0.0


def test_score_contribution_candidate_branches() -> None:
    """Contribution scoring should reward action/topic keywords and penalize negatives."""
    builder = _make_builder()
    assert builder._score_contribution_candidate("") == 0
    assert builder._score_contribution_candidate("This paper introduces a novel method.") > 0
    assert builder._score_contribution_candidate("The evaluation of baselines is omitted.") == 0
    assert builder._score_contribution_candidate("The method has limitations in public validation.") == 0


def test_is_viable_fallback_contribution() -> None:
    """Fallback contributions should be affirmative and mention concrete concepts."""
    builder = _make_builder()
    assert builder._is_viable_fallback_contribution("The system enables a new framework.") is True
    assert builder._is_viable_fallback_contribution("The method has limitations.") is False
    assert builder._is_viable_fallback_contribution("Random generic statement.") is False


def test_dedupe_contribution_candidate() -> None:
    """Near-identical contributions should collapse to the same normalized key."""
    builder = _make_builder()
    key1 = builder._dedupe_contribution_candidate("Introduces a novel Method...")
    key2 = builder._dedupe_contribution_candidate("introduces a novel method")
    assert key1 == key2


def test_identify_divergent_views_groups_by_topic() -> None:
    """Multiple challenges on the same topic should be grouped into one divergent view."""
    builder = _make_builder()
    insight = AgentInsight(
        insight_id="INS-01",
        agent_id=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="The proposed method is optimal.",
        importance_score=0.8,
        confidence=0.8,
    )
    question = Question(
        question_id="Q-01",
        from_agent=AgentPersonality.CRITICAL_EVALUATOR,
        to_agent=AgentPersonality.INNOVATIVE_INSIGHTER,
        content="Are you sure?",
        target_insight="The proposed method is optimal.",
        target_insight_id="INS-01",
        question_type="challenge",
        priority=0.9,
    )
    responses = [
        Response(
            response_id="R-01",
            question_id="Q-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="It fails on large inputs.",
            stance="challenge",
            confidence=0.8,
        ),
        Response(
            response_id="R-02",
            question_id="Q-01",
            from_agent=AgentPersonality.CRITICAL_EVALUATOR,
            content="It is also slow.",
            stance="challenge",
            confidence=0.7,
        ),
    ]

    views = asyncio.run(
        builder._identify_divergent_views(
            {AgentPersonality.INNOVATIVE_INSIGHTER: [insight]},
            [question],
            responses,
        )
    )

    assert len(views) == 1
    assert views[0].topic == "方法"
    assert views[0].holding_agent == AgentPersonality.INNOVATIVE_INSIGHTER
    assert len(views[0].counter_arguments) == 1


def test_build_consensus_result_returns_error_result_on_failure() -> None:
    """A catastrophic failure during consensus building should return a stable error result."""
    builder = _make_builder()
    builder._extract_top_insights = lambda _agent_insights: (_ for _ in ()).throw(RuntimeError("boom"))

    result = asyncio.run(
        builder.build_consensus_result(
            document=SimpleNamespace(metadata=SimpleNamespace(title="Paper")),
            discussion_state=None,
            agent_insights={},
            questions=[],
            responses=[],
        )
    )

    assert result.document_title == "Paper"
    assert result.confidence_score == 0.0
    assert result.significance == "分析失败"
    assert "boom" in result.summary
