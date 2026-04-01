import asyncio

from sciread.agent.discussion.consensus import ConsensusBuilder
from sciread.agent.discussion.models import AgentInsight
from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.models import ConsensusPoint
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
