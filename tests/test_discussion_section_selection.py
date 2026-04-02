"""Tests for discussion-agent section selection heuristics."""

from types import SimpleNamespace

import pytest

from sciread.agent.discussion.models import AgentPersonality
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
