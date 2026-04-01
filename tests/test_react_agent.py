"""Tests for the ReAct agent helpers and state management."""

from sciread.agent.react.agent import normalize_section_names
from sciread.agent.react.models import ReActAnalysisState
from sciread.agent.react.models import ReActIterationDeps
from sciread.agent.react.models import ReActIterationInput
from sciread.agent.react.models import ReActIterationOutput
from sciread.agent.react.models import ReActIterationState
from sciread.agent.react.prompts import build_iteration_system_prompt


def test_normalize_section_names_handles_common_input_shapes() -> None:
    """Section-name normalization should accept list, JSON string, and plain string inputs."""
    assert normalize_section_names([" Abstract ", "Methods", "Methods", 123]) == ["Abstract", "Methods"]
    assert normalize_section_names('["Introduction", " Results "]') == ["Introduction", "Results"]
    assert normalize_section_names("Conclusion") == ["Conclusion"]
    assert normalize_section_names("") is None


def test_build_iteration_system_prompt_switches_to_final_iteration() -> None:
    """The system prompt should encode different instructions for regular and final loops."""
    regular_prompt = build_iteration_system_prompt(
        ReActIterationDeps(
            document=None,
            task="Summarize the paper",
            iteration_input=ReActIterationInput(
                task="Summarize the paper",
                previous_thoughts="",
                processed_sections=[],
                available_sections=["Abstract", "Methods"],
            ),
            current_loop=1,
            max_loops=3,
        )
    )
    final_prompt = build_iteration_system_prompt(
        ReActIterationDeps(
            document=None,
            task="Summarize the paper",
            iteration_input=ReActIterationInput(
                task="Summarize the paper",
                previous_thoughts="Need to synthesize findings.",
                processed_sections=["Abstract"],
                available_sections=["Abstract", "Methods"],
            ),
            current_loop=3,
            max_loops=3,
        )
    )

    assert "=== 首轮迭代：先制定阅读策略 ===" in regular_prompt
    assert "必须且仅能调用一次 read_section()" in regular_prompt
    assert "=== 最终迭代（3/3）——仅做综合 ===" in final_prompt
    assert "不要调用 read_section()" in final_prompt


def test_react_analysis_state_accumulates_sections_memory_and_report() -> None:
    """Session state should deduplicate sections and prefer the final structured report."""
    analysis_state = ReActAnalysisState(
        task="What are the main contributions?",
        available_sections=["Abstract", "Methods", "Results"],
    )

    analysis_state.apply_iteration(
        ReActIterationOutput(thoughts="Read methods next.", should_continue=True),
        ReActIterationState(
            sections_read=["Abstract", "Methods", "Abstract"],
            memory_text="- [CLAIM] Strong baseline.\n- [RESULT] +3.2 points.",
        ),
    )
    analysis_state.apply_iteration(
        ReActIterationOutput(
            thoughts="Analysis complete.",
            should_continue=False,
            report="Structured final report.",
        ),
        ReActIterationState(sections_read=["Results"]),
    )

    assert analysis_state.processed_sections == ["Abstract", "Methods", "Results"]
    assert analysis_state.remaining_sections == []
    assert analysis_state.accumulated_memory == "- [CLAIM] Strong baseline.\n- [RESULT] +3.2 points."
    assert analysis_state.build_final_output().report == "Structured final report."
