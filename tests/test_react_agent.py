"""Tests for the ReAct agent helpers and state management."""

from types import SimpleNamespace

import pytest
from rich.console import Console

from sciread.agent.react import analyze_file_with_react
from sciread.agent.react import analyze_file_with_react_sync
from sciread.agent.react.agent import ReActAgent
from sciread.agent.react.agent import _render_sections_read
from sciread.agent.react.agent import add_memory
from sciread.agent.react.agent import get_all_memory
from sciread.agent.react.agent import normalize_section_names
from sciread.agent.react.agent import read_section
from sciread.agent.react.models import ReActAnalysisState
from sciread.agent.react.models import ReActIterationDeps
from sciread.agent.react.models import ReActIterationInput
from sciread.agent.react.models import ReActIterationOutput
from sciread.agent.react.models import ReActIterationState
from sciread.agent.react.prompts import build_iteration_system_prompt
from sciread.document.models import Chunk


class _FakeRunResult:
    def __init__(self, output) -> None:
        self.output = output


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
                available_section_lengths={"Abstract": 320, "Methods": 24},
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
                available_section_lengths={"Abstract": 320, "Methods": 24},
            ),
            current_loop=3,
            max_loops=3,
        )
    )

    assert "=== 首轮迭代：先制定阅读策略 ===" in regular_prompt
    assert "本轮至多调用一次 read_section()" in regular_prompt
    assert "- Abstract | 320 chars" in regular_prompt
    assert "- Methods | 24 chars | 可能仅标题" in regular_prompt
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


@pytest.mark.asyncio
async def test_react_tools_guard_against_repeated_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each non-final iteration should allow at most one read and one memory write."""

    class DummyDocument:
        def get_closest_section_name(self, section: str, threshold: float = 0.7) -> str | None:
            return section

    monkeypatch.setattr(
        "sciread.agent.react.agent._get_sections",
        lambda document, section_names: [(name, f"{name} content") for name in section_names],
    )

    deps = ReActIterationDeps(
        document=DummyDocument(),
        task="Summarize the paper",
        iteration_input=ReActIterationInput(
            task="Summarize the paper",
            previous_thoughts="",
            processed_sections=[],
            available_sections=["Abstract", "Methods"],
        ),
        current_loop=1,
        max_loops=3,
        show_progress=False,
    )
    iteration_state = ReActIterationState()
    ctx = SimpleNamespace(deps=deps, metadata={"iteration_state": iteration_state})

    first_read = await read_section(ctx, ["Abstract"])
    second_read = await read_section(ctx, ["Methods"])
    first_memory = await add_memory(ctx, "- [CLAIM] Main finding.")
    second_memory = await add_memory(ctx, "- [RESULT] Another finding.")

    assert "章节内容（读取自：Abstract）" in first_read
    assert "本轮已调用过 read_section()" in second_read
    assert first_memory == "记忆已记录"
    assert "本轮已调用过 add_memory()" in second_memory
    assert iteration_state.memory_text == "- [CLAIM] Main finding."


@pytest.mark.asyncio
async def test_read_section_warns_when_selected_section_is_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short sections should be flagged as likely heading-only content."""

    class DummyDocument:
        def get_closest_section_name(self, section: str, threshold: float = 0.7) -> str | None:
            return section

    monkeypatch.setattr(
        "sciread.agent.react.agent._get_sections",
        lambda document, section_names: [(name, "3.1 Proposed Method") for name in section_names],
    )

    deps = ReActIterationDeps(
        document=DummyDocument(),
        task="Summarize the paper",
        iteration_input=ReActIterationInput(
            task="Summarize the paper",
            previous_thoughts="",
            processed_sections=[],
            available_sections=["3. Method"],
            available_section_lengths={"3. Method": 19},
        ),
        current_loop=1,
        max_loops=3,
        show_progress=False,
    )
    iteration_state = ReActIterationState()
    ctx = SimpleNamespace(deps=deps, metadata={"iteration_state": iteration_state})

    read_result = await read_section(ctx, ["3. Method"])

    assert "可能只有标题或过渡句" in read_result
    assert "3. Method (19 chars)" in read_result
    assert "章节内容（读取自：3. Method）" in read_result


@pytest.mark.asyncio
async def test_react_tools_enforce_call_order_and_final_memory_access() -> None:
    """Memory retrieval should be final-only, and memory writes require a read first."""
    regular_deps = ReActIterationDeps(
        document=None,
        task="Summarize the paper",
        iteration_input=ReActIterationInput(
            task="Summarize the paper",
            previous_thoughts="",
            processed_sections=[],
            available_sections=["Abstract"],
        ),
        current_loop=1,
        max_loops=3,
        accumulated_memory="- [CLAIM] Prior finding.",
        show_progress=False,
    )
    regular_state = ReActIterationState()
    regular_ctx = SimpleNamespace(deps=regular_deps, metadata={"iteration_state": regular_state})

    add_before_read = await add_memory(regular_ctx, "- [CLAIM] Main finding.")
    get_memory_too_early = await get_all_memory(regular_ctx)

    assert "请先读取章节" in add_before_read
    assert "仅允许在最终迭代调用" in get_memory_too_early

    final_deps = ReActIterationDeps(
        document=None,
        task="Summarize the paper",
        iteration_input=ReActIterationInput(
            task="Summarize the paper",
            previous_thoughts="Ready to synthesize.",
            processed_sections=["Abstract"],
            available_sections=["Abstract"],
        ),
        current_loop=3,
        max_loops=3,
        accumulated_memory="- [CLAIM] Prior finding.",
        show_progress=False,
    )
    final_state = ReActIterationState()
    final_ctx = SimpleNamespace(deps=final_deps, metadata={"iteration_state": final_state})

    first_get_memory = await get_all_memory(final_ctx)
    second_get_memory = await get_all_memory(final_ctx)

    assert "完整累计记忆" in first_get_memory
    assert "本轮已调用过 get_all_memory()" in second_get_memory


def test_react_public_api_uses_clarified_names() -> None:
    """The public API should expose file-oriented entrypoints and analysis-oriented methods."""
    assert analyze_file_with_react.__name__ == "analyze_file_with_react"
    assert analyze_file_with_react_sync.__name__ == "analyze_file_with_react_sync"


def test_render_sections_read_shows_names_and_lengths_without_body_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress output should show which sections were read, but not echo the section body."""
    test_console = Console(record=True, width=120)
    monkeypatch.setattr("sciread.agent.react.agent.console", test_console)

    _render_sections_read(
        [
            ("Methods", "Detailed architecture description with equations."),
            ("Results", "Ablations and numerical comparisons."),
        ]
    )

    rendered = test_console.export_text()
    assert "Read Sections" in rendered
    assert "Methods" in rendered
    assert "Results" in rendered
    assert "Chars" in rendered
    assert "Detailed architecture description with equations." not in rendered
    assert "Ablations and numerical comparisons." not in rendered


@pytest.mark.asyncio
async def test_run_iteration_falls_back_when_agent_run_raises() -> None:
    """Iteration execution should degrade to the safe fallback when the model call fails."""

    class FailingAgent:
        async def run(self, *args, **kwargs):
            raise RuntimeError("model offline")

    agent = ReActAgent.__new__(ReActAgent)
    agent.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None)
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.agent = FailingAgent()

    output, iteration_state = await agent.run_iteration(
        document=SimpleNamespace(),
        iteration_input=ReActIterationInput(task="Summarize", available_sections=["Abstract"]),
        current_loop=1,
        max_loops=3,
        accumulated_memory="",
        show_progress=False,
    )

    assert output.thoughts == "Iteration failed: could not complete analysis."
    assert output.should_continue is True
    assert output.report == ""
    assert iteration_state.sections_read == []


@pytest.mark.asyncio
async def test_run_iteration_forces_final_report_from_accumulated_memory() -> None:
    """The last loop should force completion and use accumulated memory when no report is returned."""

    class ContinuingAgent:
        async def run(self, *args, **kwargs):
            return _FakeRunResult(ReActIterationOutput(thoughts="Need one more pass.", should_continue=True))

    agent = ReActAgent.__new__(ReActAgent)
    agent.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None)
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.agent = ContinuingAgent()

    output, _iteration_state = await agent.run_iteration(
        document=SimpleNamespace(),
        iteration_input=ReActIterationInput(task="Summarize", available_sections=["Abstract"]),
        current_loop=3,
        max_loops=3,
        accumulated_memory="- [CLAIM] Final memory.",
        show_progress=False,
    )

    assert output.should_continue is False
    assert output.report == "- [CLAIM] Final memory."


@pytest.mark.asyncio
async def test_run_analysis_stops_when_all_sections_are_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analysis should stop early once there are no remaining sections to read."""
    agent = ReActAgent.__new__(ReActAgent)
    agent.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None)
    agent.model = object()
    agent.model_identifier = "mock-model"
    agent.agent = object()

    async def fake_run_iteration(
        document,
        iteration_input: ReActIterationInput,
        current_loop: int,
        max_loops: int,
        accumulated_memory: str,
        show_progress: bool,
    ) -> tuple[ReActIterationOutput, ReActIterationState]:
        assert current_loop == 1
        assert iteration_input.available_sections == ["Abstract"]
        return (
            ReActIterationOutput(thoughts="Read abstract.", should_continue=True),
            ReActIterationState(sections_read=["Abstract"], memory_text="- [CLAIM] Abstract summary."),
        )

    monkeypatch.setattr(agent, "run_iteration", fake_run_iteration)
    monkeypatch.setattr(
        "sciread.agent.react.agent.get_section_length_map",
        lambda document, sections: {"Abstract": 120},
    )

    document = SimpleNamespace(
        get_section_names=lambda: ["Abstract"],
        chunks=[Chunk(content="Abstract body", chunk_name="abstract")],
    )

    result = await agent.run_analysis(document, "Summarize", max_loops=3, show_progress=False)

    assert result.should_continue is False
    assert result.report == "- [CLAIM] Abstract summary."
    assert result.thoughts == "Read abstract."
