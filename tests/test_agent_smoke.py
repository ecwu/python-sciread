import re
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from pydantic import BaseModel

from sciread.agent.coordinate_agent import CoordinateAgent
from sciread.agent.discussion.agent import DiscussionAgent
from sciread.agent.discussion.tools import clear_agent_cache
from sciread.agent.react_agent import ReActAgent
from sciread.agent.simple_agent import SimpleAgent
from sciread.document import Document


class DummyRunResult:
    """Minimal pydantic-ai run result for smoke tests."""

    def __init__(self, output: Any, history: list[dict[str, str]] | None = None):
        self.output = output
        self._history = history or []

    def all_messages(self) -> list[dict[str, str]]:
        return self._history


class _DummyRunContext:
    """Minimal context object carrying deps for tool callbacks."""

    def __init__(self, deps: Any):
        self.deps = deps


class DummyAgent:
    """Drop-in stand-in for pydantic_ai.Agent in smoke tests."""

    def __init__(
        self,
        model: Any,
        deps_type: Any | None = None,
        output_type: Any | None = str,
        retries: int = 0,
        system_prompt: str | None = None,
        **_: Any,
    ):
        self.model = model
        self.deps_type = deps_type
        self.output_type = output_type
        self.retries = retries
        self.system_prompt_text = system_prompt
        self._tools: dict[str, Any] = {}
        self._system_prompt_callbacks: list[Any] = []

    def system_prompt(self, callback):
        self._system_prompt_callbacks.append(callback)
        return callback

    def tool(self, callback):
        self._tools[callback.__name__] = callback
        return callback

    async def run(
        self,
        prompt: str,
        deps: Any | None = None,
        message_history: list[dict[str, str]] | None = None,
    ):
        output = await self._build_output(prompt, deps)
        history = list(message_history or [])
        history.append({"role": "assistant", "content": str(output)})
        return DummyRunResult(output=output, history=history)

    async def _build_output(self, prompt: str, deps: Any | None) -> Any:
        prompt_lower = prompt.lower()

        # ReAct smoke path: exercise registered tools and dependency state updates.
        if (
            deps is not None
            and "start analysis. use tools to read sections" in prompt_lower
            and "read_section" in self._tools
            and "append_to_report" in self._tools
        ):
            ctx = _DummyRunContext(deps)
            read_section = self._tools["read_section"]
            append_to_report = self._tools["append_to_report"]

            read_result = await read_section(ctx, deps.current_sections or None)
            await append_to_report(ctx, f"Dummy ReAct report fragment\n{read_result[:400]}")

            if deps.loop_count < deps.max_loops:
                extra_read_result = await read_section(ctx, None)
                await append_to_report(ctx, f"Dummy ReAct extra fragment\n{extra_read_result[:240]}")

            return deps.current_report or "Dummy ReAct report: pipeline is alive."

        output_type = self.output_type
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            return _build_dummy_model_output(output_type)
        return _build_dummy_text_output(prompt)


def _build_dummy_model_output(model_type: type[BaseModel]) -> BaseModel:
    if model_type.__name__ == "AnalysisPlan":
        return model_type(
            analyze_metadata=True,
            analyze_previous_methods=True,
            analyze_research_questions=True,
            analyze_methodology=True,
            analyze_experiments=True,
            analyze_future_directions=True,
            previous_methods_sections=["abstract", "introduction"],
            research_questions_sections=["introduction"],
            methodology_sections=["methodology"],
            experiments_sections=["experiments", "results"],
            future_directions_sections=["discussion", "conclusion"],
            reasoning="Dummy plan for smoke testing.",
        )

    try:
        return model_type()
    except Exception:
        return model_type.model_validate({})


def _build_dummy_text_output(prompt: str) -> str:
    prompt_lower = prompt.lower()

    if "start analysis. use tools to read sections" in prompt_lower:
        return "Dummy ReAct report: pipeline is alive."

    if "provide your evaluation in this format" in prompt_lower and "convergence score" in prompt_lower:
        return "Convergence Score: 0.86\nContinue Discussion: no\nKey Issues Remaining: None\nRecommendations: Proceed to final consensus."

    if "for each question, provide your answer using this exact format" in prompt_lower:
        question_ids = re.findall(r"\[(Q-[A-Z]+-\d+)\]", prompt)
        if not question_ids:
            question_ids = ["Q-CE-01"]

        blocks = []
        for question_id in question_ids:
            blocks.append(
                "\n".join(
                    [
                        "---",
                        f"Answer to [{question_id}]:",
                        "Response: Dummy answer for smoke test.",
                        "Stance: clarify",
                        "Revised Insight: None",
                        "Confidence: 0.80",
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    if "for each insight you evaluate, use this format" in prompt_lower:
        insight_ids = re.findall(r"\[(INS-[A-Z]+-\d+)\]", prompt)
        if not insight_ids:
            insight_ids = ["INS-CE-01"]

        blocks = []
        for index, insight_id in enumerate(insight_ids):
            decision = "ask" if index == 0 else "skip"
            question_text = "What direct evidence supports this point?" if decision == "ask" else "None"
            priority = "0.80" if decision == "ask" else "0.0"
            question_type = "clarification" if decision == "ask" else "none"
            blocks.append(
                "\n".join(
                    [
                        "---",
                        f"Question about [{insight_id}]:",
                        f"Decision: {decision}",
                        "Reason: Dummy decision for smoke test.",
                        f"Question: {question_text}",
                        f"Priority: {priority}",
                        f"Type: {question_type}",
                        "---",
                    ]
                )
            )
        return "\n".join(blocks)

    if "select which sections of a paper to read" in prompt_lower:
        return "abstract\nintroduction\nmethodology\nexperiments"

    if "format your response as:" in prompt_lower and "summary:" in prompt_lower and "significance:" in prompt_lower:
        return "SUMMARY:\nDummy summary for smoke testing.\n\nSIGNIFICANCE:\nDummy significance assessment."

    if "your task is:" in prompt_lower and "insight" in prompt_lower:
        return (
            "Insight: Dummy insight on methodology.\n"
            "Importance: 0.85\n"
            "Confidence: 0.80\n"
            "Evidence: Evidence snippet from selected sections.\n"
            "Questions: What are the boundary conditions?\n\n"
            "Insight: Dummy insight on experiments.\n"
            "Importance: 0.75\n"
            "Confidence: 0.78\n"
            "Evidence: Experiment section indicates stable behavior."
        )

    return "Dummy LLM output for smoke test."


def _build_smoke_document() -> Document:
    text = """
Abstract
This is a smoke-test paper for validating end-to-end agent pipelines.

Introduction
The paper introduces a predictable workflow for analysis.

Methodology
We use a deterministic fake model and fixed prompts.

Experiments
The setup validates all code paths without external APIs.

Results
The observed behavior is stable and repeatable.

Discussion
Findings indicate pipeline integrity.

Conclusion
This setup is suitable for smoke testing.
"""

    document = Document.from_text(text)
    document.source_path = Path("smoke-test-paper.pdf")
    document.metadata.title = "Smoke Test Paper"
    return document


@pytest.fixture
def dummy_llm(monkeypatch: pytest.MonkeyPatch):
    from sciread.agent import coordinate_agent as coordinate_agent_module
    from sciread.agent import react_agent as react_agent_module
    from sciread.agent import simple_agent as simple_agent_module
    from sciread.agent.discussion import agent as discussion_agent_module
    from sciread.agent.discussion import consensus as consensus_module
    from sciread.agent.discussion import personalities as personalities_module

    def _dummy_get_model(_model_name: str):
        return object()

    patched_modules = [
        simple_agent_module,
        react_agent_module,
        coordinate_agent_module,
        discussion_agent_module,
        personalities_module,
        consensus_module,
    ]

    for module in patched_modules:
        monkeypatch.setattr(module, "Agent", DummyAgent)
        monkeypatch.setattr(module, "get_model", _dummy_get_model)

    clear_agent_cache()
    yield
    clear_agent_cache()


@pytest.mark.asyncio
async def test_simple_agent_pipeline_smoke(dummy_llm):
    document = _build_smoke_document()
    agent = SimpleAgent(model=object())

    result = await agent.analyze(document=document, task_prompt="Summarize this document.")

    assert "Dummy" in result


def test_react_agent_pipeline_smoke(dummy_llm):
    document = _build_smoke_document()
    agent = ReActAgent(model="dummy", max_loops=2)

    outcome: dict[str, Any] = {}

    def _target() -> None:
        outcome["result"] = agent.analyze_document(
            document,
            task="Summarize methodology and results.",
            show_progress=False,
        )

    thread = Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    result = outcome["result"]

    assert "Dummy ReAct report fragment" in result
    assert "READ_RESULT" in result


@pytest.mark.asyncio
async def test_coordinate_agent_pipeline_smoke(dummy_llm):
    document = _build_smoke_document()
    agent = CoordinateAgent(model=object(), timeout=15.0)

    result = await agent.analyze(document)

    assert "Dummy" in result.final_report
    assert result.execution_summary["successful_agents"] > 0


@pytest.mark.asyncio
async def test_discussion_agent_pipeline_smoke(dummy_llm):
    document = _build_smoke_document()
    agent = DiscussionAgent(model_name="dummy", max_iterations=1, max_discussion_time_minutes=2)

    result = await agent.analyze_document(document)

    assert "Dummy" in result.summary
    assert "Dummy" in result.significance
    assert result.confidence_score >= 0.0
