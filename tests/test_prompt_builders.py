"""Tests for prompt builders and prompt modules."""

from sciread.agent.discussion.models import AgentPersonality
from sciread.agent.discussion.prompts import personalities as personality_prompts
from sciread.agent.search_react.models import SearchReactDeps
from sciread.agent.search_react.models import SearchReactIterationInput
from sciread.agent.search_react.prompts import build_iteration_system_prompt
from sciread.agent.search_react.prompts import build_iteration_user_prompt
from sciread.agent.simple.prompts import build_analysis_prompt


def test_build_analysis_prompt_includes_metadata_and_non_empty_context() -> None:
    """Simple prompt building should include metadata and skip empty supplemental values."""
    prompt = build_analysis_prompt(
        "Document body",
        "Explain the paper",
        {"title": "Paper Title", "authors": "Alice", "venue": ""},
        audience="PhD students",
        note="",
    )

    assert "分析任务：" in prompt
    assert "- Title: Paper Title" in prompt
    assert "- Authors: Alice" in prompt
    assert "- venue" not in prompt
    assert "补充上下文：" in prompt
    assert "- audience: PhD students" in prompt
    assert "- note:" not in prompt
    assert prompt.endswith("请基于文档内容与上述任务要求，给出深入、完整且结构清晰的分析。")


def test_search_react_prompt_builders_cover_first_regular_and_final_iterations() -> None:
    """Search-react prompts should change instructions across loop states."""
    regular_deps = SearchReactDeps(
        document=object(),
        task="Summarize the paper",
        iteration_input=SearchReactIterationInput(
            task="Summarize the paper",
            previous_thoughts="Need better evidence",
            processed_queries=["baseline", "results"],
            strategy="tree",
        ),
        current_loop=1,
        max_loops=3,
    )
    regular_prompt = build_iteration_system_prompt(regular_deps)

    assert "首轮先确定检索策略" in regular_prompt
    assert "已执行检索：" in regular_prompt
    assert "- baseline" in regular_prompt
    assert "上一轮思考：" in regular_prompt
    assert "默认策略是 tree" in regular_prompt
    assert build_iteration_user_prompt(1, 3).startswith("先判断是否需要查看 section tree")

    final_deps = SearchReactDeps(
        document=object(),
        task="Summarize the paper",
        iteration_input=SearchReactIterationInput(task="Summarize the paper"),
        current_loop=3,
        max_loops=3,
    )
    final_prompt = build_iteration_system_prompt(final_deps)

    assert "最终迭代（3/3）" in final_prompt
    assert "不要调用 search_document()" in final_prompt
    assert "先调用 get_all_memory()" in final_prompt
    assert "尚未执行任何检索。" in final_prompt
    assert build_iteration_user_prompt(3, 3) == "最终综合轮：调用 get_all_memory()，然后返回最终报告。"


def test_discussion_personality_prompts_are_accessible() -> None:
    """Discussion prompt coverage should track the prompts used by PersonalityAgent."""
    assert "批判性评估者" in personality_prompts.get_personality_system_prompt(AgentPersonality.CRITICAL_EVALUATOR)
    assert personality_prompts.get_personality_system_prompt("unknown") == "你是一名资深学术研究分析师。"

    prompt_with_sections = personality_prompts.build_insight_generation_prompt(
        AgentPersonality.PRACTICAL_APPLICATOR,
        "Paper Title",
        "Abstract text",
        ["Introduction", "Methods"],
        {"Methods": "Detailed method body"},
        {"phase": "questioning", "iteration": 2, "total_insights": 4},
    )
    assert "### Methods" in prompt_with_sections
    assert "当前阶段：questioning" in prompt_with_sections
    assert "Insight:" in prompt_with_sections

    prompt_without_sections = personality_prompts.build_insight_generation_prompt(
        AgentPersonality.THEORETICAL_INTEGRATOR,
        "Paper Title",
        "Abstract text",
        ["Introduction"],
        {},
        {},
    )
    assert "当前未提供具体章节内容" in prompt_without_sections
    assert "- Introduction" in prompt_without_sections
