"""Prompt builders for the search-react agent."""

from .models import SearchReactDeps


def _format_previous_context(previous_thoughts: str) -> str:
    """Format the previous iteration thoughts block."""
    return f"上一轮思考：\n{previous_thoughts}" if previous_thoughts else ""


def _format_processed_queries(processed_queries: list[str]) -> str:
    """Format previously issued retrieval queries."""
    if not processed_queries:
        return "尚未执行任何检索。"
    return "已执行检索：\n" + "\n".join(f"- {query}" for query in processed_queries)


def build_iteration_system_prompt(deps: SearchReactDeps) -> str:
    """Build the system prompt for one search-react iteration."""
    iteration_input = deps.iteration_input
    previous_context = _format_previous_context(iteration_input.previous_thoughts)
    processed_queries = _format_processed_queries(iteration_input.processed_queries)

    if deps.current_loop >= deps.max_loops:
        return (
            f"任务：{deps.task}\n\n"
            f"=== 最终迭代（{deps.current_loop}/{deps.max_loops}）===\n"
            f"{processed_queries}\n\n"
            f"{previous_context}\n\n"
            "本轮规则：\n"
            "1. 不要调用 search_document()。\n"
            "2. 先调用 get_all_memory()。\n"
            "3. 基于累计记忆输出最终结构化报告。\n"
            "4. 返回 SearchReactIterationOutput，设置 should_continue=False，并填写 report。\n"
            "写作时直接回答任务，不要按工具调用过程复述。"
        )

    planning_hint = ""
    if deps.current_loop == 1:
        planning_hint = (
            "首轮先确定检索策略：优先设计高价值 query，必要时先调用 inspect_section_tree() 理解结构，"
            f"默认策略是 {iteration_input.strategy}，每次 search_document() 只做一轮检索并返回检索结果包。\n\n"
        )

    return (
        f"任务：{deps.task}\n\n"
        f"=== 迭代 {deps.current_loop}/{deps.max_loops} ===\n"
        f"{planning_hint}"
        f"{processed_queries}\n\n"
        f"{previous_context}\n\n"
        "本轮规则：\n"
        "1. 可以调用一次 inspect_section_tree() 或一次 search_document()。\n"
        "2. 若调用了 search_document()，本轮随后最多调用一次 add_memory()。\n"
        "3. memory 只记录贡献、主张、结果、方法、局限，不写流水账摘要。\n"
        "4. 若还需要补证据或换 query，should_continue=True；仅在准备好直接产出报告时才设置 should_continue=False。\n"
        "5. 默认 report 留空，最终综合轮再填写完整报告。\n"
    )


def build_iteration_user_prompt(current_loop: int, max_loops: int) -> str:
    """Build the user prompt for a single iteration."""
    if current_loop >= max_loops:
        return "最终综合轮：调用 get_all_memory()，然后返回最终报告。"
    return "先判断是否需要查看 section tree，再执行一次高价值检索，提炼记忆并给出下一步计划。"
