"""Prompt builders for the ReAct analysis loop."""

from ...document.structure.renderers import format_section_choices
from .models import ReActIterationDeps


def _format_previous_context(previous_thoughts: str) -> str:
    """Format the previous iteration thoughts block."""
    return f"上一轮思考：\n{previous_thoughts}" if previous_thoughts else ""


def _format_processed_sections(processed_sections: list[str]) -> str:
    """Format processed section names."""
    return f"已处理章节：{', '.join(processed_sections)}" if processed_sections else "尚未处理任何章节。"


def _format_unprocessed_sections(
    available_sections: list[str],
    processed_sections: list[str],
    available_section_lengths: dict[str, int],
) -> str:
    """Format remaining section names with clean-text lengths."""
    remaining_sections = [section for section in available_sections if section not in processed_sections]
    if not remaining_sections:
        return "所有章节都已阅读。"

    return "剩余未处理章节（含正文长度）:\n" + format_section_choices(remaining_sections, available_section_lengths)


def _build_final_iteration_prompt(deps: ReActIterationDeps, processed_sections: str, previous_context: str) -> str:
    """Build the prompt for the final synthesis iteration."""
    return (
        f"任务：{deps.task}\n\n"
        f"=== 最终迭代（{deps.current_loop}/{deps.max_loops}）——仅做综合 ===\n\n"
        f"{processed_sections}\n\n"
        f"{previous_context}\n\n"
        f"本轮严格规则：\n"
        f"  ✗ 不要调用 read_section() —— 所有阅读已结束。\n"
        f"  ✓ 第 1 步：调用 get_all_memory()，获取到目前为止累计的全部发现。\n"
        f"  ✓ 第 2 步：基于这些发现综合生成最终结构化报告（见下方格式）。\n"
        f"  ✓ 第 3 步：返回 ReActIterationOutput，并设置 should_continue=False 且填写 report。\n\n"
        f"=== 最终报告格式 ===\n"
        f"撰写简洁、聚焦贡献的报告。不要按章节逐段复述。\n"
        f"请按以下结构输出：\n\n"
        f"1. **核心研究问题与主张**\n"
        f"   论文试图填补什么空白？核心论断是什么？\n\n"
        f"2. **关键贡献**（最重要部分）\n"
        f"   列出 3-5 条具体且明确的贡献。\n"
        f"   语言要精确：指出方法、数据集、指标、性能提升等要点。\n\n"
        f"3. **方法论（概念层）**\n"
        f"   从架构/算法层描述方法，不展开实现细节。\n\n"
        f"4. **主要结果与意义**\n"
        f"   实验显示了什么？关键数字代表什么意义？\n\n"
        f"5. **局限性与开放问题**\n"
        f"   论文承认了哪些尚未解决或不在范围内的问题？\n\n"
        f"语气要求：精确、学术、分析性。避免“论文讨论了……”这类空泛表达，直接陈述结论。"
    )


def _build_regular_iteration_prompt(
    deps: ReActIterationDeps,
    processed_sections: str,
    unprocessed_sections: str,
    previous_context: str,
) -> str:
    """Build the prompt for a normal reading iteration."""
    remaining_loops = max(deps.max_loops - deps.current_loop, 0)
    planning_block = ""
    if deps.current_loop == 1:
        planning_block = (
            "=== 首轮迭代：先制定阅读策略 ===\n"
            "在开始阅读前，先浏览所有可用章节并判断：\n"
            "  • 哪些章节最能直接揭示论文的主张（CLAIMS）与贡献（CONTRIBUTIONS）？\n"
            "    （摘要、引言、结论，以及任何“贡献”小节优先级最高。）\n"
            "  • 哪些章节包含你后续需要的实验证据？\n"
            "  • 哪些章节对当前任务价值较低（如附录、致谢）？\n"
            "  • 优先看正文长度更长的 section；如果长度很短，往往只有标题或过渡句，应优先寻找其下一级子章节。\n"
            "先读信息密度最高的章节。本轮只能调用一次 read_section()，但可以一次读取多个相关章节。\n"
            "请在输出的 thoughts 字段中记录你的阅读计划。\n\n"
        )

    return (
        f"任务：{deps.task}\n\n"
        f"=== 迭代 {deps.current_loop}/{deps.max_loops} （本轮结束后剩余轮次：{remaining_loops}）===\n\n"
        f"{planning_block}"
        f"{processed_sections}\n"
        f"{unprocessed_sections}\n\n"
        f"{previous_context}\n\n"
        f"=== 本轮规则 ===\n"
        f"1. 本轮至多调用一次 read_section()。\n"
        f"   • 章节选择要有策略，不要只按顺序读下一个。\n"
        f"   • 优先选择能直接回答：论文主张了什么？创新点是什么？\n"
        f"   • 可用章节后的 chars 表示该 section 的正文长度；若长度很短，通常说明它只有标题，没有实际内容。\n"
        f"   • 可以一次读取多个主题相关章节，但不要重复读取章节。\n"
        f"   • 剩余轮次：{remaining_loops}。如果本轮后只剩 1 轮，\n"
        f"     请聚焦最高价值的未读章节，跳过低优先级内容。\n\n"
        f"2. 若已完成阅读，本轮至多调用一次 add_memory()。\n"
        f"   • 记录贡献（CONTRIBUTIONS）、主张（CLAIMS）和关键发现（KEY FINDINGS），不要写内容摘要。\n"
        f"   • 记忆内容请使用要点格式：\n"
        f"     - [CLAIM] <论文提出的主张>\n"
        f"     - [CONTRIBUTION] <论文的新颖贡献>\n"
        f"     - [RESULT] <关键实验结果，尽量包含数字>\n"
        f"     - [METHOD] <核心技术，仅在架构层面重要时记录>\n"
        f"   • 不要记录背景、动机或模板化描述。\n\n"
        f"3. 完成工具调用后，立即返回 ReActIterationOutput，不要继续调用其他工具。\n"
        f"   • thoughts：说明本轮阅读依据，以及下一步准备读什么（和原因）。\n"
        f"   • should_continue：\n"
        f"     - 若仍有高价值未读章节，设为 True。\n"
        f"     - 仅当你已准备好在本次输出中亲自写出最终报告时，才设为 False。\n"
        f"       警告：设置 should_continue=False 意味着这是你产出报告的最后机会。\n"
        f"       若设为 should_continue=False，你必须同时完整填写 report 字段，\n"
        f"       之后不会再有自动综合步骤。\n"
        f"       若 report 为空，不要设置 should_continue=False。\n\n"
        f"   • report：保持为空（仅最终迭代才进行综合）。\n\n"
        f"=== 允许的工具 ===\n"
        f"  ✓ read_section(section_names)  —— 本轮最多一次\n"
        f"  ✓ add_memory(memory)           —— 本轮最多一次\n"
        f"  ✓ get_all_memory()             —— 仅在生成最终报告时使用\n"
    )


def build_iteration_system_prompt(deps: ReActIterationDeps) -> str:
    """Build the full system prompt for a single iteration."""
    iteration_input = deps.iteration_input
    processed_sections = _format_processed_sections(iteration_input.processed_sections)
    unprocessed_sections = _format_unprocessed_sections(
        iteration_input.available_sections,
        iteration_input.processed_sections,
        iteration_input.available_section_lengths,
    )
    previous_context = _format_previous_context(iteration_input.previous_thoughts)

    if deps.current_loop >= deps.max_loops:
        return _build_final_iteration_prompt(deps, processed_sections, previous_context)

    return _build_regular_iteration_prompt(
        deps,
        processed_sections,
        unprocessed_sections,
        previous_context,
    )


def build_iteration_user_prompt(current_loop: int, max_loops: int) -> str:
    """Build the short user prompt for one agent call."""
    if current_loop >= max_loops:
        return "最终迭代——不要调用 read_section()。先调用 get_all_memory()，然后返回结构化最终报告，并设置 should_continue=False。"

    return "读取最具策略价值的未处理章节，将贡献与主张提炼为记忆，然后返回你的思考与阅读计划。"
