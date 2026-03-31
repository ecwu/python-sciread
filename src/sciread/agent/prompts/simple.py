"""Prompts for the SimpleAgent (formerly DocumentAgent).

This module contains system prompts and instruction templates used by the
SimpleAgent for basic document analysis.
"""

# Default system prompt for academic document analysis
DEFAULT_SYSTEM_PROMPT = """你是一名资深的学术文献分析专家，具备跨学科科研背景知识。你的任务是严谨分析学术论文，并输出全面、准确且有洞察力的报告。

核心要求：
1. 在形成结论前，完整阅读并理解全文
2. 聚焦研究问题、方法、关键发现与学术意义
3. 保持客观，并以文中证据为依据进行判断
4. 同时指出研究的优势与局限
5. 使用清晰、规范、学术化的表达
6. 结构化组织输出，使其对研究者和学生都有参考价值

当你对论文内容做出判断时，请尽量引用或指向论文中的具体部分作为依据。"""


def build_analysis_prompt(text: str, task_prompt: str, document_metadata: dict | None = None, **kwargs) -> str:
    """Build the full analysis prompt for SimpleAgent.

    Args:
        text: Document text content
        task_prompt: Specific task for the analysis
        document_metadata: Optional document metadata information
        **kwargs: Additional context information

    Returns:
        Complete prompt for the agent
    """
    prompt_parts = []

    # Add task prompt first
    prompt_parts.append(f"分析任务：\n{task_prompt}")
    prompt_parts.append("")

    # Add document metadata if available
    if document_metadata:
        prompt_parts.append("文档元数据：")
        for key, value in document_metadata.items():
            if value:
                prompt_parts.append(f"- {key.title()}: {value}")
        prompt_parts.append("")

    # Add the document text
    prompt_parts.append("文档正文：")
    prompt_parts.append(text)
    prompt_parts.append("")

    # Add any additional context from kwargs
    if kwargs:
        prompt_parts.append("补充上下文：")
        for key, value in kwargs.items():
            if value:
                prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Add final instructions
    prompt_parts.append("请基于文档内容与上述任务要求，给出深入、完整且结构清晰的分析。")

    return "\n".join(prompt_parts)
