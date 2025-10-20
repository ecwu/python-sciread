"""Prompts for the SimpleAgent (formerly DocumentAgent).

This module contains system prompts and instruction templates used by the
SimpleAgent for basic document analysis.
"""

# Default system prompt for academic document analysis
DEFAULT_SYSTEM_PROMPT = """You are an expert academic document analyst with deep knowledge across multiple scientific disciplines. Your task is to carefully analyze academic papers and provide comprehensive, accurate, and insightful reports.

Key guidelines:
1. Read the entire document thoroughly before forming conclusions
2. Focus on the main research question, methodology, findings, and implications
3. Be objective and evidence-based in your analysis
4. Highlight both strengths and limitations of the research
5. Use clear, academic language appropriate for scholarly discourse
6. Structure your responses to be useful for researchers and students

Always provide citations or references to specific parts of the paper when making claims about its content."""


def build_analysis_prompt(
    text: str,
    task_prompt: str,
    document_metadata: dict = None,
    **kwargs
) -> str:
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
    prompt_parts.append(f"ANALYSIS TASK:\n{task_prompt}")
    prompt_parts.append("")

    # Add document metadata if available
    if document_metadata:
        prompt_parts.append("DOCUMENT METADATA:")
        for key, value in document_metadata.items():
            if value:
                prompt_parts.append(f"- {key.title()}: {value}")
        prompt_parts.append("")

    # Add the document text
    prompt_parts.append("DOCUMENT TEXT:")
    prompt_parts.append(text)
    prompt_parts.append("")

    # Add any additional context from kwargs
    if kwargs:
        prompt_parts.append("ADDITIONAL CONTEXT:")
        for key, value in kwargs.items():
            if value:
                prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Add final instructions
    prompt_parts.append("Please provide a thorough analysis based on the document content and the specific task requirements above.")

    return "\n".join(prompt_parts)