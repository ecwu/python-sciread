"""Factory functions for creating document analysis agents.

This module provides convenient factory functions for creating DocumentAgent
instances with common configurations and model setups.
"""

from typing import Optional

from .document_agent import DocumentAgent


def create_agent(
    model: str = "deepseek/deepseek-chat",
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 300.0,
) -> DocumentAgent:
    """Create a DocumentAgent with the specified configuration.

    This is a convenience factory function for creating DocumentAgent instances
    with common configurations.

    Args:
        model: Model identifier for the LLM provider (e.g., "deepseek/deepseek-chat")
        system_prompt: Optional custom system prompt for the agent
        max_retries: Maximum number of retries for failed requests
        timeout: Timeout in seconds for analysis requests

    Returns:
        Configured DocumentAgent instance

    Example:
        >>> agent = create_agent("deepseek/deepseek-chat")
        >>> result = await agent.analyze_document(doc, "Summarize this paper")
    """
    return DocumentAgent(
        model=model,
        system_prompt=system_prompt,
        max_retries=max_retries,
        timeout=timeout,
    )