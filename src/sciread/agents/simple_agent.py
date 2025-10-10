"""SimpleAgent implementation using Pydantic AI's Agent class."""

from datetime import date
from typing import Any

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ..llm_provider import get_model
from .prompts import get_simple_analysis_prompt
from .prompts import remove_citations_section
from .schemas import DocumentDeps
from .schemas import SimpleAnalysisResult


def create_simple_agent(model_identifier: str = "deepseek-chat") -> Agent[DocumentDeps, SimpleAnalysisResult]:
    """Create a simple document analysis agent using Pydantic AI.

    This agent uses the Feynman technique to create detailed explanations
    as if written by the paper's author. It processes the full document
    with a single LLM call.

    Args:
        model_identifier: The model identifier to use

    Returns:
        A configured Pydantic AI Agent
    """

    agent = Agent[DocumentDeps, SimpleAnalysisResult](
        model=get_model(model_identifier),
        deps_type=DocumentDeps,
        output_type=SimpleAnalysisResult,
        system_prompt=(
            "You are an expert academic analyst who excels at explaining complex research papers "
            "using the Feynman technique. You create detailed, accessible explanations that "
            "demonstrate deep understanding by making complex ideas simple and relatable."
        )
    )

    @agent.tool
    async def get_document_text(ctx: RunContext[DocumentDeps]) -> str:
        """Get the full document text, with citations removed to save tokens."""
        text = ctx.deps.document_text
        return remove_citations_section(text)

    @agent.tool
    async def get_document_length(ctx: RunContext[DocumentDeps]) -> int:
        """Get the length of the document text."""
        return len(ctx.deps.document_text)

    @agent.tool
    async def estimate_tokens(ctx: RunContext[DocumentDeps]) -> int:
        """Estimate token count for the document."""
        text = remove_citations_section(ctx.deps.document_text)
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4

    @agent.tool
    async def check_document_suitability(ctx: RunContext[DocumentDeps]) -> dict[str, Any]:
        """Check if the document is suitable for simple analysis."""
        text = ctx.deps.document_text
        estimated_tokens = len(text) // 4
        context_limit = 128000  # Typical model context limit

        is_suitable = estimated_tokens < context_limit
        reason = (
            f"Document suitable for simple processing (estimated {estimated_tokens:,} tokens)."
            if is_suitable else
            f"Document too large for simple processing (estimated {estimated_tokens:,} tokens > {context_limit:,} limit)."
        )

        return {
            "is_suitable": is_suitable,
            "reason": reason,
            "estimated_tokens": estimated_tokens,
            "context_limit": context_limit
        }

    @agent.instructions
    def add_date_context() -> str:
        """Add current date context to the analysis."""
        return f"Current date: {date.today()}"

    @agent.instructions
    def add_analysis_instructions() -> str:
        """Add specific instructions for Feynman technique analysis."""
        return (
            "When analyzing the document, use the Feynman technique to:\n"
            "1. Break down complex concepts into simple, accessible terms\n"
            "2. Use analogies and examples to explain abstract concepts\n"
            "3. Identify core principles underlying the research\n"
            "4. Write as if teaching this material to someone intelligent but unfamiliar with the field\n"
            "5. Adopt the perspective of the paper's author, explaining your work with confidence and clarity\n\n"
            "Your analysis should include:\n"
            "- Introduction: Overview of the problem and why it matters (in simple terms)\n"
            "- Core Concepts: Explanation of the fundamental ideas and principles\n"
            "- Methodology: Clear description of the approach as if explaining to a colleague\n"
            "- Key Insights: The main discoveries and why they're important\n"
            "- Implications: What this work means for the field and practice\n"
            "- Future Directions: Where this research could lead next\n\n"
            "Write in clear, accessible language while maintaining technical accuracy. "
            "Use examples and analogies to make complex ideas understandable. "
            "Demonstrate mastery by making the complex simple."
        )

    return agent


# Pre-configured agent instances for common use cases
simple_agent = create_simple_agent("deepseek-chat")
simple_agent_gpt4 = create_simple_agent("openai:gpt-4o")
simple_agent_claude = create_simple_agent("anthropic:claude-3-5-sonnet-latest")


async def analyze_document_simple(
    document_text: str,
    question: str,
    model_identifier: str = "deepseek-chat",
    document_chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> SimpleAnalysisResult:
    """Analyze a document using the simple agent approach.

    Args:
        document_text: The full text of the document
        question: The analysis question or prompt
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        SimpleAnalysisResult with the analysis
    """
    import time
    start_time = time.time()

    # Create dependencies
    deps = DocumentDeps(
        document_text=document_text,
        document_chunks=document_chunks or [],
        available_sections=list(set(chunk.get('chunk_type', 'unknown') for chunk in (document_chunks or []))),
        metadata=metadata or {},
        model_identifier=model_identifier,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Create agent
    agent = create_simple_agent(model_identifier)

    # Format the prompt using the existing template
    prompt_template = get_simple_analysis_prompt()
    formatted_question = prompt_template.format(
        context=document_text,
        question=question
    )

    # Run the agent
    result = await agent.run(
        formatted_question,
        deps=deps,
        model_settings={"temperature": temperature, "max_tokens": max_tokens}
    )

    # Update processing time
    processing_time = time.time() - start_time
    result_data = result.output
    result_data.processing_time = processing_time

    return result_data