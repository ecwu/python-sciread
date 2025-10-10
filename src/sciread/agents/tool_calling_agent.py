"""ToolCallingAgent implementation using Pydantic AI's Agent class."""

from typing import Any
from typing import Dict
from typing import List

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ..llm_provider import get_model
from .prompts import get_section_specific_prompt
from .prompts import get_synthesis_prompt
from .schemas import DocumentAnalysisResult
from .schemas import DocumentDeps
from .schemas import DocumentMetadata
from .schemas import SectionAnalysis


def create_tool_calling_agent(model_identifier: str = "deepseek-chat") -> Agent[DocumentDeps, DocumentAnalysisResult]:
    """Create a tool-calling document analysis agent using Pydantic AI.

    This agent uses a controller approach with specialized tools for analyzing
    different sections of a document, then synthesizes the results.

    Args:
        model_identifier: The model identifier to use

    Returns:
        A configured Pydantic AI Agent
    """

    agent = Agent[DocumentDeps, DocumentAnalysisResult](
        model=get_model(model_identifier),
        deps_type=DocumentDeps,
        output_type=DocumentAnalysisResult,
        system_prompt=(
            "You are a research analysis coordinator responsible for orchestrating the analysis "
            "of academic papers. You use specialized tools to analyze different sections of the document, "
            "then synthesize the results into a comprehensive analysis. "
            "Your role is to coordinate multiple analyses and integrate insights effectively."
        )
    )

    @agent.tool
    async def get_available_sections(ctx: RunContext[DocumentDeps]) -> List[str]:
        """Get list of available sections in the document."""
        return ctx.deps.available_sections

    @agent.tool
    async def get_section_chunks(ctx: RunContext[DocumentDeps], section_type: str) -> List[Dict[str, Any]]:
        """Get chunks of a specific section type."""
        return ctx.deps.get_section_chunks(section_type)

    @agent.tool
    async def analyze_section(
        ctx: RunContext[DocumentDeps],
        section_type: str,
        analysis_task: str
    ) -> SectionAnalysis:
        """Analyze a specific section of the document."""
        import time
        start_time = time.time()

        # Get section chunks
        section_chunks = ctx.deps.get_section_chunks(section_type)
        if not section_chunks:
            return SectionAnalysis(
                section_type=section_type,
                content_summary=f"No {section_type} section found in the document.",
                relevance_score=0.0,
                processing_time=time.time() - start_time
            )

        # Combine section content
        section_content = "\n\n".join([
            chunk.get('content', '') for chunk in section_chunks
        ])

        # Get section-specific prompt
        prompt_template = get_section_specific_prompt(section_type)
        prompt = prompt_template.format(
            context=section_content,
            task=analysis_task
        )

        # Create a simple analysis agent for this section
        from .simple_agent import create_simple_agent
        section_agent = create_simple_agent(ctx.deps.model_identifier)

        # Analyze the section
        try:
            # Use the agent to analyze the section content
            result = await section_agent.run(prompt, deps=ctx.deps)

            # Extract key insights from the analysis
            content = result.output.content
            # Simple heuristic to extract insights - look for bullet points, numbered lists, etc.
            insights = []
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('-', '*', '•')) or (line and line[0].isdigit() and '.' in line[:5]):
                    insights.append(line.lstrip('-*•0123456789. ').strip())

            processing_time = time.time() - start_time
            return SectionAnalysis(
                section_type=section_type,
                content_summary=content,
                key_insights=insights[:10],  # Limit to top 10 insights
                relevance_score=0.8,  # Default relevance score
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return SectionAnalysis(
                section_type=section_type,
                content_summary=f"Error analyzing {section_type} section: {str(e)}",
                relevance_score=0.0,
                processing_time=processing_time
            )

    @agent.tool
    async def analyze_multiple_sections(
        ctx: RunContext[DocumentDeps],
        section_types: List[str],
        analysis_task: str
    ) -> List[SectionAnalysis]:
        """Analyze multiple sections in parallel."""
        import asyncio

        # Create analysis tasks for each section
        tasks = []
        for section_type in section_types:
            if ctx.deps.has_section(section_type):
                task = analyze_section(ctx, section_type, analysis_task)
                tasks.append(task)

        # Run analyses in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and return successful analyses
            successful_results = []
            for result in results:
                if isinstance(result, SectionAnalysis):
                    successful_results.append(result)
                else:
                    # Create error analysis for failed sections
                    successful_results.append(SectionAnalysis(
                        section_type="error",
                        content_summary=f"Analysis failed: {str(result)}",
                        relevance_score=0.0,
                        processing_time=0.0
                    ))
            return successful_results
        else:
            return []

    @agent.tool
    async def synthesize_analyses(
        ctx: RunContext[DocumentDeps],
        section_analyses: List[SectionAnalysis],
        original_task: str
    ) -> str:
        """Synthesize multiple section analyses into a comprehensive result."""
        # Create analyses text for synthesis
        analyses_text = ""
        for analysis in section_analyses:
            analyses_text += f"\n\n## {analysis.section_type.upper()} ANALYSIS:\n{analysis.content_summary}"

        # Get synthesis prompt
        synthesis_prompt = get_synthesis_prompt({
            analysis.section_type: analysis.content_summary
            for analysis in section_analyses
        })
        synthesis_prompt = synthesis_prompt.format(task=original_task)

        # Create simple agent for synthesis
        from .simple_agent import create_simple_agent
        synthesis_agent = create_simple_agent(ctx.deps.model_identifier)

        # Perform synthesis
        result = await synthesis_agent.run(synthesis_prompt, deps=ctx.deps)
        return result.output.content

    @agent.tool
    async def create_document_metadata(ctx: RunContext[DocumentDeps]) -> DocumentMetadata:
        """Create metadata about the document."""
        return DocumentMetadata(
            title=ctx.deps.metadata.get('title', 'Unknown'),
            authors=ctx.deps.metadata.get('authors', []),
            chunk_count=len(ctx.deps.document_chunks),
            section_types=ctx.deps.available_sections
        )

    @agent.tool
    async def check_suitability_for_section_analysis(ctx: RunContext[DocumentDeps]) -> Dict[str, Any]:
        """Check if the document is suitable for section-based analysis."""
        available_sections = ctx.deps.available_sections
        chunk_count = len(ctx.deps.document_chunks)
        chunk_types = set(chunk.get('chunk_type', 'unknown') for chunk in ctx.deps.document_chunks)

        # Document is suitable if it has multiple sections or diverse chunk types
        has_multiple_sections = len(available_sections) > 2
        has_diverse_types = len(chunk_types) > 2
        has_many_chunks = chunk_count > 5

        is_suitable = has_multiple_sections or (has_diverse_types and has_many_chunks)

        if is_suitable:
            reason = (
                f"Document has {len(available_sections)} identifiable sections "
                f"and {len(chunk_types)} chunk types, suitable for section-based analysis."
            )
        else:
            reason = (
                f"Document lacks clear section structure (only {len(available_sections)} sections, "
                f"{len(chunk_types)} types), better suited for simple analysis."
            )

        return {
            "is_suitable": is_suitable,
            "reason": reason,
            "available_sections": available_sections,
            "chunk_types": list(chunk_types),
            "chunk_count": chunk_count
        }

    return agent


# Pre-configured agent instances
tool_calling_agent = create_tool_calling_agent("deepseek-chat")
tool_calling_agent_gpt4 = create_tool_calling_agent("openai:gpt-4o")
tool_calling_agent_claude = create_tool_calling_agent("anthropic:claude-3-5-sonnet-latest")


async def analyze_document_with_sections(
    document_text: str,
    question: str,
    model_identifier: str = "deepseek-chat",
    document_chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> DocumentAnalysisResult:
    """Analyze a document using the tool-calling agent approach.

    Args:
        document_text: The full text of the document
        question: The analysis question or prompt
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        DocumentAnalysisResult with the comprehensive analysis
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
    agent = create_tool_calling_agent(model_identifier)

    # Run the agent
    result = await agent.run(
        question,
        deps=deps,
        model_settings={"temperature": temperature, "max_tokens": max_tokens}
    )

    # Update processing time
    processing_time = time.time() - start_time
    result_data = result.output
    result_data.execution_time = processing_time

    return result_data