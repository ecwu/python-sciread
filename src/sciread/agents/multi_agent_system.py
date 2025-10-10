"""MultiAgentSystem implementation using Pydantic AI's Agent class."""

from typing import Any
from typing import Dict
from typing import List

from pydantic_ai import Agent
from pydantic_ai import RunContext

from ..llm_provider import get_model
from .prompts import get_collaborative_agent_prompt
from .prompts import get_final_synthesis_prompt
from .prompts import get_research_question_prompts
from .schemas import ContributionAnalysis
from .schemas import DocumentAnalysisResult
from .schemas import DocumentDeps
from .schemas import DocumentMetadata
from .schemas import ResearchQuestionAnalysis


def create_research_question_agent(
    question_type: str,
    role_description: str,
    model_identifier: str = "deepseek-chat"
) -> Agent[DocumentDeps, str]:
    """Create a specialized research question agent.

    Args:
        question_type: Type of research question this agent handles
        role_description: Description of the agent's role and focus
        model_identifier: Model to use

    Returns:
        A configured Pydantic AI Agent for research question analysis
    """
    agent = Agent[DocumentDeps, str](
        model=get_model(model_identifier),
        deps_type=DocumentDeps,
        output_type=str,
        system_prompt=(
            f"You are '{question_type.title().replace('_', '')}Agent', part of a collaborative team "
            f"of research analysts working together to understand a research paper. "
            f"Your specific role is to focus on: {role_description}"
        )
    )

    @agent.tool
    async def get_document_context(ctx: RunContext[DocumentDeps]) -> str:
        """Get the full document context for analysis."""
        return ctx.deps.document_text

    @agent.tool
    async def get_available_sections(ctx: RunContext[DocumentDeps]) -> List[str]:
        """Get available document sections."""
        return ctx.deps.available_sections

    @agent.tool
    async def get_section_content(ctx: RunContext[DocumentDeps], section_type: str) -> str:
        """Get content for a specific section."""
        sections = ctx.deps.get_section_chunks(section_type)
        return "\n\n".join([chunk.get('content', '') for chunk in sections])

    @agent.instructions
    def add_collaboration_instructions() -> str:
        """Add instructions for collaborative analysis."""
        return (
            "Since this is a collaborative analysis:\n"
            "1. Provide your initial analysis from your perspective\n"
            "2. Identify areas where input from other team members would be valuable\n"
            "3. Note any questions or uncertainties that other agents might help resolve\n"
            "4. Suggest coordination points with other analysts\n\n"
            "Format your response as:\n"
            "1. **[Your Role] Analysis**: Your primary analysis and insights\n"
            "2. **Collaboration Needs**: Where you need input from other team members\n"
            "3. **Questions for Others**: Specific questions for other agents\n"
            "4. **Coordination Suggestions**: How to integrate your analysis with others"
        )

    return agent


def create_coordinator_agent(model_identifier: str = "deepseek-chat") -> Agent[DocumentDeps, DocumentAnalysisResult]:
    """Create a coordinator agent for multi-agent analysis.

    Args:
        model_identifier: Model to use

    Returns:
        A configured Pydantic AI Agent for coordination
    """
    agent = Agent[DocumentDeps, DocumentAnalysisResult](
        model=get_model(model_identifier),
        deps_type=DocumentDeps,
        output_type=DocumentAnalysisResult,
        system_prompt=(
            "You are a coordinator agent managing the collaborative analysis of academic papers. "
            "You coordinate multiple specialized analysts, each focusing on a specific aspect "
            "of research analysis. Your role is to orchestrate the collaboration process "
            "and synthesize results into a comprehensive final analysis."
        )
    )

    @agent.tool
    async def get_research_question_prompts(ctx: RunContext[DocumentDeps]) -> Dict[str, str]:
        """Get available research question prompts."""
        return get_research_question_prompts()

    @agent.tool
    async def create_specialized_agents(ctx: RunContext[DocumentDeps]) -> Dict[str, str]:
        """Create specialized agents for different research aspects."""
        prompts = get_research_question_prompts()
        agents_info = {}

        # Define role focuses for different question types
        role_focuses = {
            "research_question": "Identify and analyze the core research question(s) that the paper addresses",
            "motivation": "Analyze why the authors chose this research topic and what makes it important",
            "methodology": "Examine how the authors conducted their research and what methods they used",
            "findings": "Investigate what the authors discovered and what results they obtained",
            "contribution": "Evaluate the main contributions and impact of the research",
        }

        for question_type, prompt in prompts.items():
            role_focus = role_focuses.get(question_type, f"Analyze the {question_type.replace('_', ' ')} of the research")
            agent_name = f"{question_type.title().replace('_', '')}Agent"
            agents_info[agent_name] = role_focus

        return agents_info

    @agent.tool
    async def analyze_with_specialized_agent(
        ctx: RunContext[DocumentDeps],
        agent_type: str,
        role_description: str,
        research_question: str
    ) -> str:
        """Analyze using a specialized agent."""
        # Create the specialized agent
        specialized_agent = create_research_question_agent(
            agent_type.lower().replace('agent', ''),
            role_description,
            ctx.deps.model_identifier
        )

        # Get collaborative prompt
        other_agents = await create_specialized_agents(ctx)
        other_roles = [name for name in other_agents.keys() if name != agent_type]

        collaborative_prompt = get_collaborative_agent_prompt(agent_type, other_roles)
        collaborative_prompt = collaborative_prompt.format(
            context=ctx.deps.document_text,
            research_question=research_question,
            role_specific_focus=role_description
        )

        # Run the specialized analysis
        try:
            result = await specialized_agent.run(collaborative_prompt, deps=ctx.deps)
            return result.output
        except Exception as e:
            return f"Analysis failed for {agent_type}: {str(e)}"

    @agent.tool
    async def run_parallel_analyses(
        ctx: RunContext[DocumentDeps],
        research_question: str,
        agent_types: List[str] | None = None
    ) -> Dict[str, str]:
        """Run multiple specialized agents in parallel."""
        import asyncio

        if agent_types is None:
            # Default agent types
            agent_types = ["ResearchQuestionAgent", "MotivationAgent", "MethodologyAgent", "FindingsAgent", "ContributionAgent"]

        # Get agents info
        agents_info = await create_specialized_agents(ctx)

        # Create analysis tasks
        tasks = []
        for agent_type in agent_types:
            if agent_type in agents_info:
                task = analyze_with_specialized_agent(
                    ctx, agent_type, agents_info[agent_type], research_question
                )
                tasks.append((agent_type, task))

        # Run analyses in parallel
        results = {}
        if tasks:
            agent_tasks = [task for _, task in tasks]
            agent_names = [name for name, _ in tasks]

            task_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            for name, result in zip(agent_names, task_results):
                if isinstance(result, Exception):
                    results[name] = f"Analysis failed: {str(result)}"
                else:
                    results[name] = result

        return results

    @agent.tool
    async def synthesize_collaborative_results(
        ctx: RunContext[DocumentDeps],
        agent_analyses: Dict[str, str],
        research_question: str
    ) -> str:
        """Synthesize results from multiple specialized agents."""
        # Create analyses text
        analyses_text = ""
        for agent, analysis in agent_analyses.items():
            analyses_text += f"\n\n## {agent} ANALYSIS:\n{analysis}"

        # Get synthesis prompt
        synthesis_prompt = get_final_synthesis_prompt(agent_analyses)
        synthesis_prompt = synthesis_prompt.format(research_question=research_question)

        # Create simple agent for synthesis
        from .simple_agent import create_simple_agent
        synthesis_agent = create_simple_agent(ctx.deps.model_identifier)

        # Perform synthesis
        result = await synthesis_agent.run(synthesis_prompt, deps=ctx.deps)
        return result.output

    @agent.tool
    async def extract_key_contributions(
        ctx: RunContext[DocumentDeps],
        synthesized_analysis: str
    ) -> List[str]:
        """Extract key contributions from the synthesized analysis."""
        # Simple heuristic to extract contributions
        # Look for sentences that mention contributions, innovations, advances, etc.
        contributions = []
        lines = synthesized_analysis.split('\n')

        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['contribution', 'innovat', 'advanc', 'propos', 'introduc']):
                clean_line = line.strip('-*•0123456789. ')
                if len(clean_line) > 20:  # Filter out very short lines
                    contributions.append(clean_line)

        return contributions[:10]  # Limit to top 10

    @agent.tool
    async def extract_main_findings(
        ctx: RunContext[DocumentDeps],
        synthesized_analysis: str
    ) -> List[str]:
        """Extract main findings from the synthesized analysis."""
        # Simple heuristic to extract findings
        findings = []
        lines = synthesized_analysis.split('\n')

        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['finding', 'result', 'discover', 'found', 'show', 'demonstrat']):
                clean_line = line.strip('-*•0123456789. ')
                if len(clean_line) > 20:
                    findings.append(clean_line)

        return findings[:10]  # Limit to top 10

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
    async def assess_research_suitability(
        ctx: RunContext[DocumentDeps],
        research_question: str
    ) -> Dict[str, Any]:
        """Assess if the research question is suitable for multi-agent analysis."""
        # Check if the question is high-level and complex
        question_indicators = [
            "research question", "why", "how", "what is the", "motivation",
            "contribution", "findings", "methodology", "approach"
        ]

        question_lower = research_question.lower()
        is_high_level = any(indicator in question_lower for indicator in question_indicators)

        if is_high_level:
            reason = "Research question appears to be high-level and complex, well-suited for multi-agent collaborative analysis."
            suitable = True
        else:
            reason = "Research question appears to be specific or factual, better suited for simple analysis."
            suitable = False

        return {
            "is_suitable": suitable,
            "reason": reason,
            "question_type": "high_level" if is_high_level else "specific",
            "question_length": len(research_question),
            "indicators_found": [indicator for indicator in question_indicators if indicator in question_lower]
        }

    return agent


# Pre-configured agent instances
coordinator_agent = create_coordinator_agent("deepseek-chat")
coordinator_agent_gpt4 = create_coordinator_agent("openai:gpt-4o")
coordinator_agent_claude = create_coordinator_agent("anthropic:claude-3-5-sonnet-latest")


async def analyze_document_with_multi_agent(
    document_text: str,
    research_question: str,
    model_identifier: str = "deepseek-chat",
    document_chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> DocumentAnalysisResult:
    """Analyze a document using the multi-agent collaborative approach.

    Args:
        document_text: The full text of the document
        research_question: The high-level research question to analyze
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        DocumentAnalysisResult with the collaborative analysis
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

    # Create coordinator agent
    agent = create_coordinator_agent(model_identifier)

    # Run the multi-agent analysis
    result = await agent.run(
        research_question,
        deps=deps,
        model_settings={"temperature": temperature, "max_tokens": max_tokens}
    )

    # Update processing time
    processing_time = time.time() - start_time
    result_data = result.output
    result_data.execution_time = processing_time

    return result_data


async def analyze_with_agent_details(
    document_text: str,
    research_question: str,
    model_identifier: str = "deepseek-chat",
    document_chunks: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Analyze with detailed information about each agent's contribution.

    Args:
        document_text: The full text of the document
        research_question: The high-level research question to analyze
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with final analysis and agent details
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

    # Create coordinator agent
    agent = create_coordinator_agent(model_identifier)

    # Get detailed agent analyses
    agents_info = await agent._function_toolset.get_tool("create_specialized_agents").func(deps=agent._get_deps(deps))

    agent_analyses = {}
    total_agent_time = 0

    for agent_type, role_description in agents_info.items():
        agent_start = time.time()
        try:
            analysis = await agent._function_toolset.get_tool("analyze_with_specialized_agent").func(
                deps=agent._get_deps(deps),
                agent_type=agent_type,
                role_description=role_description,
                research_question=research_question
            )
            agent_time = time.time() - agent_start
            total_agent_time += agent_time

            agent_analyses[agent_type] = {
                "success": True,
                "content": analysis,
                "execution_time": agent_time,
                "role": role_description
            }
        except Exception as e:
            agent_time = time.time() - agent_start
            total_agent_time += agent_time

            agent_analyses[agent_type] = {
                "success": False,
                "error": str(e),
                "content": "",
                "execution_time": agent_time,
                "role": role_description
            }

    # Get final coordinated analysis
    final_result = await analyze_document_with_multi_agent(
        document_text=document_text,
        research_question=research_question,
        model_identifier=model_identifier,
        document_chunks=document_chunks,
        metadata=metadata,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        "final_analysis": final_result,
        "agent_analyses": agent_analyses,
        "agents_used": list(agents_info.keys()),
        "research_question": research_question,
        "total_agent_execution_time": total_agent_time,
        "total_execution_time": time.time() - start_time,
        "success": final_result.confidence_score > 0.5,
    }