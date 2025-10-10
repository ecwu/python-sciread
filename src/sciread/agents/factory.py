"""Factory functions for creating and selecting appropriate Pydantic AI agents."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from .multi_agent_system import analyze_document_with_multi_agent
from .multi_agent_system import analyze_with_agent_details
from .schemas import DocumentAnalysisResult
from .schemas import SimpleAnalysisResult
from .simple_agent import analyze_document_simple
from .tool_calling_agent import analyze_document_with_sections


class AgentSelector:
    """Utility class for selecting the appropriate agent based on document and question characteristics."""

    @staticmethod
    def select_agent(
        document_text: str,
        question: str,
        document_chunks: List[Dict[str, Any]] | None = None,
        model_identifier: str = "deepseek-chat",
        force_agent: str | None = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Select the most appropriate agent for the given document and question.

        Args:
            document_text: The full text of the document
            question: Analysis question or prompt
            document_chunks: Optional list of document chunks
            model_identifier: Model to use for analysis
            force_agent: Force a specific agent type ('simple', 'tool_calling', 'multi_agent')

        Returns:
            Tuple of (agent_function_name, agent_display_name, selection_metadata)
        """
        if force_agent:
            return AgentSelector._create_forced_agent(force_agent)

        # Analyze document characteristics
        doc_length = len(document_text)
        estimated_tokens = doc_length // 4  # Rough estimation

        # Analyze document structure
        chunk_count = len(document_chunks) if document_chunks else 0
        available_sections = list(set(chunk.get('chunk_type', 'unknown') for chunk in (document_chunks or []))) if document_chunks else []
        section_types = set(available_sections)

        # Analyze question complexity
        question_lower = question.lower()
        research_indicators = [
            "research question", "why", "how", "what is the", "motivation",
            "contribution", "findings", "methodology", "approach"
        ]
        is_research_question = any(indicator in question_lower for indicator in research_indicators)

        # Decision logic
        if estimated_tokens > 100000:  # Very large document
            return (
                "analyze_document_simple",
                "SimpleAgent",
                {
                    "reason": f"Document too large for complex processing ({estimated_tokens:,} tokens). Using simple approach.",
                    "estimated_tokens": estimated_tokens,
                    "document_length": doc_length,
                    "recommended_approach": "simple"
                }
            )

        elif is_research_question and len(section_types) > 2:
            return (
                "analyze_document_with_multi_agent",
                "MultiAgentSystem",
                {
                    "reason": f"High-level research question with structured document ({len(section_types)} sections). Ideal for collaborative analysis.",
                    "estimated_tokens": estimated_tokens,
                    "document_length": doc_length,
                    "section_types": list(section_types),
                    "recommended_approach": "multi_agent"
                }
            )

        elif chunk_count > 10 and len(section_types) > 2:
            return (
                "analyze_document_with_sections",
                "ToolCallingAgent",
                {
                    "reason": f"Document has clear section structure ({chunk_count} chunks, {len(section_types)} sections). Suitable for section-based analysis.",
                    "estimated_tokens": estimated_tokens,
                    "document_length": doc_length,
                    "chunk_count": chunk_count,
                    "section_types": list(section_types),
                    "recommended_approach": "tool_calling"
                }
            )

        else:
            return (
                "analyze_document_simple",
                "SimpleAgent",
                {
                    "reason": f"Document suitable for simple processing ({estimated_tokens:,} tokens, {len(section_types)} sections).",
                    "estimated_tokens": estimated_tokens,
                    "document_length": doc_length,
                    "section_types": list(section_types),
                    "recommended_approach": "simple"
                }
            )

    @staticmethod
    def _create_forced_agent(agent_type: str) -> Tuple[str, str, Dict[str, Any]]:
        """Create a specific agent type.

        Args:
            agent_type: Type of agent to create

        Returns:
            Tuple of (agent_function_name, agent_display_name, selection_metadata)
        """
        agent_mapping = {
            "simple": ("analyze_document_simple", "SimpleAgent", {"forced": True}),
            "tool_calling": ("analyze_document_with_sections", "ToolCallingAgent", {"forced": True}),
            "multi_agent": ("analyze_document_with_multi_agent", "MultiAgentSystem", {"forced": True}),
        }

        if agent_type not in agent_mapping:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_mapping.keys())}")

        func_name, display_name, metadata = agent_mapping[agent_type]
        metadata["reason"] = f"Forced selection of {display_name}"

        return func_name, display_name, metadata


async def create_agent_analysis(
    document_text: str,
    question: str,
    agent_type: str = "auto",
    model_identifier: str = "deepseek-chat",
    document_chunks: List[Dict[str, Any]] | None = None,
    metadata: Dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> SimpleAnalysisResult | DocumentAnalysisResult:
    """Create and run an agent analysis.

    Args:
        document_text: The full text of the document
        question: Analysis question or prompt
        agent_type: Type of agent ('simple', 'tool_calling', 'multi_agent', 'auto')
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Analysis result from the selected agent
    """
    # Select the appropriate agent
    func_name, display_name, selection_info = AgentSelector.select_agent(
        document_text=document_text,
        question=question,
        document_chunks=document_chunks,
        model_identifier=model_identifier,
        force_agent=agent_type if agent_type != "auto" else None
    )

    # Log the selection
    from ..logging_config import get_logger
    logger = get_logger(__name__)
    logger.info(f"Selected agent: {display_name} - {selection_info['reason']}")

    # Map function names to actual functions
    agent_functions = {
        "analyze_document_simple": analyze_document_simple,
        "analyze_document_with_sections": analyze_document_with_sections,
        "analyze_document_with_multi_agent": analyze_document_with_multi_agent,
    }

    if func_name not in agent_functions:
        raise ValueError(f"Unknown agent function: {func_name}")

    # Execute the selected agent
    agent_function = agent_functions[func_name]

    try:
        result = await agent_function(
            document_text=document_text,
            question=question,
            model_identifier=model_identifier,
            document_chunks=document_chunks,
            metadata=metadata,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add selection metadata to the result if possible
        if hasattr(result, 'metadata'):
            result.metadata['agent_selection'] = selection_info
            result.metadata['agent_type'] = display_name

        return result

    except Exception as e:
        logger.error(f"Agent execution failed for {display_name}: {str(e)}")
        # Fallback to simple agent
        if func_name != "analyze_document_simple":
            logger.info("Falling back to SimpleAgent")
            return await analyze_document_simple(
                document_text=document_text,
                question=question,
                model_identifier=model_identifier,
                document_chunks=document_chunks,
                metadata=metadata,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise


async def analyze_document(
    document_text: str,
    question: str,
    agent_type: str = "auto",
    model_identifier: str = "deepseek-chat",
    document_chunks: List[Dict[str, Any]] | None = None,
    metadata: Dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> SimpleAnalysisResult | DocumentAnalysisResult:
    """Convenience function to analyze a document with automatic agent selection.

    Args:
        document_text: The full text of the document
        question: Analysis question or prompt
        agent_type: Type of agent to use ('simple', 'tool_calling', 'multi_agent', 'auto')
        model_identifier: Model to use for analysis
        document_chunks: Optional list of document chunks
        metadata: Optional document metadata
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Analysis result from the selected agent
    """
    return await create_agent_analysis(
        document_text=document_text,
        question=question,
        agent_type=agent_type,
        model_identifier=model_identifier,
        document_chunks=document_chunks,
        metadata=metadata,
        temperature=temperature,
        max_tokens=max_tokens
    )


async def get_agent_recommendations(
    document_text: str,
    question: str,
    document_chunks: List[Dict[str, Any]] | None = None,
    model_identifier: str = "deepseek-chat"
) -> List[Dict[str, Any]]:
    """Get recommendations for all available agents with their suitability.

    Args:
        document_text: The full text of the document
        question: Analysis question or prompt
        document_chunks: Optional list of document chunks
        model_identifier: Model to use for analysis

    Returns:
        List of agent recommendations with suitability ratings
    """
    recommendations = []

    # Analyze document characteristics
    doc_length = len(document_text)
    estimated_tokens = doc_length // 4
    chunk_count = len(document_chunks) if document_chunks else 0
    available_sections = list(set(chunk.get('chunk_type', 'unknown') for chunk in (document_chunks or []))) if document_chunks else []
    section_types = set(available_sections)

    # Analyze question complexity
    question_lower = question.lower()
    research_indicators = [
        "research question", "why", "how", "what is the", "motivation",
        "contribution", "findings", "methodology", "approach"
    ]
    is_research_question = any(indicator in question_lower for indicator in research_indicators)

    # SimpleAgent recommendation
    simple_suitable = estimated_tokens < 128000
    simple_reason = (
        f"SimpleAgent: Document suitable for simple processing ({estimated_tokens:,} tokens)."
        if simple_suitable else
        f"SimpleAgent: Document large but simple agent can handle it ({estimated_tokens:,} tokens)."
    )

    recommendations.append({
        "agent_name": "SimpleAgent",
        "agent_type": "simple",
        "is_suitable": True,  # Always suitable as fallback
        "reason": simple_reason,
        "estimated_tokens": estimated_tokens,
        "confidence": 0.8 if simple_suitable else 0.6,
        "best_for": ["General analysis", "Simple questions", "Large documents"]
    })

    # ToolCallingAgent recommendation
    tool_suitable = chunk_count > 5 and len(section_types) > 2
    tool_reason = (
        f"ToolCallingAgent: Document has clear section structure ({chunk_count} chunks, {len(section_types)} sections)."
        if tool_suitable else
        f"ToolCallingAgent: Document lacks clear section structure for effective section-based analysis."
    )

    recommendations.append({
        "agent_name": "ToolCallingAgent",
        "agent_type": "tool_calling",
        "is_suitable": tool_suitable,
        "reason": tool_reason,
        "section_count": len(section_types),
        "chunk_count": chunk_count,
        "confidence": 0.9 if tool_suitable else 0.3,
        "best_for": ["Structured documents", "Section-by-section analysis", "Methodical breakdown"]
    })

    # MultiAgentSystem recommendation
    multi_suitable = is_research_question and len(section_types) > 2
    multi_reason = (
        f"MultiAgentSystem: High-level research question with structured document, ideal for collaborative analysis."
        if multi_suitable else
        f"MultiAgentSystem: Question appears specific rather than high-level, better suited for simpler analysis."
    )

    recommendations.append({
        "agent_name": "MultiAgentSystem",
        "agent_type": "multi_agent",
        "is_suitable": multi_suitable,
        "reason": multi_reason,
        "is_research_question": is_research_question,
        "section_count": len(section_types),
        "confidence": 0.9 if multi_suitable else 0.4,
        "best_for": ["Research questions", "Comprehensive analysis", "Multiple perspectives"]
    })

    # Sort by confidence
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    return recommendations


class AgentOrchestrator:
    """High-level orchestrator for managing multiple agent analyses."""

    def __init__(self, model_identifier: str = "deepseek-chat"):
        """Initialize the orchestrator.

        Args:
            model_identifier: Default model to use for agents
        """
        self.model_identifier = model_identifier

    async def comprehensive_analysis(
        self,
        document_text: str,
        question: str,
        agent_types: List[str] | None = None,
        document_chunks: List[Dict[str, Any]] | None = None,
        metadata: Dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis using multiple agent types.

        Args:
            document_text: The full text of the document
            question: Analysis question or prompt
            agent_types: List of agent types to use (default: all suitable agents)
            document_chunks: Optional list of document chunks
            metadata: Optional document metadata
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with results from all agents and a synthesis
        """
        if agent_types is None:
            # Use all agents
            agent_types = ["simple", "tool_calling", "multi_agent"]

        # Filter to only available agent types
        available_types = ["simple", "tool_calling", "multi_agent"]
        agent_types = [at for at in agent_types if at in available_types]

        if not agent_types:
            raise ValueError("No valid agent types specified")

        # Get recommendations to see which agents are suitable
        recommendations = await get_agent_recommendations(
            document_text=document_text,
            question=question,
            document_chunks=document_chunks,
            model_identifier=self.model_identifier
        )

        # Filter agent types based on recommendations
        suitable_agents = []
        for agent_type in agent_types:
            rec = next((r for r in recommendations if r["agent_type"] == agent_type), None)
            if rec and rec["is_suitable"]:
                suitable_agents.append(agent_type)

        # If no agents are suitable, use simple agent as fallback
        if not suitable_agents:
            suitable_agents = ["simple"]

        # Execute all suitable agents
        results = {}
        execution_times = {}

        for agent_type in suitable_agents:
            try:
                import time
                start_time = time.time()

                result = await create_agent_analysis(
                    document_text=document_text,
                    question=question,
                    agent_type=agent_type,
                    model_identifier=self.model_identifier,
                    document_chunks=document_chunks,
                    metadata=metadata,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                execution_time = time.time() - start_time
                results[agent_type] = result
                execution_times[agent_type] = execution_time

            except Exception as e:
                from ..logging_config import get_logger
                logger = get_logger(__name__)
                logger.error(f"Agent {agent_type} failed: {str(e)}")
                results[agent_type] = {"error": str(e), "success": False}
                execution_times[agent_type] = 0

        # Prepare results summary
        analysis_summary = {
            "question": question,
            "agents_used": list(results.keys()),
            "successful_analyses": {k: v for k, v in results.items() if not isinstance(v, dict) or v.get("success", True)},
            "failed_analyses": {k: v for k, v in results.items() if isinstance(v, dict) and not v.get("success", True)},
            "results": {},
            "execution_times": execution_times,
            "recommendations": recommendations,
            "metadata": metadata or {}
        }

        # Add individual results
        for agent_type, result in results.items():
            if isinstance(result, dict) and "error" in result:
                analysis_summary["results"][agent_type] = {
                    "success": False,
                    "error": result["error"],
                    "execution_time": execution_times[agent_type]
                }
            else:
                analysis_summary["results"][agent_type] = {
                    "success": True,
                    "content": result.content if hasattr(result, 'content') else str(result),
                    "execution_time": execution_times[agent_type],
                    "metadata": getattr(result, 'metadata', {}),
                }

        # Generate recommendations
        successful_analyses = [k for k, v in results.items() if not isinstance(v, dict) or v.get("success", True)]
        if successful_analyses:
            # Best agent = fastest successful
            best_agent = min(successful_analyses, key=lambda x: execution_times[x])
            analysis_summary["recommendations"]["best_agent"] = best_agent

            # Most detailed = agent with longest content
            most_detailed = max(
                successful_analyses,
                key=lambda x: len(results[x].content) if hasattr(results[x], 'content') else 0
            )
            analysis_summary["recommendations"]["most_detailed"] = most_detailed

        return analysis_summary

    async def get_optimal_agent(
        self,
        document_text: str,
        question: str,
        document_chunks: List[Dict[str, Any]] | None = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Get the optimal agent for the given document and question.

        Args:
            document_text: The full text of the document
            question: Analysis question or prompt
            document_chunks: Optional list of document chunks

        Returns:
            Tuple of (agent_function_name, agent_display_name, selection_metadata)
        """
        return AgentSelector.select_agent(
            document_text=document_text,
            question=question,
            document_chunks=document_chunks,
            model_identifier=self.model_identifier
        )