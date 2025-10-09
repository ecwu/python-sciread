"""Factory functions for creating and selecting appropriate agents."""

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from .base import AgentConfig
from .multi_agent_system import MultiAgentSystem
from .simple_agent import SimpleAgent
from .tool_calling_agent import ToolCallingAgent


class AgentSelector:
    """Utility class for selecting the appropriate agent based on document and question characteristics."""

    @staticmethod
    def select_agent(
        document: Any,
        question: str,
        config: Optional[AgentConfig] = None,
        force_agent: Optional[str] = None
    ) -> Tuple[Any, str]:
        """Select the most appropriate agent for the given document and question.

        Args:
            document: Document instance or text content
            question: Analysis question or prompt
            config: Optional configuration for agents
            force_agent: Force a specific agent type ('simple', 'tool_calling', 'multi_agent')

        Returns:
            Tuple of (agent_instance, selection_reason)
        """
        if force_agent:
            return AgentSelector._create_forced_agent(force_agent, config)

        # Try agents in order of complexity, checking suitability
        agents = [
            (SimpleAgent(config), "SimpleAgent"),
            (ToolCallingAgent(config), "ToolCallingAgent"),
            (MultiAgentSystem(config), "MultiAgentSystem")
        ]

        for agent, agent_name in agents:
            if hasattr(agent, 'is_suitable_for_document'):
                is_suitable, reason = agent.is_suitable_for_document(
                    document,
                    research_question=question
                )
                if is_suitable:
                    return agent, f"{agent_name}: {reason}"

        # Fallback to SimpleAgent if none are suitable
        simple_agent = SimpleAgent(config)
        return simple_agent, "SimpleAgent: Fallback option - no specific requirements matched"

    @staticmethod
    def _create_forced_agent(agent_type: str, config: Optional[AgentConfig] = None) -> Tuple[Any, str]:
        """Create a specific agent type.

        Args:
            agent_type: Type of agent to create
            config: Optional configuration

        Returns:
            Tuple of (agent_instance, selection_reason)
        """
        config = config or AgentConfig()

        agent_map = {
            "simple": SimpleAgent(config),
            "tool_calling": ToolCallingAgent(config),
            "multi_agent": MultiAgentSystem(config),
        }

        if agent_type not in agent_map:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_map.keys())}")

        agent = agent_map[agent_type]
        reason = f"Forced selection of {agent_type.title().replace('_', '')}Agent"
        return agent, reason


def create_agent(
    agent_type: str = "auto",
    config: Optional[AgentConfig] = None,
    **kwargs
) -> Any:
    """Create an agent instance.

    Args:
        agent_type: Type of agent ('simple', 'tool_calling', 'multi_agent', 'auto')
        config: Optional configuration for the agent
        **kwargs: Additional arguments for agent creation

    Returns:
        Agent instance
    """
    config = config or AgentConfig()

    if agent_type == "auto":
        # Return a selector function that will choose the appropriate agent
        def auto_analyzer(document: Any, question: str, **analyze_kwargs):
            agent, reason = AgentSelector.select_agent(document, question, config)
            from ..logging_config import get_logger
            logger = get_logger(__name__)
            logger.info(f"Auto-selected agent: {reason}")
            return agent.analyze(document, question, **analyze_kwargs)

        return auto_analyzer

    return AgentSelector._create_forced_agent(agent_type, config)[0]


def analyze_document(
    document: Any,
    question: str,
    agent_type: str = "auto",
    config: Optional[AgentConfig] = None,
    **kwargs
) -> Any:
    """Convenience function to analyze a document with automatic agent selection.

    Args:
        document: Document instance or text content
        question: Analysis question or prompt
        agent_type: Type of agent to use ('simple', 'tool_calling', 'multi_agent', 'auto')
        config: Optional configuration for the agent
        **kwargs: Additional arguments passed to the agent

    Returns:
        AgentResult from the analysis
    """
    if agent_type == "auto":
        agent, reason = AgentSelector.select_agent(document, question, config)
        from ..logging_config import get_logger
        logger = get_logger(__name__)
        logger.info(f"Auto-selected agent: {reason}")
        return agent.analyze(document, question, **kwargs)
    else:
        agent = create_agent(agent_type, config)
        return agent.analyze(document, question, **kwargs)


def get_agent_recommendations(
    document: Any,
    question: str,
    config: Optional[AgentConfig] = None
) -> List[dict]:
    """Get recommendations for all available agents with their suitability.

    Args:
        document: Document instance or text content
        question: Analysis question or prompt
        config: Optional configuration for agents

    Returns:
        List of agent recommendations with suitability ratings
    """
    config = config or AgentConfig()
    recommendations = []

    agents = [
        SimpleAgent(config),
        ToolCallingAgent(config),
        MultiAgentSystem(config)
    ]

    for agent in agents:
        recommendation = {
            "agent_name": agent.name,
            "agent_type": agent.__class__.__name__,
            "supported_questions": agent.get_supported_questions(),
        }

        if hasattr(agent, 'is_suitable_for_document'):
            # For MultiAgentSystem, pass the research_question
            is_suitable_kwargs = {}
            if isinstance(agent, MultiAgentSystem):
                is_suitable_kwargs["research_question"] = question

            is_suitable, reason = agent.is_suitable_for_document(
                document,
                **is_suitable_kwargs
            )
            recommendation["is_suitable"] = is_suitable
            recommendation["reason"] = reason
        else:
            # For SimpleAgent, always suitable unless document is invalid
            recommendation["is_suitable"] = True
            recommendation["reason"] = "SimpleAgent is suitable for most document types"

        # Add additional information if available
        if hasattr(agent, 'estimate_tokens'):
            try:
                estimated_tokens = agent.estimate_tokens(document, question=question)
                recommendation["estimated_tokens"] = estimated_tokens
            except:
                pass

        if hasattr(agent, 'get_analysis_plan'):
            try:
                analysis_plan = agent.get_analysis_plan(document, question)
                recommendation["analysis_plan"] = analysis_plan
            except:
                pass

        recommendations.append(recommendation)

    return recommendations


class AgentOrchestrator:
    """High-level orchestrator for managing multiple agent analyses."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the orchestrator.

        Args:
            config: Optional configuration for agents
        """
        self.config = config or AgentConfig()

    async def comprehensive_analysis(
        self,
        document: Any,
        question: str,
        agent_types: Optional[List[str]] = None,
        **kwargs
    ) -> dict:
        """Perform comprehensive analysis using multiple agent types.

        Args:
            document: Document instance or text content
            question: Analysis question or prompt
            agent_types: List of agent types to use (default: all suitable agents)
            **kwargs: Additional arguments for agents

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

        # Create agents
        agents = []
        for agent_type in agent_types:
            agent = create_agent(agent_type, self.config)
            agents.append(agent)

        # Execute all agents
        import asyncio
        results = []

        for agent in agents:
            try:
                result = await agent.analyze(document, question, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                from .base import AgentResult
                error_result = AgentResult(
                    content="",
                    agent_name=agent.name,
                    execution_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)

        # Prepare results summary
        analysis_summary = {
            "question": question,
            "agents_used": [result.agent_name for result in results],
            "successful_analyses": [r for r in results if r.success],
            "failed_analyses": [r for r in results if not r.success],
            "results": {},
            "recommendations": {}
        }

        # Add individual results
        for result in results:
            analysis_summary["results"][result.agent_name] = {
                "success": result.success,
                "content": result.content,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "metadata": result.metadata,
            }

        # Generate recommendations
        if analysis_summary["successful_analyses"]:
            analysis_summary["recommendations"]["best_agent"] = (
                min(analysis_summary["successful_analyses"], key=lambda x: x.execution_time).agent_name
            )
            analysis_summary["recommendations"]["most_detailed"] = (
                max(analysis_summary["successful_analyses"], key=lambda x: len(x.content)).agent_name
            )

        return analysis_summary

    def get_optimal_agent(self, document: Any, question: str) -> Tuple[Any, str]:
        """Get the optimal agent for the given document and question.

        Args:
            document: Document instance or text content
            question: Analysis question or prompt

        Returns:
            Tuple of (agent_instance, reason)
        """
        return AgentSelector.select_agent(document, question, self.config)