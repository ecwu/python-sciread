"""MultiAgentSystem implementation with collaborative agents for high-level research questions."""

import asyncio
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .base import Agent
from .base import AgentConfig
from .base import AgentResult
from .prompts import get_collaborative_agent_prompt
from .prompts import get_final_synthesis_prompt
from .prompts import get_research_question_prompts


class ResearchQuestionAgent(Agent):
    """Specialized agent for analyzing specific research questions."""

    def __init__(self, question_type: str, role_description: str, config: Optional[AgentConfig] = None):
        """Initialize a research question-specific agent.

        Args:
            question_type: Type of research question this agent handles
            role_description: Description of the agent's role and focus
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig())
        self.question_type = question_type
        self.role_description = role_description
        self.name = f"{question_type.title().replace('_', '')}Agent"

    async def analyze(self, document: Any, research_question: str, **kwargs) -> AgentResult:
        """Analyze document from the perspective of this agent's role.

        Args:
            document: Document instance or text content
            research_question: The high-level research question
            **kwargs: Additional arguments (other_agents, context, etc.)

        Returns:
            AgentResult with role-specific analysis
        """
        start_time = time.time()

        try:
            # Prepare context
            context = self.prepare_context(document)

            # Get other agents' roles for collaboration context
            other_agents = kwargs.get('other_agents', [])
            other_roles = [agent.name for agent in other_agents if agent.name != self.name]

            # Create collaborative prompt
            prompt = get_collaborative_agent_prompt(self.name, other_roles)
            prompt = prompt.format(
                context=context,
                research_question=research_question,
                role_specific_focus=self.role_description
            )

            # Execute the model
            response = await self.execute_with_retry(prompt)

            execution_time = time.time() - start_time

            return AgentResult(
                content=response,
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                chunks_processed=len(getattr(document, 'chunks', [])),
                metadata={
                    "question_type": self.question_type,
                    "research_question": research_question,
                    "role_description": self.role_description,
                    "other_agents": other_roles,
                    "context_length": len(context),
                    "response_length": len(response),
                    "model": self.config.model_identifier,
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                content="",
                agent_name=self.name,
                execution_time=execution_time,
                success=False,
                error_message=f"Agent analysis failed: {str(e)}",
                metadata={
                    "question_type": self.question_type,
                    "error_type": type(e).__name__,
                }
            )

    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types."""
        return [self.question_type]

    def get_role_focus(self) -> str:
        """Get the specific focus area of this agent."""
        return self.role_description


class CoordinatorAgent(Agent):
    """Coordinator agent that manages the multi-agent collaboration process."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the coordinator agent.

        Args:
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig())
        self.name = "CoordinatorAgent"

    async def analyze(self, document: Any, research_question: str, **kwargs) -> AgentResult:
        """Coordinate multi-agent analysis and synthesize results.

        Args:
            document: Document instance
            research_question: High-level research question
            **kwargs: Additional arguments

        Returns:
            AgentResult with coordinated analysis
        """
        start_time = time.time()

        try:
            # Get the research question agents
            agents = kwargs.get('agents', [])
            if not agents:
                # Create default agents if none provided
                agents = self._create_default_agents()

            # Execute all agents in parallel
            agent_tasks = []
            for agent in agents:
                task = agent.analyze(
                    document,
                    research_question,
                    other_agents=agents
                )
                agent_tasks.append(task)

            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            # Process agent results
            agent_analyses = {}
            for i, result in enumerate(agent_results):
                agent = agents[i]
                if isinstance(result, Exception):
                    agent_analyses[agent.name] = f"Analysis failed: {str(result)}"
                elif isinstance(result, AgentResult):
                    agent_analyses[agent.name] = result.content if result.success else result.error_message

            # Synthesize final analysis
            synthesis_prompt = get_final_synthesis_prompt(agent_analyses)
            synthesis_prompt = synthesis_prompt.format(research_question=research_question)

            final_response = await self.execute_with_retry(synthesis_prompt)

            execution_time = time.time() - start_time

            return AgentResult(
                content=final_response,
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                chunks_processed=len(getattr(document, 'chunks', [])),
                metadata={
                    "research_question": research_question,
                    "agents_used": [agent.name for agent in agents],
                    "agent_analyses": agent_analyses,
                    "model": self.config.model_identifier,
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                content="",
                agent_name=self.name,
                execution_time=execution_time,
                success=False,
                error_message=f"Coordinator analysis failed: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                }
            )

    def _create_default_agents(self) -> List[ResearchQuestionAgent]:
        """Create default set of research question agents.

        Returns:
            List of default research question agents
        """
        prompts = get_research_question_prompts()

        agents = []
        for question_type, prompt in prompts.items():
            # Extract role focus from the prompt
            role_focus = self._extract_role_focus(prompt, question_type)
            agent = ResearchQuestionAgent(question_type, role_focus, self.config)
            agents.append(agent)

        return agents

    def _extract_role_focus(self, prompt: str, question_type: str) -> str:
        """Extract role focus from the prompt template.

        Args:
            prompt: Prompt template
            question_type: Type of research question

        Returns:
            Role focus description
        """
        # Define role focuses based on question types
        role_focuses = {
            "research_question": "Identify and analyze the core research question(s) that the paper addresses",
            "motivation": "Analyze why the authors chose this research topic and what makes it important",
            "methodology": "Examine how the authors conducted their research and what methods they used",
            "findings": "Investigate what the authors discovered and what results they obtained",
            "contribution": "Evaluate the main contributions and impact of the research",
        }

        return role_focuses.get(question_type, f"Analyze the {question_type.replace('_', ' ')} of the research")

    def get_supported_questions(self) -> list[str]:
        """Get list of supported high-level research questions."""
        return [
            "research_question",
            "motivation",
            "methodology",
            "findings",
            "contribution",
            "comprehensive_analysis"
        ]


class MultiAgentSystem:
    """Multi-agent system for collaborative analysis of high-level research questions.

    This system coordinates multiple specialized agents, each focusing on a specific
    aspect of research analysis (questions, motivation, methods, findings, contributions).
    The agents collaborate to provide comprehensive answers to complex research questions.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the multi-agent system.

        Args:
            config: Optional configuration for the agents
        """
        self.config = config or AgentConfig()
        self.coordinator = CoordinatorAgent(self.config)
        self.name = "MultiAgentSystem"

    async def analyze(self, document: Any, research_question: str, **kwargs) -> AgentResult:
        """Analyze a document using collaborative multi-agent approach.

        Args:
            document: Document instance
            research_question: High-level research question
            **kwargs: Additional arguments (agents, question_type, etc.)

        Returns:
            AgentResult with comprehensive collaborative analysis
        """
        # Determine which agents to use based on the research question
        agents = self._select_agents(research_question, kwargs.get('agents'))

        # Add agents to kwargs for coordinator
        kwargs['agents'] = agents

        return await self.coordinator.analyze(document, research_question, **kwargs)

    def _select_agents(self, research_question: str, custom_agents: Optional[List[Agent]] = None) -> List[Agent]:
        """Select appropriate agents for the given research question.

        Args:
            research_question: The research question to analyze
            custom_agents: Optional custom agents to use

        Returns:
            List of agents to use for analysis
        """
        if custom_agents:
            return custom_agents

        # Map question keywords to agent types
        question_mappings = {
            "research_question": ["question", "research question", "problem", "gap", "objective"],
            "motivation": ["why", "motivation", "importance", "significance", "impact", "reason"],
            "methodology": ["how", "method", "approach", "technique", "algorithm", "methodology"],
            "findings": ["what", "result", "finding", "outcome", "discovery", "evidence"],
            "contribution": ["contribution", "advance", "novelty", "innovation", "achievement"],
        }

        # Select agents based on question content
        selected_agent_types = []
        question_lower = research_question.lower()

        for agent_type, keywords in question_mappings.items():
            if any(keyword in question_lower for keyword in keywords):
                selected_agent_types.append(agent_type)

        # If no specific agents identified, use all agents for comprehensive analysis
        if not selected_agent_types:
            selected_agent_types = list(question_mappings.keys())

        # Create the selected agents
        agents = []
        prompts = get_research_question_prompts()

        for agent_type in selected_agent_types:
            if agent_type in prompts:
                role_focus = self.coordinator._extract_role_focus(prompts[agent_type], agent_type)
                agent = ResearchQuestionAgent(agent_type, role_focus, self.config)
                agents.append(agent)

        return agents

    def get_supported_questions(self) -> list[str]:
        """Get list of supported high-level research questions."""
        base_questions = [
            "What is the Research Question?",
            "Why is the author doing this topic?",
            "How did the author do the research?",
            "What did the author get from the result?",
            "What is the main contribution?",
        ]

        question_types = self.coordinator.get_supported_questions()
        return base_questions + question_types

    def is_suitable_for_document(self, document: Any, **kwargs) -> tuple[bool, str]:
        """Check if this multi-agent system is suitable for the document.

        Args:
            document: Document instance
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_suitable, reason)
        """
        # Multi-agent system works well for complex, comprehensive questions
        research_question = kwargs.get('research_question', '')

        if not research_question:
            return (
                False,
                "No research question provided. Multi-agent system needs a specific research question to analyze."
            )

        # Check if the question is high-level and complex
        question_indicators = [
            "research question", "why", "how", "what is the", "motivation",
            "contribution", "findings", "methodology", "approach"
        ]

        question_lower = research_question.lower()
        is_high_level = any(indicator in question_lower for indicator in question_indicators)

        if is_high_level:
            return (
                True,
                "Research question appears to be high-level and complex, well-suited for multi-agent collaborative analysis."
            )

        return (
            False,
            "Research question appears to be specific or factual, better suited for SimpleAgent or ToolCallingAgent."
        )

    def get_agent_descriptions(self) -> Dict[str, str]:
        """Get descriptions of the agents in the system.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        prompts = get_research_question_prompts()
        descriptions = {}

        for question_type, prompt in prompts.items():
            role_focus = self.coordinator._extract_role_focus(prompt, question_type)
            agent_name = question_type.title().replace('_', '') + "Agent"
            descriptions[agent_name] = role_focus

        return descriptions

    async def analyze_with_agent_details(
        self,
        document: Any,
        research_question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze with detailed information about each agent's contribution.

        Args:
            document: Document instance
            research_question: High-level research question
            **kwargs: Additional arguments

        Returns:
            Dictionary with final analysis and agent details
        """
        # Select agents for this question
        agents = self._select_agents(research_question, kwargs.get('agents'))

        # Get individual agent analyses
        agent_tasks = []
        for agent in agents:
            task = agent.analyze(
                document,
                research_question,
                other_agents=agents
            )
            agent_tasks.append(task)

        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Process individual results
        agent_analyses = {}
        for i, result in enumerate(agent_results):
            agent = agents[i]
            if isinstance(result, Exception):
                agent_analyses[agent.name] = {
                    "success": False,
                    "error": str(result),
                    "content": ""
                }
            elif isinstance(result, AgentResult):
                agent_analyses[agent.name] = {
                    "success": result.success,
                    "error": result.error_message,
                    "content": result.content,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata
                }

        # Get final coordinated analysis
        final_result = await self.coordinator.analyze(
            document,
            research_question,
            agents=agents
        )

        return {
            "final_analysis": final_result.content,
            "agent_analyses": agent_analyses,
            "agents_used": [agent.name for agent in agents],
            "research_question": research_question,
            "total_execution_time": sum(
                info.get("execution_time", 0) for info in agent_analyses.values()
            ) + final_result.execution_time,
            "success": final_result.success,
        }

    def __str__(self) -> str:
        """String representation of the multi-agent system."""
        return f"MultiAgentSystem(model={self.config.model_identifier})"

    def __repr__(self) -> str:
        """Detailed string representation of the multi-agent system."""
        return (
            f"MultiAgentSystem(model='{self.config.model_identifier}', "
            f"temperature={self.config.temperature}, "
            f"max_tokens={self.config.max_tokens})"
        )