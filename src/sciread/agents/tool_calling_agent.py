"""ToolCallingAgent implementation with controller-based section analysis."""

import asyncio
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from .base import Agent
from .base import AgentConfig
from .base import AgentResult
from .prompts import get_controller_prompt
from .prompts import get_section_specific_prompt
from .prompts import get_synthesis_prompt


class SectionAgent(Agent):
    """Specialized agent for analyzing specific document sections."""

    def __init__(self, section_type: str, config: Optional[AgentConfig] = None):
        """Initialize a section-specific agent.

        Args:
            section_type: Type of section this agent handles
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig())
        self.section_type = section_type
        self.name = f"{section_type.title()}Agent"

    async def analyze(self, document: Any, question: str, **kwargs) -> AgentResult:
        """Analyze a specific section of the document.

        Args:
            document: Document instance or chunks
            question: Analysis question or task
            **kwargs: Additional arguments (section_chunks, etc.)

        Returns:
            AgentResult with section-specific analysis
        """
        start_time = time.time()

        try:
            # Get section-specific chunks or context
            section_chunks = kwargs.get('section_chunks', [])
            context = self.prepare_context(document, section_chunks)

            # Get section-specific prompt
            prompt_template = get_section_specific_prompt(self.section_type)
            prompt = prompt_template.format(
                context=context,
                task=question,
                section_type=self.section_type
            )

            # Execute the model
            response = await self.execute_with_retry(prompt)

            execution_time = time.time() - start_time

            return AgentResult(
                content=response,
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                chunks_processed=len(section_chunks),
                metadata={
                    "section_type": self.section_type,
                    "question": question,
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
                error_message=f"Section analysis failed: {str(e)}",
                metadata={
                    "section_type": self.section_type,
                    "error_type": type(e).__name__,
                }
            )

    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types for this section."""
        return ["section_analysis", "content_extraction", "key_insights"]


class ControllerAgent(Agent):
    """Controller agent that coordinates section-specific analyses."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the controller agent.

        Args:
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig())
        self.name = "ControllerAgent"

    async def analyze(self, document: Any, question: str, **kwargs) -> AgentResult:
        """Coordinate the analysis of multiple document sections.

        Args:
            document: Document instance
            question: Overall analysis question
            **kwargs: Additional arguments

        Returns:
            AgentResult with coordinated analysis
        """
        start_time = time.time()

        try:
            # Get abstract for initial understanding
            abstract_chunks = []
            if hasattr(document, 'get_chunks'):
                abstract_chunks = document.get_chunks(chunk_type="abstract")
            elif isinstance(document, list):
                abstract_chunks = [c for c in document if getattr(c, 'chunk_type', '') == 'abstract']

            abstract_text = ""
            if abstract_chunks:
                abstract_text = abstract_chunks[0].content

            # Get available sections from document
            available_sections = self._get_available_sections(document)

            # Create analysis plan
            prompt = get_controller_prompt(available_sections)
            prompt = prompt.format(abstract=abstract_text, task=question)

            plan_response = await self.execute_with_retry(prompt)

            # Determine which sections to analyze based on the plan
            sections_to_analyze = self._extract_sections_from_plan(plan_response, available_sections)

            # Execute section analyses in parallel
            section_tasks = []
            for section_type in sections_to_analyze:
                task = self._analyze_section(document, section_type, question)
                section_tasks.append(task)

            section_results = await asyncio.gather(*section_tasks, return_exceptions=True)

            # Process section results
            section_analyses = {}
            for i, result in enumerate(section_results):
                if isinstance(result, Exception):
                    section_type = sections_to_analyze[i]
                    section_analyses[section_type] = f"Analysis failed: {str(result)}"
                elif isinstance(result, AgentResult):
                    section_type = sections_to_analyze[i]
                    section_analyses[section_type] = result.content if result.success else result.error_message

            # Synthesize results
            synthesis_prompt = get_synthesis_prompt(section_analyses)
            synthesis_prompt = synthesis_prompt.format(task=question)

            final_response = await self.execute_with_retry(synthesis_prompt)

            execution_time = time.time() - start_time

            return AgentResult(
                content=final_response,
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                chunks_processed=len([c for c in getattr(document, 'chunks', []) if c.chunk_type in sections_to_analyze]),
                metadata={
                    "question": question,
                    "analysis_plan": plan_response,
                    "sections_analyzed": sections_to_analyze,
                    "section_analyses": section_analyses,
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
                error_message=f"Controller analysis failed: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                }
            )

    def _get_available_sections(self, document: Any) -> List[str]:
        """Get list of available sections in the document.

        Args:
            document: Document instance

        Returns:
            List of available section types
        """
        sections = set()

        if hasattr(document, 'get_chunks'):
            chunks = document.chunks
        elif isinstance(document, list):
            chunks = document
        else:
            return []

        for chunk in chunks:
            if hasattr(chunk, 'chunk_type'):
                sections.add(chunk.chunk_type)

        # Filter to known section types
        known_sections = {
            'abstract', 'introduction', 'methods', 'methodology',
            'experiments', 'results', 'conclusion', 'related_work',
            'discussion', 'references', 'acknowledgments'
        }

        return list(sections.intersection(known_sections))

    def _extract_sections_from_plan(self, plan_response: str, available_sections: List[str]) -> List[str]:
        """Extract sections to analyze from the controller's plan.

        Args:
            plan_response: The controller's analysis plan
            available_sections: List of available sections in document

        Returns:
            List of sections to analyze
        """
        sections_to_analyze = []

        # Simple heuristic: include sections mentioned in the plan
        plan_lower = plan_response.lower()
        for section in available_sections:
            if section.lower() in plan_lower:
                sections_to_analyze.append(section)

        # If no sections were specifically mentioned, use default priority order
        if not sections_to_analyze:
            priority_order = ['abstract', 'introduction', 'methods', 'results', 'conclusion']
            for section in priority_order:
                if section in available_sections:
                    sections_to_analyze.append(section)

        # Limit to reasonable number of sections to avoid excessive processing
        max_sections = 5
        return sections_to_analyze[:max_sections]

    async def _analyze_section(self, document: Any, section_type: str, question: str) -> AgentResult:
        """Analyze a specific section.

        Args:
            document: Document instance
            section_type: Type of section to analyze
            question: Overall analysis question

        Returns:
            AgentResult for the section
        """
        # Create section-specific agent
        section_agent = SectionAgent(section_type, self.config)

        # Get section chunks
        section_chunks = []
        if hasattr(document, 'get_chunks'):
            section_chunks = document.get_chunks(chunk_type=section_type)
        elif isinstance(document, list):
            section_chunks = [c for c in document if getattr(c, 'chunk_type', '') == section_type]

        # Create section-specific task
        section_task = f"Analyze this {section_type} section to help answer: {question}"

        # Execute section analysis
        return await section_agent.analyze(
            document,
            section_task,
            section_chunks=section_chunks
        )

    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types."""
        return [
            "comprehensive_analysis",
            "research_questions",
            "methodology_analysis",
            "findings_synthesis",
            "contribution_evaluation",
            "comparative_analysis",
        ]


class ToolCallingAgent:
    """Tool-calling agent system with controller and section-specific sub-agents.

    This agent uses a controller to analyze the abstract and determine which
    sections need detailed analysis, then coordinates multiple specialized
    sub-agents to analyze different parts of the document.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the tool-calling agent system.

        Args:
            config: Optional configuration for the agents
        """
        self.config = config or AgentConfig()
        self.controller = ControllerAgent(self.config)
        self.name = "ToolCallingAgent"

    async def analyze(self, document: Any, question: str, **kwargs) -> AgentResult:
        """Analyze a document using coordinated section analysis.

        Args:
            document: Document instance
            question: Analysis question
            **kwargs: Additional arguments

        Returns:
            AgentResult with comprehensive analysis
        """
        return await self.controller.analyze(document, question, **kwargs)

    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types."""
        return self.controller.get_supported_questions()

    def is_suitable_for_document(self, document: Any, **kwargs) -> tuple[bool, str]:
        """Check if this agent system is suitable for the document.

        Args:
            document: Document instance
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_suitable, reason)
        """
        # Check if document is split into sections
        if hasattr(document, 'chunks') and len(document.chunks) > 5:
            # Check if chunks have different types
            chunk_types = set(chunk.chunk_type for chunk in document.chunks)
            if len(chunk_types) > 2:  # More than just 'unknown' and one other type
                return (
                    True,
                    f"Document has {len(document.chunks)} chunks across {len(chunk_types)} sections, "
                    "well-suited for section-based analysis."
                )

        # Check if document has identifiable sections
        available_sections = self.controller._get_available_sections(document)
        if len(available_sections) >= 3:
            return (
                True,
                f"Document has {len(available_sections)} identifiable sections, suitable for section-based analysis."
            )

        return (
            False,
            "Document lacks clear section structure, better suited for SimpleAgent."
        )

    def get_analysis_plan(self, document: Any, question: str) -> str:
        """Get the planned analysis approach without executing it.

        Args:
            document: Document instance
            question: Analysis question

        Returns:
            Description of the planned analysis approach
        """
        available_sections = self.controller._get_available_sections(document)

        if not available_sections:
            return "No identifiable sections found. Document may need SimpleAgent instead."

        return (
            f"ToolCallingAgent will analyze {len(available_sections)} sections: "
            f"{', '.join(available_sections)}. "
            "Controller will coordinate section-specific analyses and synthesize results."
        )

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"ToolCallingAgent(model={self.config.model_identifier})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (
            f"ToolCallingAgent(model='{self.config.model_identifier}', "
            f"temperature={self.config.temperature}, "
            f"max_tokens={self.config.max_tokens})"
        )