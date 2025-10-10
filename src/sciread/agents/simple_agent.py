"""SimpleAgent implementation for direct full document processing."""

import time
from typing import Any
from typing import Optional

from .base import Agent
from .base import AgentConfig
from .base import AgentResult
from .prompts import get_simple_analysis_prompt
from .prompts import remove_citations_section


class SimpleAgent(Agent):
    """Simple agent that processes the full document with a single LLM call using the Feynman technique.

    This agent takes the entire document content (or selected chunks) and sends
    it to the LLM with a user-defined question. It uses the Feynman technique
    to create detailed explanations as if written by the paper's author.
    Automatically removes citation sections to save tokens.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the SimpleAgent.

        Args:
            config: Optional configuration for the agent
        """
        super().__init__(config or AgentConfig())
        self.name = "SimpleAgent"

    async def analyze(self, document: Any, question: str, **kwargs) -> AgentResult:
        """Analyze a document with a single LLM call.

        Args:
            document: Document instance, text content, or list of chunks
            question: Analysis question or prompt
            **kwargs: Additional arguments (chunks, max_length, etc.)

        Returns:
            AgentResult with analysis content and metadata
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not self.validate_input(document, question):
                return AgentResult(
                    content="",
                    agent_name=self.name,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid input: document and question must be provided",
                )

            # Prepare document context
            chunks = kwargs.get('chunks')
            max_length = kwargs.get('max_length')

            context = self.prepare_context(document, chunks)

            # Remove citations section to save tokens
            context = remove_citations_section(context)

            # Apply length limit if specified
            if max_length and len(context) > max_length:
                context = context[:max_length] + "...[truncated]"

            # Format the prompt
            prompt = get_simple_analysis_prompt().format(
                context=context,
                question=question
            )

            # Execute the model
            response = await self.execute_with_retry(prompt)

            # Count processed chunks
            chunks_processed = 0
            if chunks:
                chunks_processed = len(chunks)
            elif hasattr(document, 'chunks'):
                chunks_processed = len(document.chunks)
            elif isinstance(document, list) and document and hasattr(document[0], 'content'):
                chunks_processed = len(document)

            # Create result
            execution_time = time.time() - start_time
            result = AgentResult(
                content=response,
                agent_name=self.name,
                execution_time=execution_time,
                success=True,
                chunks_processed=chunks_processed,
                metadata={
                    "question": question,
                    "context_length": len(context),
                    "response_length": len(response),
                    "max_length": max_length,
                    "model": self.config.model_identifier,
                    "temperature": self.config.temperature,
                }
            )

            # Log execution
            self.log_execution(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Analysis failed: {str(e)}"

            result = AgentResult(
                content="",
                agent_name=self.name,
                execution_time=execution_time,
                success=False,
                error_message=error_message,
                metadata={
                    "question": question,
                    "error_type": type(e).__name__,
                }
            )

            self.log_execution(result)
            return result

    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types.

        Returns:
            List of supported question types
        """
        return [
            "general_summary",
            "key_contributions",
            "methodology_overview",
            "main_findings",
            "research_questions",
            "limitations",
            "future_work",
            "comparison",
            "evaluation",
            "custom_question"  # Any custom question
        ]

    def supports_document_type(self, document: Any) -> bool:
        """Check if the agent supports the given document type.

        Args:
            document: Document instance, text, or chunks

        Returns:
            True if supported, False otherwise
        """
        # SimpleAgent supports most document types as long as they can be converted to text
        return True

    def estimate_tokens(self, document: Any, **kwargs) -> int:
        """Estimate the number of tokens needed for analysis.

        Args:
            document: Document instance, text, or chunks
            **kwargs: Additional arguments

        Returns:
            Estimated token count
        """
        context = self.prepare_context(document, kwargs.get('chunks'))

        # Rough estimation: ~4 characters per token for English text
        context_tokens = len(context) // 4

        # Add prompt template tokens
        prompt_template = get_simple_analysis_prompt()
        template_tokens = len(prompt_template) // 4

        # Add question tokens
        question = kwargs.get('question', '')
        question_tokens = len(question) // 4

        # Add some buffer for response generation
        response_buffer = 1000

        total_tokens = context_tokens + template_tokens + question_tokens + response_buffer

        return total_tokens

    def is_suitable_for_document(self, document: Any, **kwargs) -> tuple[bool, str]:
        """Check if this agent is suitable for the given document.

        Args:
            document: Document instance, text, or chunks
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_suitable, reason)
        """
        # Estimate token requirements
        estimated_tokens = self.estimate_tokens(document, **kwargs)

        # Assume typical model context limit of 128k tokens
        context_limit = 128000

        if estimated_tokens > context_limit:
            return (
                False,
                f"Document too large for simple processing (estimated {estimated_tokens:,} tokens > {context_limit:,} limit). "
                "Consider using ToolCallingAgent with section-based processing."
            )

        # Check if document is already split into appropriate chunks
        if hasattr(document, 'chunks') and len(document.chunks) > 20:
            return (
                False,
                f"Document has {len(document.chunks)} chunks, which may be better suited for "
                "section-based processing with ToolCallingAgent."
            )

        return (
            True,
            f"Document suitable for simple processing (estimated {estimated_tokens:,} tokens)."
        )

    async def batch_analyze(
        self,
        documents: list[Any],
        questions: list[str],
        **kwargs
    ) -> list[AgentResult]:
        """Analyze multiple documents with multiple questions.

        Args:
            documents: List of documents to analyze
            questions: List of questions to ask
            **kwargs: Additional arguments

        Returns:
            List of AgentResult objects
        """
        results = []

        for doc in documents:
            for question in questions:
                result = await self.analyze(doc, question, **kwargs)
                results.append(result)

        return results

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"SimpleAgent(model={self.config.model_identifier})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (
            f"SimpleAgent(model='{self.config.model_identifier}', "
            f"temperature={self.config.temperature}, "
            f"max_tokens={self.config.max_tokens})"
        )