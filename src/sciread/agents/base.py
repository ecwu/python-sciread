"""Base classes and interfaces for agent implementations."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional

from ..document.models import Chunk
from ..llm_provider import get_model


@dataclass
class AgentConfig:
    """Configuration for agent execution."""

    model_identifier: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    timeout: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    include_metadata: bool = True
    track_processing: bool = True


@dataclass
class AgentResult:
    """Result from agent execution."""

    content: str
    agent_name: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    chunks_processed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
            "chunks_processed": self.chunks_processed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @property
    def summary(self) -> str:
        """Get a summary of the result."""
        if self.success:
            return f"{self.agent_name} completed successfully in {self.execution_time:.2f}s"
        else:
            return f"{self.agent_name} failed: {self.error_message}"


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent with configuration."""
        self.config = config or AgentConfig()
        self.model = get_model(self.config.model_identifier)
        self.name = self.__class__.__name__

    @abstractmethod
    async def analyze(self, document: Any, question: str, **kwargs) -> AgentResult:
        """Analyze a document and return results.

        Args:
            document: Document instance or text content
            question: Analysis question or prompt
            **kwargs: Additional arguments for specific agents

        Returns:
            AgentResult with analysis content and metadata
        """
        pass

    @abstractmethod
    def get_supported_questions(self) -> list[str]:
        """Get list of supported question types."""
        pass

    def validate_input(self, document: Any, question: str) -> bool:
        """Validate input parameters.

        Args:
            document: Document instance or text content
            question: Analysis question or prompt

        Returns:
            True if input is valid, False otherwise
        """
        if not question or not question.strip():
            return False

        if document is None:
            return False

        return True

    def prepare_context(self, document: Any, chunks: Optional[list[Chunk]] = None) -> str:
        """Prepare document context for analysis.

        Args:
            document: Document instance or text content
            chunks: Optional list of chunks to focus on

        Returns:
            Formatted context string
        """
        # Handle different document types
        if hasattr(document, 'text'):
            # Document instance
            if chunks:
                context_parts = []
                for chunk in chunks:
                    context_parts.append(f"[{chunk.chunk_type.upper()}]\n{chunk.content}")
                return "\n\n".join(context_parts)
            else:
                return document.text
        elif isinstance(document, str):
            # Raw text
            return document
        elif isinstance(document, list) and document and isinstance(document[0], Chunk):
            # List of chunks
            context_parts = []
            for chunk in document:
                context_parts.append(f"[{chunk.chunk_type.upper()}]\n{chunk.content}")
            return "\n\n".join(context_parts)
        else:
            return str(document)

    def log_execution(self, result: AgentResult) -> None:
        """Log execution result for debugging and monitoring.

        Args:
            result: Agent execution result
        """
        from ..logging_config import get_logger

        logger = get_logger(__name__)
        if result.success:
            logger.info(
                f"{result.agent_name} completed successfully",
                extra={
                    "execution_time": result.execution_time,
                    "chunks_processed": result.chunks_processed,
                    "content_length": len(result.content),
                }
            )
        else:
            logger.error(
                f"{result.agent_name} failed: {result.error_message}",
                extra={
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                }
            )

    async def execute_with_retry(self, prompt: str, **model_kwargs) -> str:
        """Execute model with retry logic.

        Args:
            prompt: The prompt to send to the model
            **model_kwargs: Additional model parameters

        Returns:
            Model response text

        Raises:
            Exception: If all retries are exhausted
        """
        import asyncio
        from ..logging_config import get_logger

        logger = get_logger(__name__)
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Use pydantic-ai's run method
                response = await self.model.run(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **model_kwargs
                )
                return response.data

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Model execution attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.max_retries + 1} model execution attempts failed: {e}")

        raise last_exception

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.name}(model={self.config.model_identifier})"