"""Synchronous tests for agent systems functionality."""

import pytest
from unittest.mock import MagicMock, patch

from sciread.agents import (
    AgentConfig,
    AgentResult,
    AgentOrchestrator,
    SimpleAgent,
    ToolCallingAgent,
    MultiAgentSystem,
    create_agent,
    get_agent_recommendations,
)
from sciread.document.models import Chunk


class TestAgentConfig:
    """Test AgentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.model_identifier == "deepseek-chat"
        assert config.temperature == 0.3
        assert config.max_tokens is None
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.include_metadata is True
        assert config.track_processing is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            model_identifier="gpt-4",
            temperature=0.7,
            max_tokens=4000,
            timeout=600,
        )
        assert config.model_identifier == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.timeout == 600


class TestAgentResult:
    """Test AgentResult class."""

    def test_successful_result(self):
        """Test successful agent result."""
        result = AgentResult(
            content="Analysis content",
            agent_name="TestAgent",
            execution_time=1.5,
            success=True,
            chunks_processed=5,
            metadata={"test": "value"}
        )
        assert result.success is True
        assert result.content == "Analysis content"
        assert result.agent_name == "TestAgent"
        assert result.execution_time == 1.5
        assert result.chunks_processed == 5
        assert result.metadata["test"] == "value"
        assert result.summary == "TestAgent completed successfully in 1.50s"

    def test_failed_result(self):
        """Test failed agent result."""
        result = AgentResult(
            content="",
            agent_name="TestAgent",
            execution_time=0.5,
            success=False,
            error_message="Test error"
        )
        assert result.success is False
        assert result.content == ""
        assert result.error_message == "Test error"
        assert result.summary == "TestAgent failed: Test error"

    def test_to_dict(self):
        """Test result serialization."""
        result = AgentResult(
            content="Test content",
            agent_name="TestAgent",
            execution_time=1.0,
            success=True
        )
        result_dict = result.to_dict()
        assert result_dict["content"] == "Test content"
        assert result_dict["agent_name"] == "TestAgent"
        assert result_dict["success"] is True
        assert "created_at" in result_dict


class TestSimpleAgent:
    """Test SimpleAgent implementation."""

    def test_simple_agent_initialization(self):
        """Test SimpleAgent initialization."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()
            assert agent.name == "SimpleAgent"
            assert agent.config.model_identifier == "deepseek-chat"

    def test_get_supported_questions(self):
        """Test supported question types."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()
            questions = agent.get_supported_questions()
            assert "general_summary" in questions
            assert "key_contributions" in questions
            assert "methodology_overview" in questions
            assert "custom_question" in questions

    def test_agent_validation(self):
        """Test input validation."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()

            # Valid input
            assert agent.validate_input("test document", "test question") is True

            # Invalid inputs
            assert agent.validate_input(None, "test question") is False
            assert agent.validate_input("test document", None) is False
            assert agent.validate_input("test document", "") is False

    def test_context_preparation(self):
        """Test context preparation for different document types."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()

            # Test with text string
            context = agent.prepare_context("test document content")
            assert context == "test document content"

            # Test with Document mock
            doc_mock = MagicMock()
            doc_mock.text = "document text"
            context = agent.prepare_context(doc_mock)
            assert context == "document text"

            # Test with chunks
            chunk1 = Chunk(content="content1", chunk_type="abstract")
            chunk2 = Chunk(content="content2", chunk_type="introduction")
            chunks = [chunk1, chunk2]
            context = agent.prepare_context(chunks)
            assert "[ABSTRACT]" in context
            assert "[INTRODUCTION]" in context
            assert "content1" in context
            assert "content2" in context

    def test_estimate_tokens(self):
        """Test token estimation."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()
            document = "Test document with multiple words for testing"

            estimated = agent.estimate_tokens(document)
            assert estimated > 0
            assert isinstance(estimated, int)

    def test_is_suitable_for_document(self):
        """Test document suitability checking."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = SimpleAgent()

            # Small document should be suitable
            small_doc = "Small document content"
            is_suitable, reason = agent.is_suitable_for_document(small_doc)
            assert is_suitable is True
            assert "suitable" in reason.lower()

            # Very large document (mocked as many chunks) might not be suitable
            doc_mock = MagicMock()
            doc_mock.chunks = [MagicMock() for _ in range(25)]  # 25 chunks
            is_suitable, reason = agent.is_suitable_for_document(doc_mock)
            assert is_suitable is False
            assert "better suited" in reason or "section-based" in reason


class TestToolCallingAgent:
    """Test ToolCallingAgent implementation."""

    def test_tool_calling_agent_initialization(self):
        """Test ToolCallingAgent initialization."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = ToolCallingAgent()
            assert agent.name == "ToolCallingAgent"
            assert agent.controller.name == "ControllerAgent"

    def test_get_supported_questions(self):
        """Test supported question types."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = ToolCallingAgent()
            questions = agent.get_supported_questions()
            assert "comprehensive_analysis" in questions
            assert "research_questions" in questions
            assert "methodology_analysis" in questions

    def test_is_suitable_for_document(self):
        """Test document suitability checking."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            agent = ToolCallingAgent()

            # Document with multiple sections should be suitable
            doc_mock = MagicMock()
            doc_mock.chunks = [
                Chunk(content="content1", chunk_type="abstract"),
                Chunk(content="content2", chunk_type="introduction"),
                Chunk(content="content3", chunk_type="methods"),
                Chunk(content="content4", chunk_type="results"),
            ]
            is_suitable, reason = agent.is_suitable_for_document(doc_mock)
            assert is_suitable is True
            assert "suitable" in reason.lower()

            # Document without clear sections should not be suitable
            doc_mock.chunks = [
                Chunk(content="content1", chunk_type="unknown"),
                Chunk(content="content2", chunk_type="unknown"),
            ]
            is_suitable, reason = agent.is_suitable_for_document(doc_mock)
            assert is_suitable is False
            assert "better suited" in reason


class TestMultiAgentSystem:
    """Test MultiAgentSystem implementation."""

    def test_multi_agent_system_initialization(self):
        """Test MultiAgentSystem initialization."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            system = MultiAgentSystem()
            assert system.name == "MultiAgentSystem"
            assert system.coordinator.name == "CoordinatorAgent"

    def test_select_agents(self):
        """Test agent selection based on question."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            system = MultiAgentSystem()

            # Question about research questions should select research_question agent
            agents = system._select_agents("What is the main research question?")
            agent_types = [agent.question_type for agent in agents]
            assert "research_question" in agent_types

            # Question about motivation should select motivation agent
            agents = system._select_agents("Why did the authors do this research?")
            agent_types = [agent.question_type for agent in agents]
            assert "motivation" in agent_types

    def test_get_supported_questions(self):
        """Test supported high-level research questions."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            system = MultiAgentSystem()
            questions = system.get_supported_questions()
            assert "What is the Research Question?" in questions
            assert "Why is the author doing this topic?" in questions
            assert "How did the author do the research?" in questions
            assert "What did the author get from the result?" in questions

    def test_is_suitable_for_document(self):
        """Test document suitability checking."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            system = MultiAgentSystem()

            # High-level question should be suitable
            is_suitable, reason = system.is_suitable_for_document(
                "document",
                research_question="What is the main research question?"
            )
            assert is_suitable is True
            assert "well-suited" in reason.lower() or "high-level" in reason.lower()

            # No research question should not be suitable
            is_suitable, reason = system.is_suitable_for_document("document")
            assert is_suitable is False
            assert "research question" in reason.lower()


class TestAgentFactory:
    """Test agent factory functions."""

    def test_create_agent(self):
        """Test agent creation."""
        with patch('sciread.agents.base.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            # Test creating specific agent types
            simple_agent = create_agent("simple")
            assert isinstance(simple_agent, SimpleAgent)

            tool_agent = create_agent("tool_calling")
            assert isinstance(tool_agent, ToolCallingAgent)

            multi_agent = create_agent("multi_agent")
            assert isinstance(multi_agent, MultiAgentSystem)

            # Test invalid agent type
            with pytest.raises(ValueError, match="Unknown agent type"):
                create_agent("invalid_type")

    def test_get_agent_recommendations(self):
        """Test getting agent recommendations."""
        doc_mock = MagicMock()
        doc_mock.chunks = [
            Chunk(content="content1", chunk_type="abstract"),
            Chunk(content="content2", chunk_type="introduction"),
        ]

        recommendations = get_agent_recommendations(doc_mock, "test question")

        assert len(recommendations) == 3  # Should have all three agent types
        assert any(rec["agent_type"] == "SimpleAgent" for rec in recommendations)
        assert any(rec["agent_type"] == "ToolCallingAgent" for rec in recommendations)
        assert any(rec["agent_type"] == "MultiAgentSystem" for rec in recommendations)


class TestAgentOrchestrator:
    """Test AgentOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = AgentOrchestrator()
        assert orchestrator.config.model_identifier == "deepseek-chat"

    def test_get_optimal_agent(self):
        """Test optimal agent selection."""
        orchestrator = AgentOrchestrator()

        # Small document should select SimpleAgent
        agent, reason = orchestrator.get_optimal_agent("small document", "simple question")
        assert isinstance(agent, SimpleAgent)


class TestIntegration:
    """Integration tests for the complete agent system."""

    def test_agent_selection_workflow(self):
        """Test the complete agent selection workflow."""
        # Test with different document types
        simple_doc = "Short document content"
        complex_doc = MagicMock()
        complex_doc.chunks = [
            Chunk(content="Abstract", chunk_type="abstract"),
            Chunk(content="Introduction", chunk_type="introduction"),
            Chunk(content="Methods", chunk_type="methods"),
            Chunk(content="Results", chunk_type="results"),
            Chunk(content="Conclusion", chunk_type="conclusion"),
        ]

        # Get recommendations for simple document
        simple_recs = get_agent_recommendations(simple_doc, "simple question")
        simple_suitable = [rec for rec in simple_recs if rec.get("is_suitable", False)]

        # Get recommendations for complex document
        complex_recs = get_agent_recommendations(complex_doc, "comprehensive analysis")
        complex_suitable = [rec for rec in complex_recs if rec.get("is_suitable", False)]

        # SimpleAgent should be suitable for simple document
        simple_agent_rec = next(rec for rec in simple_recs if rec["agent_type"] == "SimpleAgent")
        assert simple_agent_rec["is_suitable"] is True

        # ToolCallingAgent should be suitable for complex document
        tool_agent_rec = next(rec for rec in complex_recs if rec["agent_type"] == "ToolCallingAgent")
        assert tool_agent_rec["is_suitable"] is True


if __name__ == "__main__":
    pytest.main([__file__])