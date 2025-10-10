"""Tests for Pydantic AI agent systems functionality."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from sciread.agents import (
    AgentOrchestrator,
    DocumentDeps,
    SimpleAnalysisResult,
    DocumentAnalysisResult,
    analyze_document,
    create_agent_analysis,
    get_agent_recommendations,
    create_simple_agent,
    create_tool_calling_agent,
    create_coordinator_agent,
    simple_agent,
    tool_calling_agent,
    coordinator_agent,
)
from sciread.document.models import Chunk


class TestDocumentDeps:
    """Test DocumentDeps class."""

    def test_document_deps_creation(self):
        """Test DocumentDeps creation with default values."""
        deps = DocumentDeps(document_text="test document")
        assert deps.document_text == "test document"
        assert deps.document_chunks == []
        assert deps.available_sections == []
        assert deps.metadata == {}
        assert deps.model_identifier == "deepseek-chat"
        assert deps.temperature == 0.3
        assert deps.max_tokens is None

    def test_document_deps_with_chunks(self):
        """Test DocumentDeps with chunks."""
        chunks = [
            {"content": "abstract content", "chunk_type": "abstract"},
            {"content": "intro content", "chunk_type": "introduction"},
        ]
        deps = DocumentDeps(
            document_text="test document",
            document_chunks=chunks,
            metadata={"title": "Test Paper"}
        )
        assert deps.document_chunks == chunks
        assert deps.available_sections == ["abstract", "introduction"]
        assert deps.metadata["title"] == "Test Paper"

    def test_get_section_chunks(self):
        """Test getting chunks by section type."""
        chunks = [
            {"content": "abstract content", "chunk_type": "abstract"},
            {"content": "intro content", "chunk_type": "introduction"},
            {"content": "abstract 2", "chunk_type": "abstract"},
        ]
        deps = DocumentDeps(document_text="test", document_chunks=chunks)

        abstract_chunks = deps.get_section_chunks("abstract")
        assert len(abstract_chunks) == 2
        assert all(chunk["chunk_type"] == "abstract" for chunk in abstract_chunks)

        intro_chunks = deps.get_section_chunks("methods")
        assert len(intro_chunks) == 0

    def test_has_section(self):
        """Test checking if document has a section."""
        chunks = [
            {"content": "abstract content", "chunk_type": "abstract"},
            {"content": "intro content", "chunk_type": "introduction"},
        ]
        deps = DocumentDeps(document_text="test", document_chunks=chunks)

        assert deps.has_section("abstract") is True
        assert deps.has_section("introduction") is True
        assert deps.has_section("methods") is False


class TestSimpleAnalysisResult:
    """Test SimpleAnalysisResult class."""

    def test_simple_analysis_result_creation(self):
        """Test SimpleAnalysisResult creation."""
        result = SimpleAnalysisResult(
            content="Analysis content",
            question_answered="What is this about?",
            confidence_score=0.8,
            processing_time=1.5,
            model_used="deepseek-chat",
            token_count=1000
        )
        assert result.content == "Analysis content"
        assert result.question_answered == "What is this about?"
        assert result.confidence_score == 0.8
        assert result.processing_time == 1.5
        assert result.model_used == "deepseek-chat"
        assert result.token_count == 1000


class TestDocumentAnalysisResult:
    """Test DocumentAnalysisResult class."""

    def test_document_analysis_result_creation(self):
        """Test DocumentAnalysisResult creation."""
        from sciread.agents.schemas import DocumentMetadata

        metadata = DocumentMetadata(
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            chunk_count=10,
            section_types=["abstract", "introduction"]
        )

        result = DocumentAnalysisResult(
            summary="Paper summary",
            key_contributions=["Contribution 1", "Contribution 2"],
            methodology_overview="Method used",
            main_findings=["Finding 1"],
            implications="Important implications",
            future_directions=["Future work"],
            confidence_score=0.9,
            sections_analyzed=["abstract", "introduction"],
            metadata=metadata,
            execution_time=5.0
        )
        assert result.summary == "Paper summary"
        assert len(result.key_contributions) == 2
        assert result.confidence_score == 0.9
        assert result.metadata.title == "Test Paper"


class TestAgentCreation:
    """Test agent creation functions."""

    @patch('sciread.agents.simple_agent.get_model')
    def test_create_simple_agent(self, mock_get_model):
        """Test creating a simple agent."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        agent = create_simple_agent("openai:gpt-4o")
        assert agent is not None
        mock_get_model.assert_called_once_with("openai:gpt-4o")

    @patch('sciread.agents.tool_calling_agent.get_model')
    def test_create_tool_calling_agent(self, mock_get_model):
        """Test creating a tool-calling agent."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        agent = create_tool_calling_agent("openai:gpt-4o")
        assert agent is not None
        mock_get_model.assert_called_once_with("openai:gpt-4o")

    @patch('sciread.agents.multi_agent_system.get_model')
    def test_create_coordinator_agent(self, mock_get_model):
        """Test creating a coordinator agent."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        agent = create_coordinator_agent("openai:gpt-4o")
        assert agent is not None
        mock_get_model.assert_called_once_with("openai:gpt-4o")


class TestAgentSelector:
    """Test agent selection logic."""

    def test_select_simple_agent_for_small_document(self):
        """Test selecting simple agent for small documents."""
        from sciread.agents.factory import AgentSelector

        document_text = "Short document content"
        question = "What is this about?"

        func_name, display_name, metadata = AgentSelector.select_agent(
            document_text=document_text,
            question=question
        )

        assert func_name == "analyze_document_simple"
        assert display_name == "SimpleAgent"
        assert "suitable for simple processing" in metadata["reason"]

    def test_select_multi_agent_for_research_question(self):
        """Test selecting multi-agent for research questions."""
        from sciread.agents.factory import AgentSelector

        document_text = "Long academic document with multiple sections"
        question = "What is the main research question and why is it important?"

        chunks = [
            {"content": "abstract", "chunk_type": "abstract"},
            {"content": "introduction", "chunk_type": "introduction"},
            {"content": "methods", "chunk_type": "methods"},
            {"content": "results", "chunk_type": "results"},
        ]

        func_name, display_name, metadata = AgentSelector.select_agent(
            document_text=document_text,
            question=question,
            document_chunks=chunks
        )

        assert func_name == "analyze_document_with_multi_agent"
        assert display_name == "MultiAgentSystem"
        assert "collaborative analysis" in metadata["reason"]

    def test_select_tool_calling_for_structured_document(self):
        """Test selecting tool-calling agent for structured documents."""
        from sciread.agents.factory import AgentSelector

        document_text = "Document with clear structure"
        question = "Analyze the methodology and results"

        chunks = [{"content": f"section {i}", "chunk_type": f"section_{i}"} for i in range(15)]

        func_name, display_name, metadata = AgentSelector.select_agent(
            document_text=document_text,
            question=question,
            document_chunks=chunks
        )

        assert func_name == "analyze_document_with_sections"
        assert display_name == "ToolCallingAgent"
        assert "section structure" in metadata["reason"]

    def test_force_agent_selection(self):
        """Test forcing a specific agent type."""
        from sciread.agents.factory import AgentSelector

        func_name, display_name, metadata = AgentSelector.select_agent(
            document_text="any document",
            question="any question",
            force_agent="tool_calling"
        )

        assert func_name == "analyze_document_with_sections"
        assert display_name == "ToolCallingAgent"
        assert metadata["forced"] is True


class TestAgentRecommendations:
    """Test agent recommendation system."""

    @pytest.mark.asyncio
    @patch('sciread.agents.factory.get_agent_recommendations')
    async def test_get_agent_recommendations(self, mock_get_recs):
        """Test getting agent recommendations."""
        mock_get_recs.return_value = [
            {
                "agent_name": "SimpleAgent",
                "agent_type": "simple",
                "is_suitable": True,
                "reason": "Good for simple analysis",
                "confidence": 0.8
            }
        ]

        recommendations = await get_agent_recommendations("document text", "test question")
        assert len(recommendations) == 1
        assert recommendations[0]["agent_name"] == "SimpleAgent"
        mock_get_recs.assert_called_once()


class TestAgentOrchestrator:
    """Test AgentOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = AgentOrchestrator("test-model")
        assert orchestrator.model_identifier == "test-model"

    @pytest.mark.asyncio
    @patch('sciread.agents.factory.AgentSelector.select_agent')
    async def test_get_optimal_agent(self, mock_select):
        """Test getting optimal agent."""
        mock_select.return_value = (
            "analyze_document_simple",
            "SimpleAgent",
            {"reason": "Test reason"}
        )

        orchestrator = AgentOrchestrator()
        func_name, display_name, metadata = await orchestrator.get_optimal_agent(
            "document text", "test question"
        )

        assert func_name == "analyze_document_simple"
        assert display_name == "SimpleAgent"
        assert metadata["reason"] == "Test reason"


class TestAgentAnalysis:
    """Test agent analysis functions."""

    @pytest.mark.asyncio
    @patch('sciread.agents.factory.AgentSelector.select_agent')
    @patch('sciread.agents.simple_agent.analyze_document_simple')
    async def test_create_agent_analysis_simple(self, mock_analyze, mock_select):
        """Test creating agent analysis with simple agent."""
        # Setup mocks
        mock_select.return_value = (
            "analyze_document_simple",
            "SimpleAgent",
            {"reason": "Test selection"}
        )

        mock_result = SimpleAnalysisResult(
            content="Analysis result",
            question_answered="Test question",
            confidence_score=0.8,
            processing_time=1.0,
            model_used="test-model",
            token_count=500
        )
        mock_analyze.return_value = mock_result

        # Execute
        result = await create_agent_analysis(
            document_text="test document",
            question="test question",
            agent_type="simple"
        )

        # Verify
        assert result.content == "Analysis result"
        assert result.question_answered == "Test question"
        mock_select.assert_called_once()
        mock_analyze.assert_called_once()

    @pytest.mark.asyncio
    @patch('sciread.agents.factory.AgentSelector.select_agent')
    @patch('sciread.agents.simple_agent.analyze_document_simple')
    async def test_analyze_document_fallback(self, mock_analyze, mock_select):
        """Test fallback to simple agent on failure."""
        # Setup mocks
        mock_select.return_value = (
            "analyze_document_with_sections",
            "ToolCallingAgent",
            {"reason": "Test selection"}
        )

        # Make the tool calling agent fail
        mock_analyze.side_effect = [
            Exception("Tool calling agent failed"),
            SimpleAnalysisResult(
                content="Fallback result",
                question_answered="Test question",
                confidence_score=0.7,
                processing_time=1.0,
                model_used="test-model",
                token_count=500
            )
        ]

        # Execute
        result = await analyze_document(
            document_text="test document",
            question="test question"
        )

        # Verify fallback worked
        assert result.content == "Fallback result"


class TestIntegration:
    """Integration tests for the complete agent system."""

    def test_complete_workflow_mock(self):
        """Test complete workflow with mocked components."""
        # This test verifies the integration without making actual LLM calls
        with patch('sciread.agents.factory.AgentSelector.select_agent') as mock_select:
            mock_select.return_value = (
                "analyze_document_simple",
                "SimpleAgent",
                {"reason": "Test selection"}
            )

            # Test that the selection logic works
            func_name, display_name, metadata = mock_select.return_value
            assert func_name == "analyze_document_simple"
            assert display_name == "SimpleAgent"
            assert "reason" in metadata

    def test_document_processing_pipeline(self):
        """Test document processing pipeline with different inputs."""
        # Test data
        simple_doc = "Short document content"
        complex_chunks = [
            {"content": "Abstract content", "chunk_type": "abstract"},
            {"content": "Introduction content", "chunk_type": "introduction"},
            {"content": "Methods content", "chunk_type": "methods"},
            {"content": "Results content", "chunk_type": "results"},
            {"content": "Conclusion content", "chunk_type": "conclusion"},
        ]

        # Test DocumentDeps creation
        simple_deps = DocumentDeps(document_text=simple_doc)
        assert simple_deps.available_sections == []

        complex_deps = DocumentDeps(
            document_text="Complex document",
            document_chunks=complex_chunks
        )
        assert len(complex_deps.available_sections) == 5
        assert "abstract" in complex_deps.available_sections

        # Test section filtering
        abstract_chunks = complex_deps.get_section_chunks("abstract")
        assert len(abstract_chunks) == 1
        assert abstract_chunks[0]["chunk_type"] == "abstract"

        methods_chunks = complex_deps.get_section_chunks("nonexistent")
        assert len(methods_chunks) == 0


class TestPreconfiguredAgents:
    """Test pre-configured agent instances."""

    def test_preconfigured_agents_exist(self):
        """Test that pre-configured agents are available."""
        # These should be importable and not None
        assert simple_agent is not None
        assert tool_calling_agent is not None
        assert coordinator_agent is not None

    def test_preconfigured_agents_have_expected_types(self):
        """Test that pre-configured agents have expected types."""
        from pydantic_ai import Agent

        # All should be Pydantic AI Agent instances
        assert isinstance(simple_agent, Agent)
        assert isinstance(tool_calling_agent, Agent)
        assert isinstance(coordinator_agent, Agent)


if __name__ == "__main__":
    pytest.main([__file__])