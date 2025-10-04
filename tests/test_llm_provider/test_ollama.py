"""Simplified tests for Ollama provider based on official documentation."""

from unittest.mock import MagicMock
from unittest.mock import patch

from sciread.llm_provider.ollama import OllamaProvider


class TestOllamaProvider:
    """Test cases for Ollama provider."""

    @patch("sciread.llm_provider.ollama.get_config")
    def test_create_model_success(self, mock_config):
        """Test successful model creation following official API pattern."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://localhost:11434/v1"

        with (
            patch("sciread.llm_provider.ollama.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.ollama.PydanticOllamaProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            result = OllamaProvider.create_model("qwen3:4b")

            # Verify the provider was created with base_url
            mock_provider.assert_called_once_with(base_url="http://localhost:11434/v1")

            # Verify the model was created following the official pattern
            mock_model_class.assert_called_once_with(model_name="qwen3:4b", provider=mock_provider.return_value)
            assert result == mock_model

    @patch("sciread.llm_provider.ollama.get_config")
    def test_create_model_with_kwargs(self, mock_config):
        """Test model creation with additional parameters."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://localhost:11434/v1"

        with (
            patch("sciread.llm_provider.ollama.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.ollama.PydanticOllamaProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            OllamaProvider.create_model("qwen3:4b", temperature=0.7)

            # Verify kwargs are passed through correctly
            mock_model_class.assert_called_once_with(model_name="qwen3:4b", provider=mock_provider.return_value, temperature=0.7)

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = OllamaProvider.get_supported_models()
        assert "qwen3:4b" in models
        assert "llama3:8b" in models
        assert "mistral:7b" in models
        assert models["qwen3:4b"] == "Qwen3 4B parameter model"

    def test_is_model_supported(self):
        """Test checking if model is supported using pattern matching."""
        # Models with typical Ollama patterns should be supported
        assert OllamaProvider.is_model_supported("qwen3:4b")  # Has colon
        assert OllamaProvider.is_model_supported("llama2:7b")  # Has colon
        assert OllamaProvider.is_model_supported("mistral:latest")  # Has colon
        assert OllamaProvider.is_model_supported("llama3-8b")  # Has llama
        assert OllamaProvider.is_model_supported("mistral-7b")  # Has mistral

        # Generic models without Ollama patterns should not be supported
        assert not OllamaProvider.is_model_supported("gpt-4")  # Generic model
        assert not OllamaProvider.is_model_supported("")  # Empty
        assert not OllamaProvider.is_model_supported("   ")  # Whitespace only
