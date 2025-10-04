"""Tests for Ollama provider."""

import pytest
from unittest.mock import patch, MagicMock

from sciread.llm_provider.ollama import OllamaProvider


class TestOllamaProvider:
    """Test cases for Ollama provider."""

    @patch('sciread.llm_provider.ollama.get_config')
    def test_create_model_success(self, mock_config):
        """Test successful model creation."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://localhost:11434/v1"

        with patch('sciread.llm_provider.ollama.OpenAIChatModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            result = OllamaProvider.create_model("qwen3:4b")

            mock_model_class.assert_called_once_with(
                model_name="qwen3:4b",
                provider=mock_model_class.return_value.provider
            )
            assert result == mock_model

    @patch('sciread.llm_provider.ollama.get_config')
    def test_create_model_with_custom_base_url(self, mock_config):
        """Test model creation with custom base URL."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://custom-ollama:11434/v1"

        with patch('sciread.llm_provider.ollama.OpenAIChatModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            OllamaProvider.create_model("qwen3:4b", temperature=0.7)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args.kwargs['provider'].base_url == "http://custom-ollama:11434/v1"

    @patch('sciread.llm_provider.ollama.get_config')
    def test_create_model_default_base_url(self, mock_config):
        """Test model creation with default base URL when not configured."""
        mock_config.return_value.get_provider_config.return_value.base_url = None

        with patch('sciread.llm_provider.ollama.OpenAIChatModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            OllamaProvider.create_model("qwen3:4b")

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args.kwargs['provider'].base_url == "http://localhost:11434/v1"

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = OllamaProvider.get_supported_models()
        assert "qwen3:4b" in models
        assert "llama3:8b" in models
        assert "mistral:7b" in models
        assert models["qwen3:4b"] == "Qwen3 4B parameter model"

    def test_is_model_supported(self):
        """Test checking if model is supported."""
        # Ollama supports any non-empty model name
        assert OllamaProvider.is_model_supported("qwen3:4b")
        assert OllamaProvider.is_model_supported("custom-model")
        assert OllamaProvider.is_model_supported("any-model-name")
        assert not OllamaProvider.is_model_supported("")
        assert not OllamaProvider.is_model_supported("   ")
        assert not OllamaProvider.is_model_supported(None)