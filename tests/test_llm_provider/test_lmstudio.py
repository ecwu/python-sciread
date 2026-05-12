"""Tests for LM Studio provider."""

from unittest.mock import MagicMock
from unittest.mock import patch

from sciread.llm_provider.lmstudio import LMStudioProvider


class TestLMStudioProvider:
    """Test cases for LM Studio provider."""

    @patch("sciread.llm_provider.lmstudio.get_config")
    def test_create_model_success(self, mock_config):
        """Test successful model creation with OpenAI-compatible settings."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://localhost:1234/v1"
        mock_config.return_value.get_provider_config.return_value.api_key = "lm_studio"

        with (
            patch("sciread.llm_provider.lmstudio.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.lmstudio.PydanticOpenAIProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            result = LMStudioProvider.create_model("qwen3:4b")

            mock_provider.assert_called_once_with(api_key="lm_studio", base_url="http://localhost:1234/v1")
            mock_model_class.assert_called_once_with(model_name="qwen3:4b", provider=mock_provider_instance)
            assert result == mock_model

    @patch("sciread.llm_provider.lmstudio.get_config")
    def test_create_model_defaults_to_local_settings(self, mock_config):
        """Test fallback base URL and API key."""
        mock_config.return_value.get_provider_config.return_value.base_url = None
        mock_config.return_value.get_provider_config.return_value.api_key = None

        with (
            patch("sciread.llm_provider.lmstudio.OpenAIChatModel"),
            patch("sciread.llm_provider.lmstudio.PydanticOpenAIProvider") as mock_provider,
        ):
            LMStudioProvider.create_model("qwen3:4b", temperature=0.7)

            mock_provider.assert_called_once_with(api_key="lm_studio", base_url="http://localhost:1234/v1")

    def test_get_supported_models(self):
        """Test getting supported model examples."""
        models = LMStudioProvider.get_supported_models()
        assert "qwen3:4b" in models
        assert "llama-3.2-3b-it" in models

    def test_is_model_supported(self):
        """Test checking local model naming patterns."""
        assert LMStudioProvider.is_model_supported("qwen3:4b")
        assert LMStudioProvider.is_model_supported("llama-3.2-3b-it")
        assert LMStudioProvider.is_model_supported("mistral:latest")
        assert not LMStudioProvider.is_model_supported("gpt-4")
        assert not LMStudioProvider.is_model_supported("")
        assert not LMStudioProvider.is_model_supported("   ")
