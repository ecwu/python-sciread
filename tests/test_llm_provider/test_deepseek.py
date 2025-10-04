"""Tests for DeepSeek provider."""

import pytest
from unittest.mock import patch, MagicMock

from sciread.llm_provider.deepseek import DeepSeekProvider


class TestDeepSeekProvider:
    """Test cases for DeepSeek provider."""

    @patch('sciread.llm_provider.deepseek.get_config')
    def test_create_model_success(self, mock_config):
        """Test successful model creation."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://api.deepseek.com"
        mock_config.return_value.get_api_key.return_value = "test-api-key"

        with patch('sciread.llm_provider.deepseek.OpenAIChatModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            result = DeepSeekProvider.create_model("deepseek-chat")

            mock_model_class.assert_called_once_with(
                model_name="deepseek-chat",
                provider=mock_model_class.return_value.provider,
                base_url="https://api.deepseek.com"
            )
            assert result == mock_model

    @patch('sciread.llm_provider.deepseek.get_config')
    def test_create_model_with_custom_base_url(self, mock_config):
        """Test model creation with custom base URL."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://custom.deepseek.com"
        mock_config.return_value.get_api_key.return_value = "test-api-key"

        with patch('sciread.llm_provider.deepseek.OpenAIChatModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            DeepSeekProvider.create_model("deepseek-chat", temperature=0.7)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args.kwargs['base_url'] == "https://custom.deepseek.com"

    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
            DeepSeekProvider.create_model("unsupported-model")

    @patch('sciread.llm_provider.deepseek.get_config')
    def test_create_model_missing_api_key(self, mock_config):
        """Test creating model when API key is missing."""
        mock_config.return_value.get_api_key.side_effect = ValueError("API key not found")

        with pytest.raises(ValueError, match="API key not found"):
            DeepSeekProvider.create_model("deepseek-chat")

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = DeepSeekProvider.get_supported_models()
        assert "deepseek-chat" in models
        assert "deepseek-reasoner" in models
        assert models["deepseek-chat"] == "DeepSeek Chat Model"

    def test_is_model_supported(self):
        """Test checking if model is supported."""
        assert DeepSeekProvider.is_model_supported("deepseek-chat")
        assert DeepSeekProvider.is_model_supported("deepseek-reasoner")
        assert not DeepSeekProvider.is_model_supported("unsupported-model")
        assert not DeepSeekProvider.is_model_supported("")