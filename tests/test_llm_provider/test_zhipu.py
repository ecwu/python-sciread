"""Tests for Zhipu provider."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sciread.llm_provider.zhipu import ZhipuProvider


class TestZhipuProvider:
    """Test cases for Zhipu provider."""

    @patch("sciread.llm_provider.zhipu.get_config")
    def test_create_model_success(self, mock_config):
        """Test successful model creation."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://open.bigmodel.cn/api/anthropic"
        mock_config.return_value.get_api_key.return_value = "test-api-key"

        with patch("sciread.llm_provider.zhipu.AnthropicModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            result = ZhipuProvider.create_model("glm-4.6")

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args.kwargs["model_name"] == "glm-4.6"  # model_name
            assert result == mock_model

    @patch("sciread.llm_provider.zhipu.get_config")
    def test_create_model_with_custom_base_url(self, mock_config):
        """Test model creation with custom base URL."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://custom.zhipu.com"
        mock_config.return_value.get_api_key.return_value = "test-api-key"

        with patch("sciread.llm_provider.zhipu.AnthropicModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            ZhipuProvider.create_model("glm-4.6", temperature=0.7)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            # Note: The base_url is passed to AnthropicProvider, not checked here
            # since it's set during provider initialization
            assert call_args.kwargs["model_name"] == "glm-4.6"

    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        with pytest.raises(ValueError, match="Unsupported Zhipu model"):
            ZhipuProvider.create_model("unsupported-model")

    @patch("sciread.llm_provider.zhipu.get_config")
    def test_create_model_missing_api_key(self, mock_config):
        """Test creating model when API key is missing."""
        mock_config.return_value.get_api_key.side_effect = ValueError("API key not found")

        with pytest.raises(ValueError, match="API key not found"):
            ZhipuProvider.create_model("glm-4.6")

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = ZhipuProvider.get_supported_models()
        assert "glm-4.6" in models
        assert "glm-4.5" in models
        assert models["glm-4.6"] == "GLM-4.6 Model"

    def test_is_model_supported(self):
        """Test checking if model is supported."""
        assert ZhipuProvider.is_model_supported("glm-4.6")
        assert ZhipuProvider.is_model_supported("glm-4.5")
        assert not ZhipuProvider.is_model_supported("unsupported-model")
        assert not ZhipuProvider.is_model_supported("")
