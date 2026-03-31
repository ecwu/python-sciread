"""Tests for Volcengine provider."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sciread.llm_provider.volcengine import VolcengineProvider


class TestVolcengineProvider:
    """Test cases for Volcengine provider."""

    @patch("sciread.llm_provider.volcengine.get_config")
    def test_create_model_success(self, mock_config):
        """Test successful model creation."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        mock_config.return_value.get_provider_config.return_value.api_key = "test-api-key"

        with (
            patch("sciread.llm_provider.volcengine.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.volcengine.PydanticOpenAIProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            result = VolcengineProvider.create_model("glm-4.7")

            mock_provider.assert_called_once_with(
                api_key="test-api-key",
                base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            )
            mock_model_class.assert_called_once_with(model_name="glm-4.7", provider=mock_provider_instance)
            assert result == mock_model

    @patch("sciread.llm_provider.volcengine.get_config")
    def test_create_model_with_custom_base_url(self, mock_config):
        """Test model creation with custom base URL."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://custom.volcengine.com/v1"
        mock_config.return_value.get_provider_config.return_value.api_key = "test-api-key"

        with (
            patch("sciread.llm_provider.volcengine.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.volcengine.PydanticOpenAIProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_provider.return_value = MagicMock()

            VolcengineProvider.create_model("doubao-seed-2.0-code", temperature=0.7)

            mock_provider.assert_called_once_with(api_key="test-api-key", base_url="https://custom.volcengine.com/v1")
            mock_model_class.assert_called_once_with(
                model_name="doubao-seed-2.0-code",
                provider=mock_provider.return_value,
                temperature=0.7,
            )

    def test_create_model_unsupported(self):
        """Test creating unsupported model."""
        with pytest.raises(ValueError, match="Unsupported Volcengine model"):
            VolcengineProvider.create_model("unsupported-model")

    @patch("sciread.llm_provider.volcengine.get_config")
    def test_create_model_missing_api_key(self, mock_config):
        """Test creating model when API key is missing."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        mock_config.return_value.get_provider_config.return_value.api_key = None

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="VOLCES_API"):
                VolcengineProvider.create_model("glm-4.7")

    @patch("sciread.llm_provider.volcengine.get_config")
    def test_create_model_api_key_from_env(self, mock_config):
        """Test reading API key from VOLCES_API when config key is empty."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        mock_config.return_value.get_provider_config.return_value.api_key = None

        with (
            patch.dict("os.environ", {"VOLCES_API": "env-api-key"}),
            patch("sciread.llm_provider.volcengine.OpenAIChatModel") as mock_model_class,
            patch("sciread.llm_provider.volcengine.PydanticOpenAIProvider") as mock_provider,
        ):
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            VolcengineProvider.create_model("deepseek-v3.2")

            mock_provider.assert_called_once_with(
                api_key="env-api-key",
                base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            )

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = VolcengineProvider.get_supported_models()
        assert "doubao-seed-2.0-code" in models
        assert "glm-4.7" in models
        assert "kimi-k2.5" in models
        assert models["glm-4.7"] == "GLM-4.7 Model"

    def test_is_model_supported(self):
        """Test checking if model is supported."""
        assert VolcengineProvider.is_model_supported("doubao-seed-2.0-code")
        assert VolcengineProvider.is_model_supported("glm-4.7")
        assert not VolcengineProvider.is_model_supported("unsupported-model")
        assert not VolcengineProvider.is_model_supported("")
