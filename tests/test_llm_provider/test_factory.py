"""Tests for the model factory."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from sciread.llm_provider.deepseek import DeepSeekProvider
from sciread.llm_provider.factory import InvalidModelIdentifierError
from sciread.llm_provider.factory import ModelFactory
from sciread.llm_provider.factory import UnsupportedModelError
from sciread.llm_provider.factory import get_model
from sciread.llm_provider.ollama import OllamaProvider
from sciread.llm_provider.zhipu import ZhipuProvider


class TestModelFactory:
    """Test cases for ModelFactory class."""

    def test_parse_model_identifier_with_provider(self):
        """Test parsing model identifier with explicit provider."""
        provider, model = ModelFactory.parse_model_identifier("deepseek/deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    def test_parse_model_identifier_without_provider(self):
        """Test parsing model identifier without explicit provider."""
        with patch("sciread.llm_provider.factory.get_config") as mock_config:
            mock_config.return_value.default.provider = "deepseek"
            mock_config.return_value.default.model = "deepseek-chat"

            provider, model = ModelFactory.parse_model_identifier("custom-model")
            assert provider == "deepseek"
            assert model == "custom-model"

    def test_parse_model_identifier_known_model(self):
        """Test parsing model identifier with known model name."""
        provider, model = ModelFactory.parse_model_identifier("deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    def test_parse_model_identifier_empty(self):
        """Test parsing empty model identifier."""
        with pytest.raises(InvalidModelIdentifierError):
            ModelFactory.parse_model_identifier("")

    def test_parse_model_identifier_invalid_format(self):
        """Test parsing model identifier with invalid format."""
        with pytest.raises(InvalidModelIdentifierError):
            ModelFactory.parse_model_identifier("provider/")

        with pytest.raises(InvalidModelIdentifierError):
            ModelFactory.parse_model_identifier("/model")

    def test_get_provider_class_valid(self):
        """Test getting provider class for valid provider."""
        provider_class = ModelFactory.get_provider_class("deepseek")
        assert provider_class == DeepSeekProvider

    def test_get_provider_class_invalid(self):
        """Test getting provider class for invalid provider."""
        with pytest.raises(UnsupportedModelError):
            ModelFactory.get_provider_class("invalid")

    @patch("sciread.llm_provider.factory.get_config")
    def test_create_model_deepseek(self, mock_config):
        """Test creating DeepSeek model."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://api.deepseek.com"
        mock_config.return_value.get_api_key.return_value = "test-key"

        with patch.object(DeepSeekProvider, "create_model") as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            result = ModelFactory.create_model("deepseek/deepseek-chat")

            mock_create.assert_called_once_with("deepseek-chat")
            assert result == mock_model

    @patch("sciread.llm_provider.factory.get_config")
    def test_create_model_zhipu(self, mock_config):
        """Test creating Zhipu model."""
        mock_config.return_value.get_provider_config.return_value.base_url = "https://open.bigmodel.cn/api/anthropic"
        mock_config.return_value.get_api_key.return_value = "test-key"

        with patch.object(ZhipuProvider, "create_model") as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            result = ModelFactory.create_model("zhipu/glm-4.6")

            mock_create.assert_called_once_with("glm-4.6")
            assert result == mock_model

    @patch("sciread.llm_provider.factory.get_config")
    def test_create_model_ollama(self, mock_config):
        """Test creating Ollama model."""
        mock_config.return_value.get_provider_config.return_value.base_url = "http://localhost:11434/v1"

        with patch.object(OllamaProvider, "create_model") as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            result = ModelFactory.create_model("ollama/qwen3:4b")

            mock_create.assert_called_once_with("qwen3:4b")
            assert result == mock_model

    def test_create_model_unsupported_provider(self):
        """Test creating model with unsupported provider."""
        with pytest.raises(UnsupportedModelError):
            ModelFactory.create_model("invalid/model")

    def test_create_model_unsupported_model(self):
        """Test creating model with unsupported model."""
        with patch("sciread.llm_provider.factory.get_config"):
            with patch.object(DeepSeekProvider, "is_model_supported", return_value=False):
                with pytest.raises(UnsupportedModelError):
                    ModelFactory.create_model("deepseek/unsupported-model")

    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = ModelFactory.get_supported_providers()
        assert "deepseek" in providers
        assert "zhipu" in providers
        assert "ollama" in providers
        assert "deepseek-chat" in providers["deepseek"]
        assert "glm-4.6" in providers["zhipu"]

    def test_list_all_supported_models(self):
        """Test listing all supported models."""
        models = ModelFactory.list_all_supported_models()
        assert "deepseek/deepseek-chat" in models
        assert "zhipu/glm-4.6" in models
        assert "ollama/qwen3:4b" in models


class TestGetModel:
    """Test cases for get_model convenience function."""

    @patch("sciread.llm_provider.factory.ModelFactory.create_model")
    def test_get_model(self, mock_create):
        """Test get_model convenience function."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        result = get_model("deepseek/deepseek-chat")

        mock_create.assert_called_once_with("deepseek/deepseek-chat")
        assert result == mock_model

    @patch("sciread.llm_provider.factory.ModelFactory.create_model")
    def test_get_model_with_kwargs(self, mock_create):
        """Test get_model with additional kwargs."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        result = get_model("deepseek/deepseek-chat", temperature=0.7)

        mock_create.assert_called_once_with("deepseek/deepseek-chat", temperature=0.7)
        assert result == mock_model
