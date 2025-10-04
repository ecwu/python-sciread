"""Integration tests for LLM provider module."""

import pytest
from unittest.mock import patch, MagicMock

from sciread.llm_provider import get_model, ModelFactory
from sciread.config import ScireadConfig, LLMProviderConfig, DefaultConfig


class TestIntegration:
    """Integration tests that verify end-to-end functionality."""

    def test_deepseek_end_to_end(self):
        """Test DeepSeek model creation end-to-end with simplified mocking."""
        with patch('sciread.llm_provider.deepseek.get_config') as mock_config, \
             patch('sciread.llm_provider.deepseek.PydanticDeepSeekProvider') as mock_provider, \
             patch('sciread.llm_provider.deepseek.OpenAIChatModel') as mock_openai:

            # Mock config
            mock_config.return_value.get_provider_config.return_value.base_url = "https://api.deepseek.com"
            mock_config.return_value.get_api_key.return_value = "test-api-key"

            # Mock provider and model
            mock_model = MagicMock()
            mock_openai.return_value = mock_model
            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            # Test actual model creation
            model = get_model("deepseek/deepseek-chat")

            # Verify the correct calls were made
            mock_provider.assert_called_once_with(api_key="test-api-key")
            mock_openai.assert_called_once_with(
                model_name="deepseek-chat",
                provider=mock_provider_instance
            )
            assert model == mock_model

    def test_model_parsing_edge_cases(self):
        """Test model parsing with various edge cases."""
        # Test case sensitivity
        provider, model = ModelFactory.parse_model_identifier("DeepSeek-Chat")
        assert provider == "deepseek"
        assert model == "DeepSeek-Chat"

        # Test model names with special characters
        provider, model = ModelFactory.parse_model_identifier("ollama/model:latest")
        assert provider == "ollama"
        assert model == "model:latest"

    def test_provider_priority_order(self):
        """Test that providers are checked in the correct order."""
        # This should go to deepseek, not ollama (even though ollama supports any model)
        provider, model = ModelFactory.parse_model_identifier("deepseek-chat")
        assert provider == "deepseek"
        assert model == "deepseek-chat"

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'env-test-key'})
    def test_environment_variable_fallback(self, tmp_path):
        """Test that environment variables are used as fallback."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[llm_providers.deepseek]
api_key = ""  # Empty in config
default_model = "deepseek-chat"
"""
        config_file.write_text(config_content)

        with patch('sciread.llm_provider.factory.get_config') as mock_get_config:
            config = ScireadConfig.load_from_file(config_file)
            mock_get_config.return_value = config

            # Should not raise an error since env var is available
            api_key = config.get_api_key('deepseek')
            assert api_key == 'env-test-key'