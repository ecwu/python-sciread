"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sciread.config import LLMProviderConfig
from sciread.config import ScireadConfig


class TestScireadConfig:
    """Test configuration loading and management."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ScireadConfig()

        assert config.default.provider == "deepseek"
        assert config.default.model == "deepseek-chat"
        assert "deepseek" in config.llm_providers
        assert "volcengine" in config.llm_providers
        assert "ollama" in config.llm_providers

    def test_load_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[llm_providers.custom]
api_key = "custom-key"
default_model = "custom-model"
base_url = "https://custom.api.com"

[llm_providers.default]
provider = "custom"
model = "custom-model"
"""
        config_file.write_text(config_content)

        config = ScireadConfig.load_from_file(config_file)

        assert config.default.provider == "custom"
        assert config.default.model == "custom-model"
        assert "custom" in config.llm_providers
        assert config.llm_providers["custom"].api_key == "custom-key"

    def test_environment_variable_substitution(self, tmp_path):
        """Test environment variable substitution in config."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[llm_providers.test]
api_key = "${TEST_API_KEY}"
default_model = "test-model"
"""
        config_file.write_text(config_content)

        with patch.dict(os.environ, {"TEST_API_KEY": "env-substituted-key"}):
            config = ScireadConfig.load_from_file(config_file)
            assert config.llm_providers["test"].api_key == "env-substituted-key"

    def test_get_provider_config(self):
        """Test getting provider configuration."""
        config = ScireadConfig()

        deepseek_config = config.get_provider_config("deepseek")
        assert isinstance(deepseek_config, LLMProviderConfig)
        assert deepseek_config.default_model == "deepseek-chat"

    def test_get_provider_config_invalid(self):
        """Test getting configuration for invalid provider."""
        config = ScireadConfig()

        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_provider_config("invalid")

    def test_get_api_key_from_config(self, tmp_path):
        """Test getting API key from configuration."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[llm_providers.test]
api_key = "config-key"
default_model = "test-model"
"""
        config_file.write_text(config_content)

        config = ScireadConfig.load_from_file(config_file)
        api_key = config.get_api_key("test")
        assert api_key == "config-key"

    def test_get_api_key_from_environment(self):
        """Test getting API key from environment variable."""
        config = ScireadConfig()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            api_key = config.get_api_key("deepseek")
            assert api_key == "env-key"

    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        config = ScireadConfig()

        # Ensure no environment variable is set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                config.get_api_key("deepseek")

    def test_config_file_priority(self, tmp_path):
        """Test that config files are found in correct priority order."""
        # Create multiple config files
        config_files = [
            tmp_path / "config" / "sciread.toml",
            tmp_path / "sciread.toml",
        ]

        for config_file in config_files:
            if config_file.parent != tmp_path:
                config_file.parent.mkdir(parents=True, exist_ok=True)

            config_content = f"""
[llm_providers.default]
provider = "priority-test"
model = "from-{config_file.stem}"
"""
            config_file.write_text(config_content)

        # Change to tmp_path to test relative paths
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            config = ScireadConfig.load_from_file()

            # Should pick the first one found (config/sciread.toml)
            assert config.default.provider == "priority-test"
        finally:
            os.chdir(original_cwd)
