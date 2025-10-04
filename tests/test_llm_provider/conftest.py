"""Test configuration and fixtures for llm_provider tests."""

import os
from pathlib import Path
from unittest.mock import patch
import pytest

from sciread.config import ScireadConfig, LLMProviderConfig, DefaultConfig


@pytest.fixture
def test_config_dir(tmp_path):
    """Create a temporary config directory for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def test_config_file(test_config_dir):
    """Create a test configuration file."""
    config_file = test_config_dir / "sciread.toml"
    config_content = """
[llm_providers.deepseek]
api_key = "test-deepseek-key"
default_model = "deepseek-chat"
base_url = "https://api.deepseek.com"

[llm_providers.zhipu]
api_key = "test-zhipu-key"
default_model = "glm-4.6"
base_url = "https://open.bigmodel.cn/api/anthropic"

[llm_providers.ollama]
base_url = "http://localhost:11434/v1"
default_model = "qwen3:4b"

[llm_providers.default]
provider = "deepseek"
model = "deepseek-chat"
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_config(test_config_file):
    """Mock configuration for testing."""
    with patch('sciread.llm_provider.deepseek.get_config') as mock_deepseek, \
         patch('sciread.llm_provider.zhipu.get_config') as mock_zhipu, \
         patch('sciread.llm_provider.ollama.get_config') as mock_ollama, \
         patch('sciread.llm_provider.factory.get_config') as mock_factory:

        config = ScireadConfig.load_from_file(test_config_file)
        mock_deepseek.return_value = config
        mock_zhipu.return_value = config
        mock_ollama.return_value = config
        mock_factory.return_value = config

        yield config


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'DEEPSEEK_API_KEY': 'test-deepseek-key-from-env',
        'ZHIPU_API_KEY': 'test-zhipu-key-from-env',
    }):
        yield