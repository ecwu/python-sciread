"""Test configuration and fixtures for LLM provider tests."""

import os
from unittest.mock import patch

import pytest

from sciread.platform.config import ScireadConfig


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
[providers.llm.deepseek]
api_key = "test-deepseek-key"
default_model = "deepseek-v4-flash"
base_url = "https://api.deepseek.com"

[providers.llm.volcengine]
api_key = "test-volcengine-key"
default_model = "doubao-seed-2.0-code"
base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"

[providers.llm.lmstudio]
api_key = "lm_studio"
default_model = "qwen3:4b"
base_url = "http://localhost:1234/v1"

[providers.llm.ollama]
base_url = "http://localhost:11434/v1"
default_model = "qwen3:4b"

[providers.llm.default]
provider = "deepseek"
model = "deepseek-v4-flash"
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_config(test_config_file):
    """Mock configuration for testing."""
    with (
        patch("sciread.providers.llm.deepseek.get_config") as mock_deepseek,
        patch("sciread.providers.llm.volcengine.get_config") as mock_volcengine,
        patch("sciread.providers.llm.lmstudio.get_config") as mock_lmstudio,
        patch("sciread.providers.llm.ollama.get_config") as mock_ollama,
        patch("sciread.providers.llm.factory.get_config") as mock_factory,
    ):
        config = ScireadConfig.load_from_file(test_config_file)
        mock_deepseek.return_value = config
        mock_volcengine.return_value = config
        mock_lmstudio.return_value = config
        mock_ollama.return_value = config
        mock_factory.return_value = config

        yield config


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "test-deepseek-key-from-env",
            "VOLCES_API": "test-volcengine-key-from-env",
        },
    ):
        yield
