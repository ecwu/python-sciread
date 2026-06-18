"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sciread.platform.config import LLMProviderConfig
from sciread.platform.config import ScireadConfig
from sciread.platform.config import get_config
from sciread.platform.config import reload_config


class TestScireadConfig:
    """Test configuration loading and management."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ScireadConfig()

        assert config.providers.llm.default.provider == "deepseek"
        assert config.providers.llm.default.model == "deepseek-v4-flash"
        assert "deepseek" in config.providers.llm
        assert "volcengine" in config.providers.llm
        assert "lmstudio" in config.providers.llm
        assert "ollama" in config.providers.llm
        assert config.providers.llm["lmstudio"].base_url == "http://localhost:1234/v1"
        assert config.providers.llm["lmstudio"].api_key == "lm_studio"
        assert config.providers.embedding.default.model == "siliconflow/BAAI/bge-m3"
        assert config.document_splitters.default_splitter == "semantic"
        assert config.document_splitters.markdown.chunk_overlap == 0
        assert config.document_splitters.semantic.chunk_overlap == 0

    def test_load_from_file(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[providers.llm.custom]
api_key = "custom-key"
default_model = "custom-model"
base_url = "https://custom.api.com"

[providers.llm.default]
provider = "custom"
model = "custom-model"
"""
        config_file.write_text(config_content)

        config = ScireadConfig.load_from_file(config_file)

        assert config.providers.llm.default.provider == "custom"
        assert config.providers.llm.default.model == "custom-model"
        assert "custom" in config.providers.llm
        assert config.providers.llm["custom"].api_key == "custom-key"

    def test_load_splitter_overlap_configuration_from_file(self, tmp_path):
        """Test splitter overlap values load from TOML."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
[document_splitters.markdown]
chunk_overlap = 96

[document_splitters.semantic]
chunk_overlap = 144
""",
            encoding="utf-8",
        )

        config = ScireadConfig.load_from_file(config_file)

        assert config.document_splitters.markdown.chunk_overlap == 96
        assert config.document_splitters.semantic.chunk_overlap == 144

    def test_get_splitter_config_supports_markdown_and_semantic(self):
        """Test splitter config lookup covers the builder-backed splitters."""
        config = ScireadConfig()

        assert config.get_splitter_config("markdown") == config.document_splitters.markdown
        assert config.get_splitter_config("semantic") == config.document_splitters.semantic

    def test_environment_variable_substitution(self, tmp_path):
        """Test environment variable substitution in config."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[providers.llm.test]
api_key = "${TEST_API_KEY}"
default_model = "test-model"
"""
        config_file.write_text(config_content)

        with patch.dict(os.environ, {"TEST_API_KEY": "env-substituted-key"}):
            config = ScireadConfig.load_from_file(config_file)
            assert config.providers.llm["test"].api_key == "env-substituted-key"

    def test_get_provider_config(self):
        """Test getting provider configuration."""
        config = ScireadConfig()

        deepseek_config = config.get_provider_config("deepseek")
        assert isinstance(deepseek_config, LLMProviderConfig)
        assert deepseek_config.default_model == "deepseek-v4-flash"

    def test_loads_provider_embedding_and_rerank_defaults_from_file(self, tmp_path):
        """Embedding and rerank defaults should load from the providers namespace."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(
            """
[providers.embedding.default]
model = "lmstudio/custom-embedding"
batch_size = 7
cache_embeddings = false

[providers.rerank.default]
model = "siliconflow/custom-reranker"
candidate_multiplier = 2
""",
            encoding="utf-8",
        )

        config = ScireadConfig.load_from_file(config_file)

        assert config.providers.embedding.default.model == "lmstudio/custom-embedding"
        assert config.providers.embedding.default.batch_size == 7
        assert config.providers.embedding.default.cache_embeddings is False
        assert config.providers.rerank.default.model == "siliconflow/custom-reranker"
        assert config.providers.rerank.default.candidate_multiplier == 2

    def test_get_provider_config_invalid(self):
        """Test getting configuration for invalid provider."""
        config = ScireadConfig()

        with pytest.raises(ValueError, match="Unknown provider"):
            config.get_provider_config("invalid")

    def test_get_api_key_from_config(self, tmp_path):
        """Test getting API key from configuration."""
        config_file = tmp_path / "test_config.toml"
        config_content = """
[providers.llm.test]
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
[providers.llm.default]
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
            assert config.providers.llm.default.provider == "priority-test"
        finally:
            os.chdir(original_cwd)

    def test_load_from_file_uses_defaults_when_no_config_found(self, tmp_path):
        """When no config file exists, default settings should be returned."""
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            config = ScireadConfig.load_from_file()
            assert config.providers.llm.default.provider == "deepseek"
            assert config.config_file is None
        finally:
            os.chdir(original_cwd)

    def test_load_from_file_skips_non_dict_provider_entries(self, tmp_path):
        """Non-dict provider entries should be ignored."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("""
[providers.llm]
broken = "not-a-dict"

[providers.llm.custom]
api_key = "custom-key"
default_model = "custom-model"
""")
        config = ScireadConfig.load_from_file(config_file)
        assert "custom" in config.providers.llm
        assert "broken" not in config.providers.llm

    def test_load_from_file_ignores_legacy_provider_schema(self, tmp_path):
        """The old llm_providers namespace should not configure providers."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("""
[llm_providers.custom]
api_key = "legacy-key"
default_model = "legacy-model"

[llm_providers.default]
provider = "custom"
model = "legacy-model"
""")
        config = ScireadConfig.load_from_file(config_file)

        assert config.providers.llm.default.provider == "deepseek"
        assert "custom" not in config.providers.llm

    def test_load_from_file_uses_defaults_on_invalid_toml(self, tmp_path, capsys):
        """Invalid TOML should fall back to defaults with a warning."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("not valid toml {{[")

        config = ScireadConfig.load_from_file(config_file)
        assert config.providers.llm.default.provider == "deepseek"
        captured = capsys.readouterr()
        assert "Failed to load configuration" in captured.out

    def test_get_splitter_config_rejects_unknown_splitter(self):
        """Unknown splitter names should raise a clear error."""
        config = ScireadConfig()
        with pytest.raises(ValueError, match="Unknown splitter: unknown"):
            config.get_splitter_config("unknown")

    def test_get_default_splitter_config(self):
        """Default splitter config should match the configured default."""
        config = ScireadConfig()
        assert config.get_default_splitter_config() is config.document_splitters.semantic

    def test_get_mineru_token_from_config(self, tmp_path):
        """Mineru token should be readable from configuration."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("""
[mineru]
token = "mineru-config-token"
""")
        config = ScireadConfig.load_from_file(config_file)
        assert config.get_mineru_token() == "mineru-config-token"

    def test_get_mineru_token_from_environment(self, tmp_path):
        """Mineru token should fall back to the MINERU_TOKEN environment variable."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("[mineru]\n")
        config = ScireadConfig.load_from_file(config_file)

        with patch.dict(os.environ, {"MINERU_TOKEN": "mineru-env-token"}):
            assert config.get_mineru_token() == "mineru-env-token"

    def test_get_mineru_token_missing_raises(self, tmp_path):
        """Missing Mineru token should raise a clear error."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("[mineru]\n")
        config = ScireadConfig.load_from_file(config_file)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No Mineru token found"):
                config.get_mineru_token()

    def test_get_api_key_falls_back_to_generic_env_name(self):
        """Providers without a standard env mapping should use PROVIDER_API_KEY."""
        config = ScireadConfig()
        config.providers.llm["custom"] = LLMProviderConfig(default_model="custom-model")

        with patch.dict(os.environ, {"CUSTOM_API_KEY": "custom-env-key"}):
            assert config.get_api_key("custom") == "custom-env-key"

    def test_get_api_key_uses_volces_env_for_volcengine(self):
        """volcengine should read the VOLCES_API environment variable."""
        config = ScireadConfig()

        with patch.dict(os.environ, {"VOLCES_API": "volces-env-key"}, clear=False):
            assert config.get_api_key("volcengine") == "volces-env-key"

    def test_get_api_key_uses_ark_env_for_ark_provider(self):
        """An ark provider should read the ARK_API_KEY environment variable."""
        config = ScireadConfig()
        config.providers.llm["ark"] = LLMProviderConfig(default_model="doubao-pro")

        with patch.dict(os.environ, {"ARK_API_KEY": "ark-env-key"}):
            assert config.get_api_key("ark") == "ark-env-key"

    def test_global_config_and_reload(self, tmp_path):
        """get_config and reload_config should manage the global config instance."""
        config_file = tmp_path / "sciread.toml"
        config_file.write_text("""
[providers.llm.default]
provider = "custom"
model = "custom-model"

[providers.llm.custom]
default_model = "custom-model"
""")
        original = get_config()
        assert original.providers.llm.default.provider == "deepseek"

        reloaded = reload_config(config_file)
        assert reloaded.providers.llm.default.provider == "custom"
        assert get_config().providers.llm.default.provider == "custom"

        # Restore default global config for other tests
        reload_config()
