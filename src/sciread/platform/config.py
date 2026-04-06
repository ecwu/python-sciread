"""Configuration management for sciread package."""

import os

# Handle toml library compatibility
import tomllib
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: str | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Base URL for the provider API")
    default_model: str = Field(description="Default model name for this provider")


class RegexSectionSplitterConfig(BaseModel):
    """Configuration for RegexSectionSplitter."""

    min_chunk_size: int = Field(default=200, description="Minimum chunk size in characters")
    confidence_threshold: float = Field(default=0.3, description="Minimum confidence score for chunks")
    custom_patterns: dict[str, str] = Field(default_factory=dict, description="Custom regex patterns")


class DocumentSplitterConfig(BaseModel):
    """Configuration for document splitters."""

    default_splitter: str = Field(default="regex_section", description="Default splitter to use")
    regex_section: RegexSectionSplitterConfig = Field(default_factory=RegexSectionSplitterConfig)


class MineruConfig(BaseModel):
    """Configuration for Mineru API."""

    token: str | None = Field(default=None, description="API token for Mineru service")
    enable_formula: bool = Field(default=True, description="Enable formula extraction")
    enable_table: bool = Field(default=True, description="Enable table extraction")
    language: str = Field(default="ch", description="Document language (ch/en)")
    timeout: int = Field(default=600, description="Processing timeout in seconds")
    poll_interval: int = Field(default=10, description="Status poll interval in seconds")
    enable_cache: bool = Field(default=True, description="Enable caching of API responses")
    cache_dir: str | None = Field(
        default=None,
        description="Directory for cache storage (default: ~/.sciread/mineru_cache)",
    )


class VectorStoreConfig(BaseModel):
    """Configuration for vector store (RAG functionality)."""

    path: str = Field(default="~/.sciread/vector_store", description="Path to store vector indices")
    embedding_model: str = Field(default="embeddinggemma:latest", description="Model for embeddings")
    batch_size: int = Field(default=10, description="Embedding batch size")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings for better performance")


class DefaultConfig(BaseModel):
    """Default provider and model settings."""

    provider: str = Field(default="deepseek", description="Default provider name")
    model: str = Field(default="deepseek-chat", description="Default model name")


class ScireadConfig(BaseSettings):
    """Main configuration class for sciread package."""

    model_config = SettingsConfigDict(
        env_prefix="SCIREAD_",
        env_file=".env",
        extra="ignore",
    )

    llm_providers: dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "deepseek": LLMProviderConfig(default_model="deepseek-chat", base_url="https://api.deepseek.com"),
            "volcengine": LLMProviderConfig(
                default_model="doubao-seed-2.0-code",
                base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            ),
            "ollama": LLMProviderConfig(default_model="qwen3:4b", base_url="http://localhost:11434/v1"),
        }
    )

    default: DefaultConfig = Field(default=DefaultConfig(provider="deepseek", model="deepseek-chat"))
    document_splitters: DocumentSplitterConfig = Field(default_factory=DocumentSplitterConfig)
    mineru: MineruConfig = Field(default_factory=MineruConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    config_file: Path | None = Field(default=None, description="Path to configuration file")

    @classmethod
    def load_from_file(cls, config_path: Path | None = None) -> "ScireadConfig":
        """Load configuration from file and environment variables."""

        found_config_path = config_path
        if found_config_path is None:
            # Look for config in standard locations
            config_paths = [
                Path.cwd() / "config" / "sciread.toml",
                Path.cwd() / "sciread.toml",
                Path.home() / ".config" / "sciread" / "config.toml",
                Path.home() / ".sciread.toml",
            ]

            for path in config_paths:
                if path.exists():
                    found_config_path = path
                    break

        # If no config file found, start with default settings
        if not found_config_path:
            return cls()

        try:
            with found_config_path.open("rb") as f:
                config_data = tomllib.load(f)

            # Extract provider configurations
            raw_providers = config_data.get("llm_providers", {})
            providers = {}

            for name, data in raw_providers.items():
                if name == "default":
                    continue

                if not isinstance(data, dict):
                    continue

                # Process environment variable substitution in file
                api_key = data.get("api_key")
                if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.getenv(env_var)

                # If still None, it will be lazily loaded from env by get_api_key()
                providers[name] = LLMProviderConfig(
                    api_key=api_key,
                    base_url=data.get("base_url"),
                    default_model=data.get("default_model", name),
                )

            # Extract default configuration
            raw_default = raw_providers.get("default", {})
            default_settings = DefaultConfig(
                provider=raw_default.get("provider", "deepseek"),
                model=raw_default.get("model", "deepseek-chat"),
            )

            # Extract document splitters
            splitters_config = config_data.get("document_splitters", {})
            document_splitters = DocumentSplitterConfig(**splitters_config)

            # Extract Mineru configuration
            mineru_config = config_data.get("mineru", {})
            mineru_token = mineru_config.get("token")
            if isinstance(mineru_token, str) and mineru_token.startswith("${") and mineru_token.endswith("}"):
                env_var = mineru_token[2:-1]
                mineru_token = os.getenv(env_var)
            mineru_config["token"] = mineru_token
            mineru = MineruConfig(**mineru_config)

            # Extract vector store configuration
            vector_store_config = config_data.get("vector_store", {})
            vector_store = VectorStoreConfig(**vector_store_config)

            return cls(
                llm_providers=providers,
                default=default_settings,
                document_splitters=document_splitters,
                mineru=mineru,
                vector_store=vector_store,
                config_file=found_config_path,
            )

        except Exception as e:
            # If config file is invalid, use defaults but log warning
            print(f"Warning: Failed to load configuration from {found_config_path}: {e}")
            return cls()

    def get_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name not in self.llm_providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.llm_providers[provider_name]

    def get_api_key(self, provider_name: str) -> str:
        """Get API key for a specific provider.

        Prioritizes:
        1. Contextual configuration (from file or explicit set)
        2. Standard environment variables (e.g., DEEPSEEK_API_KEY)
        3. Prefixed environment variables (e.g., SCIREAD_LLM_PROVIDERS__DEEPSEEK__API_KEY)
        """
        provider_config = self.get_provider_config(provider_name)
        if provider_config.api_key:
            return provider_config.api_key

        # Try standard environment variable names first
        standard_env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "volcengine": "VOLCES_API",
            "ark": "ARK_API_KEY",
        }

        env_var_name = standard_env_map.get(provider_name.lower())
        if not env_var_name:
            env_var_name = f"{provider_name.upper()}_API_KEY"

        api_key = os.getenv(env_var_name)
        if api_key:
            return api_key

        # If not found, raise informative error
        raise ValueError(
            f"No API key found for provider '{provider_name}'. Set {env_var_name} environment variable or configure in sciread.toml."
        )

    def get_splitter_config(self, splitter_name: str):
        """Get configuration for a specific splitter."""
        if splitter_name == "regex_section":
            return self.document_splitters.regex_section
        else:
            raise ValueError(f"Unknown splitter: {splitter_name}. Available splitters: regex_section")

    def get_default_splitter_config(self):
        """Get configuration for the default splitter."""
        default_name = self.document_splitters.default_splitter
        return self.get_splitter_config(default_name)

    def get_mineru_token(self) -> str:
        """Get Mineru API token."""
        if self.mineru.token:
            return self.mineru.token

        # Try standard environment variable
        api_key = os.getenv("MINERU_TOKEN")
        if not api_key:
            raise ValueError("No Mineru token found. Set MINERU_TOKEN environment variable or configure in sciread.toml.")
        return api_key


# Global configuration instance
_config: ScireadConfig | None = None


def get_config() -> ScireadConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ScireadConfig.load_from_file()
    return _config


def reload_config(config_path: Path | None = None) -> ScireadConfig:
    """Reload configuration from file."""
    global _config
    _config = ScireadConfig.load_from_file(config_path)
    return _config
