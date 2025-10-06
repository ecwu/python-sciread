"""Configuration management for sciread package."""

import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# Handle toml library compatibility
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        pass


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider API")
    default_model: str = Field(description="Default model name for this provider")


class RegexSectionSplitterConfig(BaseModel):
    """Configuration for RegexSectionSplitter."""

    min_chunk_size: int = Field(default=200, description="Minimum chunk size in characters")
    confidence_threshold: float = Field(default=0.3, description="Minimum confidence score for chunks")
    merge_small_chunks: bool = Field(default=True, description="Whether to merge small chunks with neighbors")
    custom_patterns: dict[str, str] = Field(default_factory=dict, description="Custom regex patterns")


class TopicFlowSplitterConfig(BaseModel):
    """Configuration for TopicFlowSplitter."""

    model: str = Field(default="embeddinggemma:latest", description="Ollama model for embeddings")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    # Continuity thresholds
    local_continuity_threshold: float = Field(default=0.6, description="Threshold for local continuity (adjacent sentences)")
    context_continuity_threshold: float = Field(default=0.65, description="Threshold for context continuity (segment vs sentence)")
    # Size constraints
    min_segment_sentences: int = Field(default=4, description="Minimum sentences per segment for content-based cuts")
    min_segment_chars: int = Field(default=300, description="Minimum characters per segment")
    max_segment_chars: int = Field(default=2000, description="Maximum characters per segment (hard budget limit)")
    # Processing parameters
    embedding_batch_size: int = Field(default=10, description="Batch size for embedding requests")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    cache_embeddings: bool = Field(default=True, description="Whether to cache embeddings")
    # Adaptive thresholds
    adaptive_floor: float = Field(default=0.4, description="Adaptive floor for local continuity detection")
    soft_target: float = Field(default=0.7, description="Soft target for context continuity")


class DocumentSplitterConfig(BaseModel):
    """Configuration for document splitters."""

    default_splitter: str = Field(default="topic_flow", description="Default splitter to use")
    regex_section: RegexSectionSplitterConfig = Field(default_factory=RegexSectionSplitterConfig)
    topic_flow: TopicFlowSplitterConfig = Field(default_factory=TopicFlowSplitterConfig)


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
            "zhipu": LLMProviderConfig(
                default_model="glm-4.6",
                base_url="https://open.bigmodel.cn/api/anthropic",
            ),
            "ollama": LLMProviderConfig(default_model="qwen3:4b", base_url="http://localhost:11434/v1"),
        }
    )

    default: DefaultConfig = Field(default=DefaultConfig(provider="deepseek", model="deepseek-chat"))
    document_splitters: DocumentSplitterConfig = Field(default_factory=DocumentSplitterConfig)

    config_file: Optional[Path] = Field(default=None, description="Path to configuration file")

    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> "ScireadConfig":
        """Load configuration from file and environment variables."""

        if config_path is None:
            # Look for config in standard locations
            config_paths = [
                Path.cwd() / "config" / "sciread.toml",
                Path.cwd() / "sciread.toml",
                Path.home() / ".config" / "sciread" / "config.toml",
                Path.home() / ".sciread.toml",
            ]

            for path in config_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                # No config file found, use defaults
                return cls()

        try:
            with config_path.open("rb") as f:
                config_data = tomllib.load(f)

            # Extract provider configurations
            providers_config = config_data.get("llm_providers", {})
            providers = {}

            for provider_name, provider_data in providers_config.items():
                # Support environment variable substitution
                api_key = provider_data.get("api_key")
                if api_key and isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.getenv(env_var)

                providers[provider_name] = LLMProviderConfig(
                    api_key=api_key,
                    base_url=provider_data.get("base_url"),
                    default_model=provider_data.get("default_model", provider_name),
                )

            # Extract default configuration
            default_config = config_data.get("llm_providers", {}).get("default", {})
            default_settings = DefaultConfig(
                provider=default_config.get("provider", "deepseek"),
                model=default_config.get("model", "deepseek-chat"),
            )

            # Extract document splitters configuration
            splitters_config = config_data.get("document_splitters", {})
            document_splitters = DocumentSplitterConfig(**splitters_config)

            return cls(
                llm_providers=providers,
                default=default_settings,
                document_splitters=document_splitters,
                config_file=config_path,
            )

        except Exception:
            # If config file is invalid, use defaults
            return cls()

    def get_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name not in self.llm_providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.llm_providers[provider_name]

    def get_api_key(self, provider_name: str) -> str:
        """Get API key for a specific provider."""
        provider_config = self.get_provider_config(provider_name)
        if not provider_config.api_key:
            # Try environment variable
            env_var_name = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(env_var_name)
            if not api_key:
                raise ValueError(
                    f"No API key found for provider '{provider_name}'. Set {env_var_name} environment variable or configure in config file."
                )
            return api_key
        return provider_config.api_key

    def get_splitter_config(self, splitter_name: str):
        """Get configuration for a specific splitter."""
        if splitter_name == "regex_section":
            return self.document_splitters.regex_section
        elif splitter_name == "regex":
            # Backward compatibility - map to regex_section
            return self.document_splitters.regex_section
        elif splitter_name == "topic_flow":
            return self.document_splitters.topic_flow
        else:
            raise ValueError(f"Unknown splitter: {splitter_name}. Available splitters: regex_section, topic_flow")

    def get_default_splitter_config(self):
        """Get configuration for the default splitter."""
        default_name = self.document_splitters.default_splitter
        return self.get_splitter_config(default_name)


# Global configuration instance
_config: Optional[ScireadConfig] = None


def get_config() -> ScireadConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ScireadConfig.load_from_file()
    return _config


def reload_config(config_path: Optional[Path] = None) -> ScireadConfig:
    """Reload configuration from file."""
    global _config
    _config = ScireadConfig.load_from_file(config_path)
    return _config
