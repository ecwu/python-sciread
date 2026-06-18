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


class DefaultConfig(BaseModel):
    """Default provider and model settings."""

    provider: str = Field(default="deepseek", description="Default provider name")
    model: str = Field(default="deepseek-v4-flash", description="Default model name")


class LLMProvidersConfig(BaseModel):
    """Configuration for LLM providers."""

    default: DefaultConfig = Field(default_factory=DefaultConfig)
    configs: dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "deepseek": LLMProviderConfig(default_model="deepseek-v4-flash", base_url="https://api.deepseek.com"),
            "volcengine": LLMProviderConfig(
                default_model="doubao-seed-2.0-code",
                base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            ),
            "lmstudio": LLMProviderConfig(default_model="qwen3:4b", base_url="http://localhost:1234/v1", api_key="lm_studio"),
            "ollama": LLMProviderConfig(default_model="qwen3:4b", base_url="http://localhost:11434/v1"),
        },
        description="Provider-specific LLM configuration",
    )

    def __contains__(self, provider_name: str) -> bool:
        return provider_name in self.configs

    def __getitem__(self, provider_name: str) -> LLMProviderConfig:
        return self.configs[provider_name]

    def __setitem__(self, provider_name: str, provider_config: LLMProviderConfig) -> None:
        self.configs[provider_name] = provider_config


class EmbeddingDefaultConfig(BaseModel):
    """Default embedding provider settings."""

    model: str = Field(default="siliconflow/BAAI/bge-m3", description="Default model for embeddings")
    batch_size: int = Field(default=10, ge=1, description="Embedding batch size")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings for better performance")


class EmbeddingProvidersConfig(BaseModel):
    """Configuration for embedding providers."""

    default: EmbeddingDefaultConfig = Field(default_factory=EmbeddingDefaultConfig)


class RerankDefaultConfig(BaseModel):
    """Default rerank provider settings."""

    model: str = Field(default="siliconflow/BAAI/bge-reranker-v2-m3", description="Model for reranking semantic candidates")
    candidate_multiplier: int = Field(default=4, ge=1, description="How many semantic candidates to rerank per requested result")


class RerankProvidersConfig(BaseModel):
    """Configuration for rerank providers."""

    default: RerankDefaultConfig = Field(default_factory=RerankDefaultConfig)


class ProvidersConfig(BaseModel):
    """Configuration grouped by provider domain."""

    llm: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)
    embedding: EmbeddingProvidersConfig = Field(default_factory=EmbeddingProvidersConfig)
    rerank: RerankProvidersConfig = Field(default_factory=RerankProvidersConfig)


class MarkdownSplitterConfig(BaseModel):
    """Configuration for MarkdownSplitter."""

    min_chunk_size: int = Field(default=200, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(default=0, ge=0, description="Backward overlap between adjacent chunks in characters")
    preserve_code_blocks: bool = Field(default=True, description="Keep markdown code blocks intact while chunking")
    split_on_headers: bool = Field(default=True, description="Split markdown content on headings")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence score for chunks")


class SemanticSplitterConfig(BaseModel):
    """Configuration for SemanticSplitter."""

    min_chunk_size: int = Field(default=200, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(default=0, ge=0, description="Backward overlap between adjacent chunks in characters")
    preserve_code_blocks: bool = Field(default=True, description="Keep code blocks intact while chunking")
    split_on_headers: bool = Field(default=True, description="Split content on detected section boundaries")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence score for chunks")
    enable_academic_patterns: bool = Field(default=True, description="Enable academic section pattern detection")
    enable_markdown_patterns: bool = Field(default=False, description="Enable markdown pattern detection")


class DocumentSplitterConfig(BaseModel):
    """Configuration for document splitters."""

    default_splitter: str = Field(default="semantic", description="Default non-markdown splitter to use")
    markdown: MarkdownSplitterConfig = Field(default_factory=MarkdownSplitterConfig)
    semantic: SemanticSplitterConfig = Field(default_factory=SemanticSplitterConfig)


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


class ScireadConfig(BaseSettings):
    """Main configuration class for sciread package."""

    model_config = SettingsConfigDict(
        env_prefix="SCIREAD_",
        env_file=".env",
        extra="ignore",
    )

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
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

            # Extract provider configurations.
            raw_provider_sections = config_data.get("providers", {})
            if not isinstance(raw_provider_sections, dict):
                raw_provider_sections = {}
            raw_llm_providers = raw_provider_sections.get("llm", {})
            if not isinstance(raw_llm_providers, dict):
                raw_llm_providers = {}
            llm_provider_configs = LLMProvidersConfig().configs.copy() if not raw_llm_providers else {}

            for name, data in raw_llm_providers.items():
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
                llm_provider_configs[name] = LLMProviderConfig(
                    api_key=api_key,
                    base_url=data.get("base_url"),
                    default_model=data.get("default_model", name),
                )

            # Extract default provider configurations
            raw_llm_default = raw_llm_providers.get("default", {})
            if not isinstance(raw_llm_default, dict):
                raw_llm_default = {}
            llm_default = DefaultConfig(
                provider=raw_llm_default.get("provider", "deepseek"),
                model=raw_llm_default.get("model", "deepseek-v4-flash"),
            )

            raw_embedding = raw_provider_sections.get("embedding", {})
            if not isinstance(raw_embedding, dict):
                raw_embedding = {}
            raw_embedding_default = raw_embedding.get("default", {})
            if not isinstance(raw_embedding_default, dict):
                raw_embedding_default = {}
            embedding_default = EmbeddingDefaultConfig(**raw_embedding_default)

            raw_rerank = raw_provider_sections.get("rerank", {})
            if not isinstance(raw_rerank, dict):
                raw_rerank = {}
            raw_rerank_default = raw_rerank.get("default", {})
            if not isinstance(raw_rerank_default, dict):
                raw_rerank_default = {}
            rerank_default = RerankDefaultConfig(**raw_rerank_default)

            providers = ProvidersConfig(
                llm=LLMProvidersConfig(default=llm_default, configs=llm_provider_configs),
                embedding=EmbeddingProvidersConfig(default=embedding_default),
                rerank=RerankProvidersConfig(default=rerank_default),
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
                providers=providers,
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
        if provider_name not in self.providers.llm:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers.llm[provider_name]

    def get_api_key(self, provider_name: str) -> str:
        """Get API key for a specific provider.

        Prioritizes:
        1. Contextual configuration (from file or explicit set)
        2. Standard environment variables (e.g., DEEPSEEK_API_KEY)
        3. Generic provider environment variables (e.g., CUSTOM_API_KEY)
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
        if splitter_name == "markdown":
            return self.document_splitters.markdown
        elif splitter_name == "semantic":
            return self.document_splitters.semantic
        else:
            raise ValueError(f"Unknown splitter: {splitter_name}. Available splitters: markdown, semantic")

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
