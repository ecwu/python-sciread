"""Configuration management for sciread package."""

import os
import sys
from pathlib import Path
from typing import Dict
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
        import tomli as tomllib
    except ImportError:
        pass


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider API")
    default_model: str = Field(description="Default model name for this provider")


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

    llm_providers: Dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "deepseek": LLMProviderConfig(default_model="deepseek-chat", base_url="https://api.deepseek.com"),
            "zhipu": LLMProviderConfig(default_model="glm-4.6", base_url="https://open.bigmodel.cn/api/anthropic"),
            "ollama": LLMProviderConfig(default_model="qwen3:4b", base_url="http://localhost:11434/v1"),
        }
    )

    default: DefaultConfig = Field(default=DefaultConfig(provider="deepseek", model="deepseek-chat"))

    config_file: Optional[Path] = Field(default=None, description="Path to configuration file")

    @classmethod
    def load_from_file(cls, config_path: Optional[Path] = None) -> "ScireadConfig":
        """Load configuration from file and environment variables."""
        import tomllib

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
            with open(config_path, "rb") as f:
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
                    api_key=api_key, base_url=provider_data.get("base_url"), default_model=provider_data.get("default_model", provider_name)
                )

            # Extract default configuration
            default_config = config_data.get("llm_providers", {}).get("default", {})
            default_settings = DefaultConfig(
                provider=default_config.get("provider", "deepseek"), model=default_config.get("model", "deepseek-chat")
            )

            return cls(llm_providers=providers, default=default_settings, config_file=config_path)

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
