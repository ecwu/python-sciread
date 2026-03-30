#!/usr/bin/env python3
"""
Example usage of the sciread LLM provider module.

This script demonstrates how to use the unified LLM provider interface
to create model instances for different providers.
"""

from sciread.llm_provider import ModelFactory


def main():
    """Demonstrate LLM provider usage."""
    print("SciRead LLM Provider Example")
    print("=" * 50)

    # Show all supported models
    print("\nSupported Models:")
    models = ModelFactory.list_all_supported_models()
    for model in models:
        print(f"  - {model}")

    # Show supported providers
    print("\nSupported Providers:")
    providers = ModelFactory.get_supported_providers()
    for provider, models in providers.items():
        print(f"  - {provider}: {', '.join(models.keys())}")

    # Demonstrate model creation
    print("\nModel Creation Examples:")

    examples = [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
        "volcengine/doubao-seed-2.0-code",
        "volcengine/glm-4.7",
        "ollama/qwen3:4b",
        "deepseek-chat",  # Auto-detect provider
    ]

    for model_id in examples:
        try:
            print(f"\n  Creating model: {model_id}")
            provider, model_name = ModelFactory.parse_model_identifier(model_id)
            print(f"    -> Provider: {provider}")
            print(f"    -> Model: {model_name}")

            # Note: Actual model creation requires API keys
            # model = get_model(model_id)
            # print(f"    -> Model instance created")

        except Exception as e:
            print(f"    -> Error: {e}")

    print("\nConfiguration:")
    print("  Set these environment variables for API keys:")
    print("    - DEEPSEEK_API_KEY")
    print("    - VOLCES_API")
    print("  - Ollama doesn't require an API key (local)")

    print("\nUsage in your code:")
    print(
        """
from sciread.llm_provider import get_model

# Get a model instance
model = get_model("deepseek/deepseek-chat")

# Use with pydantic-ai Agent
from pydantic_ai import Agent
agent = Agent(model)

# Or use directly
result = await model.run("Hello, world!")
"""
    )


if __name__ == "__main__":
    main()
