# AGENTS.md - Agent Guidelines for sciread

## Build, Lint, and Test Commands

```bash
# Install runtime dependencies
uv sync

# Install test dependencies
uv sync --group test

# Install full local dev environment
uv sync --group test --group dev

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_cli.py
uv run pytest tests/test_document/test_document.py

# Run a specific test
uv run pytest tests/test_cli.py::test_render_coordinate_plan_shows_subagents_and_sections

# Run a test module/directory
uv run pytest tests/test_document/
uv run pytest tests/test_llm_provider/

# Run with coverage
uv run pytest --cov src/ tests/
```

## Repository Layout

The current package structure is organized by subsystem:

- `src/sciread/agent/`: agent implementations for `simple`, `coordinate`, `react`, and `discussion`
- `src/sciread/application/use_cases/`: top-level orchestration for CLI-facing workflows
- `src/sciread/document/`: document API, builders, models, ingestion, structure, and retrieval
- `src/sciread/embedding_provider/`: embedding clients and factory helpers
- `src/sciread/llm_provider/`: LLM provider integrations and factories
- `src/sciread/platform/`: shared config and logging
- `src/sciread/entrypoints/cli.py`: CLI parser and rendering
- `config/sciread.toml`: example project configuration
- `tests/`: subsystem tests plus top-level CLI and integration-style tests

## Code Style Guidelines

### Imports

Order imports as stdlib -> third-party -> local.

Use one import per line for imported names because Ruff is configured with `force_single_line = true`.

```python
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ..platform.logging import get_logger
from .models import Chunk
```

Never use `from module import *`.

### Formatting and Style

- Line length: 140
- Quotes: double quotes
- Indentation: 4 spaces
- Keep trailing whitespace out of the repo
- Run `uv run ruff format src/ tests/` after substantial edits

### Type Hints

Prefer modern Python 3.12+ syntax:

```python
def analyze(document_path: str, max_loops: int = 8) -> str: ...

def load_config(path: Path | None = None) -> ScireadConfig: ...

chunks: list[Chunk]
providers: dict[str, LLMProviderConfig]
page_range: tuple[int, int] | None

def process(**kwargs: Any) -> None: ...
```

Use `| None` instead of `Optional[...]` unless matching existing local style in untouched code.

### Naming Conventions

```python
class SimpleAgent: ...
class DocumentBuilder: ...
class AnalysisTimeoutError: ...

def run_simple_analysis(): ...
def ensure_file_exists(): ...
async def analyze_document(): ...

DEFAULT_TIMEOUT = 300.0
TEXT_FILE_EXTENSIONS = {".txt", ".md"}
```

Patterns already used in this repo:

- Agents: `*Agent`
- Result models: `*Result`
- Config models: `*Config`
- Factories/builders: `*Factory`, `*Builder`
- Exceptions: `*Error`

## Logging

Use the shared logging helpers in `src/sciread/platform/logging.py`.

```python
from ...platform.logging import get_logger

logger = get_logger(__name__)

class MyClass:
    def __init__(self):
        self.logger = get_logger(__name__)
```

Prefer `self.logger` on stateful classes and module-level `logger` in utility modules.

Debug runs typically use:

```bash
LOG_LEVEL=DEBUG uv run sciread coordinate paper.pdf
```

## Error Handling

Shared agent error utilities live in `src/sciread/agent/shared/error_handling.py`.

Use the existing helpers where applicable:

```python
from ..shared.error_handling import AnalysisTimeoutError
from ..shared.error_handling import safe_agent_execution

result = await safe_agent_execution(
    self.agent.run(prompt, deps=deps),
    timeout=self.timeout,
    operation_name="discussion analysis",
    error_type=AnalysisTimeoutError,
)
```

Rules:

- Raise domain-specific exceptions when the failure is part of the package contract
- Chain exceptions with `raise ... from e`
- Log failures before re-raising when they add useful runtime context
- Reuse `handle_model_retry()` for pydantic-ai retry translation

## Async and Agent Patterns

- Agent methods that call models should remain `async def`
- CLI handlers should bridge to async code with `asyncio.run(...)`
- Reuse application-layer use cases from `src/sciread/application/use_cases/` instead of embedding orchestration directly in the CLI
- Keep prompt definitions alongside the agent implementation that uses them
- Prefer shared document-loading helpers such as `load_document()` and `ensure_file_exists()` from `application/use_cases/common.py`

## Config and Providers

Configuration is centralized in `src/sciread/platform/config.py` and `config/sciread.toml`.

Current repo conventions:

- Environment prefix: `SCIREAD_`
- Provider config models live in `ScireadConfig`
- API keys may come from config-file placeholders or environment variables
- Do not hardcode secrets in code or tests

Current provider and document-related settings include:

- `llm_providers`
- `document_splitters`
- `mineru`
- `vector_store`

If you add a new configurable behavior, update both the config model and the sample config file when appropriate.

## Document Pipeline

The canonical document entrypoint is `sciread.document.Document`.

Preferred patterns:

```python
document = Document.from_file("paper.pdf", to_markdown=False, auto_split=True)
document = Document.from_text(raw_text, auto_split=True)
```

Implementation notes:

- `DocumentFactory` and `DocumentBuilder` own document construction
- PDF and text loading live under `document/ingestion/loaders/`
- Default splitting currently routes markdown documents through `MarkdownSplitter`
- Non-markdown documents default to `SemanticSplitter`
- Retrieval and vector indexing live under `document/retrieval/`

Avoid bypassing the builder/factory stack unless there is a clear reason.

## Data Models

Use:

- `pydantic.BaseModel` for configuration and validated structured inputs
- `@dataclass` for lightweight document and state containers

Examples already present in the repo:

- `LLMProviderConfig`, `ScireadConfig`: `BaseModel` / `BaseSettings`
- `Chunk`, `DocumentMetadata`, `ProcessingState`: dataclasses

Use `field(default_factory=...)` for mutable defaults and generated identifiers.

## Testing Guidelines

The tests directory is a mix of mirrored subsystem tests and top-level behavior tests. Follow the existing structure instead of forcing everything into a single pattern.

Examples:

- `tests/test_document/`: document subsystem tests and fixtures
- `tests/test_llm_provider/`: provider tests and fixtures
- `tests/test_cli.py`: CLI behavior and rendering tests
- `tests/test_coordinate_agent.py`, `tests/test_react_agent.py`: agent-focused integration-style tests

Guidelines:

- Add tests for new behavior
- Prefer the nearest existing test module for the subsystem you are changing
- Use `unittest.mock.patch` or fixtures to isolate provider/network dependencies
- Keep tests deterministic and avoid real external service calls unless the file is explicitly for integration coverage

## CLI Patterns

The CLI entrypoint is `src/sciread/entrypoints/cli.py` and uses subcommands.

Current commands:

- `simple`
- `coordinate`
- `react`
- `discussion`

When adding a command:

1. Add a subparser in `cli.py`
2. Prefer `--model` style options over positional model arguments
3. Wire the command to an application-layer use case in `src/sciread/application/use_cases/`
4. Return integer exit codes from the CLI entrypoint
5. Add or update CLI tests in `tests/test_cli.py`

## Architectural Principles

1. Async-first model interaction
2. Application-layer orchestration for CLI workflows
3. Factory/builder-based document creation
4. Config-driven provider and splitter behavior
5. Shared logging and error-handling utilities
6. Typed public APIs and structured result objects
7. Clear separation between ingestion, structure, retrieval, and agent logic

## Guardrails

- Never commit unless the user explicitly asks
- Never remove or revert unrelated user changes
- Never hardcode credentials or tokens
- Always run `uv run ruff check src/ tests/` after code changes when feasible
- Always add or update tests for new behavior
- Always keep `AGENTS.md`, `README.rst`, and config examples aligned when changing developer-facing workflows
