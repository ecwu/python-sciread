# AGENTS.md - Agent Guidelines for sciread

## Build, Lint, and Test Commands

```bash
# Install: pip install -e ".[test]"

# Lint (fails on errors): ruff check src/ tests/
- Format: ruff format src/ tests/

# Run all tests: tox

# Run single test file: pytest tests/test_core.py
- Run specific test: pytest tests/test_core.py::test_compute
- Run with coverage: pytest --cov src/ tests/test_core.py

# Run tests by module: pytest tests/test_document/
- Run tests with verbose: pytest -v tests/test_document/test_document.py

# Docs: tox -e docs && open dist/docs/index.html
```

## Code Style Guidelines

### Imports
**Order**: stdlib → third-party → local (relative imports)
```python
from pathlib import Path
from typing import Optional, Any
from pydantic_ai import Agent
from .error_handling import handle_model_retry
from ..document import Document
```
**Never**: `from module import *`
**Group imports** logically with blank lines between groups

### Formatting & Style
- **Line length**: 140 chars (pyproject.toml: `[tool.ruff] line-length = 140`)
- **Quotes**: Double quotes for strings
- **Indentation**: 4 spaces
- **Trailing whitespace**: Never
- **Run**: `ruff format src/ tests/` before committing

### Type Hints
Use modern syntax consistently:
```python
# Primitives and simple types
def func(name: str, count: int) -> str: ...

# Nullable
def maybe(data: Optional[dict]) -> None: ...

# Collections (modern syntax)
items: list[Chunk]
mapping: dict[str, Config]
pairs: tuple[str, int]

# Type aliasing
ModelId = str
Embedding = list[float]

# Functions with flexible args
def process(**kwargs: Any) -> None: ...
```

### Naming Conventions
```python
# Classes: PascalCase, descriptive
class SimpleAgent: ...
class DocumentFactory: ...
class AnalysisTimeoutError: ...

# Functions/methods: snake_case, verb-focused
def analyze_document(): ...
def get_config(): ...
async def process_chunks(): ...

# Private members: _prefix
def _calculate_hash(): ...
self._internal_state = ...

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300.0
SUPPORTED_MODELS = {...}

# Modules/files: snake_case
# simple_agent.py, document_builder.py, error_handling.py
```

**Class suffix patterns**:
- Agents: `*Agent`
- Models/Results: `*Model`, `*Result`
- Configs: `*Config`
- Factories: `*Factory`
- Exceptions: `*Error`

### Error Handling
```python
# Custom exceptions (inherit from base)
class AgentError(Exception):
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

# Pattern: validate, log, raise
try:
    result = await operation()
    return result
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise OperationError(f"Could not complete: {e}") from e
```
**Rules**:
- Use `raise ... from e` to chain exceptions
- Log before re-raising
- Provide user-friendly messages with suggestions
- Define domain-specific exceptions in `agent/error_handling.py`

### Logging (loguru)
```python
from ..logging_config import get_logger

logger = get_logger(__name__)

class MyClass:
    def __init__(self):
        self.logger = get_logger(__name__)

# Usage
self.logger.debug("Detailed info for debugging")
self.logger.info("Milestone completed")
self.logger.warning("Recoverable issue detected")
self.logger.error(f"Failure: {exception}")
```
**Debug mode**: `LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf`

### Async/Await Patterns
```python
# Agent methods calling LLMs must be async
async def analyze(self, document: Document) -> str:
    try:
        result = await safe_agent_execution(
            self.agent.run("Task", deps=deps),
            timeout=self.timeout,
            operation_name="analysis",
        )
        return result
    except Exception as e:
        self.logger.error(f"Analysis failed: {e}")
        raise

# CLI wiring: wrap async with asyncio.run()
result = asyncio.run(async_function(arg1, arg2))
```
**Rules**:
- Use `async def` for all LLM-calling methods
- Use `await safe_agent_execution()` for timeouts
- Use `asyncio.run()` in CLI to bridge sync→async
- Track state across async iterations with dataclasses

### Pydantic Models
```python
# Configuration (use BaseModel)
from pydantic import BaseModel, Field

class LLMProviderConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, description="API key")
    default_model: str = Field(description="Default model name")

# Data containers (use @dataclass)
from dataclasses import dataclass, field
import uuid

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    content: str
    confidence: float = 1.0

    def __post_init__(self):
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
```
**Rules**:
- `BaseModel` for config with validation
- `@dataclass` for simple data containers
- Use `Field()` for descriptions and defaults
- Use `default_factory` for mutable defaults
- Use `init=False` for computed fields

## Testing Guidelines

### Test File Structure
Mirror `src/` directory structure:
```
tests/
├── test_llm_provider/
│   ├── conftest.py          # Shared fixtures
│   └── test_factory.py       # Feature tests
├── test_document/
│   └── conftest.py
└── test_core.py
```

### Fixture Patterns
```python
# Simple fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

# Content fixtures
@pytest.fixture
def sample_text() -> str:
    return "Abstract\nTest content..."

# Composable fixtures
@pytest.fixture
def test_file(temp_dir: Path) -> Path:
    file_path = temp_dir / "test.txt"
    file_path.write_text(sample_text())
    return file_path
```

### Mock Patterns
```python
from unittest.mock import patch, Mock

# Patch external dependencies
@patch("module.get_config")
def test_function(mock_config):
    mock_config.return_value = "test-value"
    result = function_under_test()
    assert result == "expected"

# Mock HTTP calls
@patch("requests.post")
def test_api_call(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [...]}
    mock_post.return_value = mock_response
    # Test...
```

### Test Naming
- Files: `test_<module_name>.py`
- Classes: `Test<ClassName>`
- Functions: `test_<function>` or `test_<scenario>`
- Errors: `test_<function>_error` or `test_<function>_failure`
- Async: `@pytest.mark.asyncio` decorator (unused but available)

### Running Tests
```bash
# Single test: pytest tests/test_document/test_document.py::test_method
# Module tests: pytest tests/test_document/
# With coverage: pytest --cov src/ tests/
# Verbose: pytest -v tests/
```

## Architectural Principles

1. **Async-first**: Agent methods calling LLMs are `async def`
2. **Factory patterns**: Providers, loaders, splitters use factories
3. **Config-driven**: All defaults via `config/sciread.toml` or env vars
4. **Dependency injection**: Pass dependencies, don't instantiate internally
5. **Logging everywhere**: Use `get_logger(__name__)` in every module
6. **Type safety**: All public APIs have type hints
7. **Error boundaries**: Custom exceptions for domain errors

## CLI Patterns

Adding a new command (in `cli.py`):
```python
# 1. Define parser
new_parser = subparsers.add_parser("command", help="...")
new_parser.add_argument("document_file")
new_parser.add_argument("--model", default="deepseek/deepseek-chat")

# 2. Add handler (import core function first)
elif args.command == "new_command":
    try:
        result = asyncio.run(core_function(args.document_file, args.model))
        print(f"Result: {result}")
        return 0
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
```

## Guardrails

- **Never**: Edit `tests/` unless adding tests
- **Never**: Commit without user request
- **Never**: Suppress type errors with `as any`, `@ts-ignore`
- **Always**: Run `ruff check` before committing
- **Always**: Add tests for new features
- **Always**: Follow existing async/await patterns
- **Always**: Use `get_logger(__name__)` for logging
