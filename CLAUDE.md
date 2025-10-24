# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package called `sciread` designed to understand papers with LLM-driven agents. It provides comprehensive document processing capabilities including loading academic papers from various formats, intelligent text splitting, and chunk-based processing for LLM analysis.

## When to Use Context7

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

## Common Development Commands

### Testing
- Run all tests: `tox`
- Run specific test file: `pytest tests/test_core.py`
- Run tests with coverage: `pytest --cov --cov-report=term-missing tests`
- Run single test with verbose output: `pytest -v tests/test_core.py::test_name`

### Code Quality
- Run linting and formatting checks: `tox -e check`
- Format code with ruff: `ruff format src/ tests/`
- Lint code with ruff: `ruff check src/ tests/`

### Documentation
- Build documentation: `tox -e docs`
- View docs locally: Open `dist/docs/index.html` after building

### Installation
- Install in development mode: `pip install -e .`
- Install test dependencies: `pip install -e ".[test]"`

## Architecture

### Package Structure
```
src/sciread/
├── __init__.py      # Package initialization, exports main functions
├── agent/          # Agent module for LLM-driven processing
│   ├── __init__.py
│   ├── simple_agent.py   # Single SimpleAgent for basic analysis
│   ├── coordinate_agent.py # Multi-agent CoordinateAgent system for comprehensive analysis
│   ├── react_agent.py    # ReAct agent for intelligent iterative analysis
│   ├── text_utils.py     # Text processing utilities
│   └── prompts/          # Organized prompt templates for all agents
│       ├── __init__.py
│       ├── simple.py     # SimpleAgent prompts
│       ├── coordinate.py # CoordinateAgent prompts
│       └── react.py      # ReActAgent prompts
├── cli.py          # Command-line interface entry point
├── config.py       # Configuration management for API keys and provider settings
├── core.py         # Core functionality (compute function)
├── document/       # Document processing module
│   ├── __init__.py # Document module exports
│   ├── document.py # Main Document class for managing document lifecycle
│   ├── document_builder.py # DocumentBuilder and DocumentFactory classes
│   ├── external_clients.py # External API clients (Mineru, Ollama)
│   ├── mineru_cache.py # Caching for Mineru API responses
│   ├── models.py   # Core data models (Chunk, DocumentMetadata, etc.)
│   ├── loaders/    # Document loaders for different file formats
│   │   ├── __init__.py
│   │   ├── base.py      # Base loader interface
│   │   ├── pdf_loader.py # PDF file loader with Mineru support
│   │   └── txt_loader.py # Text file loader
│   └── splitters/   # Text splitters for chunking documents
│       ├── __init__.py
│       ├── base.py         # Base splitter interface
│       ├── markdown_splitter.py # Markdown-aware splitter
│       ├── regex_section_splitter.py # Regex-based academic paper splitter
│       ├── semantic_splitter.py   # Semantic splitter using embeddings
│       └── topic_flow.py   # Topic flow splitter
├── llm_provider/   # LLM provider module with factory pattern
│   ├── __init__.py  # Main interface exports get_model() function
│   ├── factory.py   # ModelFactory for creating model instances
│   ├── deepseek.py  # DeepSeek provider implementation
│   ├── zhipu.py     # Zhipu GLM provider implementation
│   └── ollama.py    # Ollama local provider implementation
├── logging_config.py # Logging configuration with loguru
└── __main__.py     # Module execution entry point
```

### Key Components
- **Core Functions**: `main()`, `comprehensive_analysis()`, `run_react_analysis()` in `src/sciread/core.py` - main async-first analysis operations
- **CLI Entry**: `run()` in `src/sciread/cli.py` - handles command-line execution with three modes (simple, coordinate, react) using direct async calls
- **Package Interface**: `__init__.py` exports core async functions and utilities as the main API
- **Configuration**: `config.py` manages API keys and provider settings via TOML configuration
- **Document Module**: `document/` provides comprehensive document processing capabilities
- **Agent Module**: `agent/` provides LLM-driven document analysis agents
- **LLM Provider Module**: `llm_provider/` provides unified interface for multiple LLM providers

#### Agent System
The `agent` module provides a complete LLM-driven analysis system:

**Multi-Agent CoordinateAgent**: `CoordinateAgent` in `src/sciread/agent/coordinate_agent.py`
- Coordinates multiple expert sub-agents for comprehensive analysis
- Expert agents: metadata extraction, methodology analysis, experiments evaluation, etc.
- Intelligent analysis planning based on document structure
- Comprehensive report synthesis from sub-agent results
- Built-in debug logging with detailed interaction traces

**Single SimpleAgent**: `SimpleAgent` in `src/sciread/agent/simple_agent.py`
- Simple, single-agent analysis for basic document processing
- Configurable analysis tasks and prompts
- Direct LLM interaction for straightforward analysis needs

**ReAct Agent**: `ReActAgent` in `src/sciread/agent/react_agent.py`
- Reasoning and Acting pattern for intelligent iterative analysis
- Dynamic analysis strategy adaptation
- Custom task execution with reasoning steps
- Progress tracking and loop control
- Unified implementation with integrated Pydantic models

**Prompt Organization**: All prompts are organized in `src/sciread/agent/prompts/`
- `prompts/simple.py` - SimpleAgent system prompts and templates
- `prompts/coordinate.py` - CoordinateAgent expert agent prompts and planning templates
- `prompts/react.py` - ReActAgent system prompts and formatting functions
- Centralized prompt management for easy maintenance and updates

**Debug Logging**: All agents support detailed debug logging
- Enable with `LOG_LEVEL=DEBUG` environment variable
- Shows detailed prompts, outputs, and agent interactions
- Example: `LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf`

#### Document Processing System
The `document` module provides a complete pipeline for processing academic papers:

**Main Document Class**: `Document` in `src/sciread/document/document.py`
- Factory methods: `Document.from_file(path, to_markdown=False)` and `Document.from_text(text)`
- Unified chunk access with flexible filtering via `get_chunks()`
- State tracking: loading status, splitting status, processing history
- Markdown conversion support for PDFs via Mineru API
- Comprehensive chunk management operations

**Document Creation and Management**: `DocumentBuilder` and `DocumentFactory` in `src/sciread/document/document_builder.py`
- **DocumentBuilder**: Builder pattern for custom document processing pipelines
- **DocumentFactory**: Static factory methods for creating documents from files or text
- Support for external clients (Mineru, Ollama) and custom processing components

**Core Data Models** in `src/sciread/document/models.py`:
- **Chunk**: Text chunk with metadata (content, type, position, confidence, processing status)
- **DocumentMetadata**: Document metadata (file info, timestamps, title, author, page count)
- **ProcessingState**: Processing lifecycle tracking (timestamps, notes, version)

**External API Clients** in `src/sciread/document/external_clients.py`:
- **MineruClient**: Client for PDF-to-markdown conversion via Mineru API
- **OllamaClient**: Client for embedding operations using Ollama models
- Caching support for API responses (`mineru_cache.py`)

**Document Loaders** in `src/sciread/document/loaders/`:
- **BaseLoader**: Abstract interface with common functionality
- **PdfLoader**: PDF loading with Mineru markdown conversion and fallback extraction methods
- **TxtLoader**: Text file loading with encoding detection
- LoadResult: Standardized result format with text, metadata, warnings, and errors

**Text Splitters** in `src/sciread/document/splitters/`:
- **BaseSplitter**: Abstract interface for text splitting strategies
- **MarkdownSplitter**: Markdown-aware splitter for structured content
- **RegexSectionSplitter**: Advanced regex-based academic paper section detection
- **SemanticSplitter**: Semantic splitter using embeddings for intelligent chunking
- **TopicFlowSplitter**: Bottom-up sentence splitter that grows segments based on semantic continuity

**Key Features**:
- Multi-format document loading (PDF, TXT) with markdown conversion support
- Intelligent text splitting optimized for academic papers with multiple strategies
- Comprehensive metadata tracking and state management
- External API integration for enhanced processing capabilities
- Error handling with detailed warnings and extraction statistics
- Processing pipeline with chunk-level operations
- Coverage tracking and progress monitoring
- Flexible chunk filtering and management system

#### LLM Provider System
The `llm_provider` module implements a factory pattern for working with different LLM providers using pydantic-ai:

**Main Interface**: `get_model(model_identifier: str) -> Model`
- Explicit provider: `get_model("deepseek/deepseek-chat")`
- Default provider: `get_model("deepseek-chat")` (uses deepseek provider)

**Supported Providers**:
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner` (uses OpenAI-compatible API)
- **Zhipu GLM**: `glm-4.6`, `glm-4.5` (uses Anthropic-compatible API)
- **Ollama**: Local models like `qwen3:4b`, `llama3:8b` (localhost:11434)

**Key Features**:
- Factory pattern for model creation (`src/sciread/llm_provider/factory.py:89`)
- Automatic provider detection for known models
- Configuration-driven API key management
- Model validation and error handling
- Support for custom base URLs and provider settings

### Configuration
- **pyproject.toml**: Modern Python packaging configuration with ruff linting/formatting rules
- **tox.ini**: Multi-environment testing setup (Python 3.9-3.13, PyPy)
- **pytest.ini**: Test configuration with doctest support and strict warnings
- **GitHub Actions**: CI/CD pipeline testing across multiple Python versions and platforms
- **config/sciread.toml**: TOML configuration file for API keys and provider settings

#### Configuration Structure
The `config/sciread.toml` file supports:
- `[default]` section for default provider configuration
- Provider-specific sections (`[deepseek]`, `[zhipu]`, `[ollama]`)
- API key management
- Base URL customization
- Model-specific settings

#### Logging System
The project uses loguru for structured logging with comprehensive configuration options:

**Logging Configuration Module**: `src/sciread/logging_config.py`
- **Default setup**: INFO level to console with colored output
- **Flexible configuration**: Support for file logging, rotation, and custom formats
- **Module-specific loggers**: Each module can get its own logger instance
- **Convenience functions**: Direct access to debug, info, warning, error, critical

**Key Features**:
- Structured logging with timestamps, levels, and source location
- Console output with colors and file logging with rotation
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Automatic setup on import with sensible defaults

**Log Level Guidelines**:

- **DEBUG**: Detailed diagnostic information for development
  - Function entry/exit with parameters
  - Internal state changes
  - Detailed processing steps
  - Use cases: Debugging complex algorithms, tracking data flow

- **INFO**: General information about application progress
  - Successful completion of major operations
  - Application startup/shutdown
  - Configuration loaded
  - File processing started/completed
  - Use cases: Monitoring application progress, user-facing status updates

- **WARNING**: Potentially problematic situations that don't prevent execution
  - Fallback to alternative methods (e.g., PDF extraction method changes)
  - Missing metadata that can be defaulted
  - Performance concerns (slow operations, large file sizes)
  - Non-critical data quality issues
  - Use cases: Alerting about recoverable issues, performance monitoring

- **ERROR**: Serious problems that prevent specific operations from completing
  - Failed file loading/parsing
  - Network/API request failures
  - Validation errors
  - Missing required dependencies
  - Use cases: Error tracking, debugging failures, user error messages

- **CRITICAL**: Severe errors that may cause application termination
  - Out of memory errors
  - Critical system failures
  - Unrecoverable data corruption
  - Use cases: System monitoring, emergency alerts

**Usage Examples**:

```python
# Basic usage (uses default configuration)
from sciread import get_logger
logger = get_logger(__name__)
logger.info("Operation completed successfully")

# Custom logging setup
from sciread import setup_logging
setup_logging(
    level="DEBUG",
    log_file="app.log",
    rotation="10 MB",
    retention="7 days"
)

# Convenience functions
from sciread.logging_config import debug, info, warning, error
debug("Detailed diagnostic information")
info("General progress information")
warning("Potential issue detected")
error("Operation failed")

# In class methods
class MyClass:
    def __init__(self):
        self.logger = get_logger(__name__)

    def process(self):
        self.logger.info("Starting processing")
        try:
            # ... processing logic
            self.logger.info("Processing completed")
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
```

**Default Configuration**:
- Console output with colors
- INFO level logging
- Format: `time | level | module:function:line | message`
- Automatic backtrace and diagnose for errors

### Code Style
- **Line Length**: 140 characters
- **Quote Style**: Double quotes
- **Import Sorting**: Single line, forced separation for conftest
- **Linting**: Uses ruff with extensive rule set (flake8, bandit, pylint, etc.)
- **Testing**: pytest with coverage reporting

## Development Notes

- The project uses setuptools with namespace packages in `src/` layout
- Tests are located in `tests/` directory and included in coverage reports
- The package is designed to be cross-platform (Windows, macOS, Linux)
- The document module provides a complete foundation for LLM-driven paper analysis
- All components follow modern Python practices with type hints and comprehensive error handling
- The architecture is designed to be extensible for new file formats and processing strategies
- The codebase uses an async-first design pattern throughout the core modules

### Document Module Usage Examples

```python
from sciread.document import Document
from pathlib import Path

# Load and process a document
doc = Document.from_file("paper.pdf", to_markdown=True)  # Convert PDF to markdown

# Get all chunks with flexible filtering
all_chunks = doc.get_chunks()
unprocessed_chunks = doc.get_chunks(processed=False)
high_quality_chunks = doc.get_chunks(confidence_threshold=0.7, min_length=100)

# Get chunks by name
abstract_chunks = doc.get_chunks(chunk_name="abstract")
introduction_chunks = doc.get_chunks(chunk_name="introduction")

# Mark chunks as processed based on quality criteria
processed_count = doc.mark_chunks_processed(confidence_threshold=0.5, min_length=50)

# Get chunks by content (alternative to search)
# Can use get_chunks() with content filtering or section-based access
matching_chunks = doc.get_sections_by_name(["abstract", "introduction"])

# Work with sections
section_names = doc.get_section_names()
sections_by_name = doc.get_sections_by_name(["abstract", "introduction"])
```

### Using DocumentBuilder for Custom Processing

```python
from sciread.document import DocumentBuilder
from sciread.document.external_clients import MineruClient, OllamaClient
from sciread.document.splitters import SemanticSplitter

# Create custom processing pipeline
builder = DocumentBuilder(
    splitter=SemanticSplitter(ollama_client=OllamaClient()),
    mineru_client=MineruClient()
)

# Build document with custom settings
doc = builder.from_file("paper.pdf", to_markdown=True, auto_split=True)
```

### Adding Custom Loaders and Splitters

The document module is designed to be extensible:

```python
from sciread.document.loaders import BaseLoader
from sciread.document.splitters import BaseSplitter

class CustomLoader(BaseLoader):
    def load(self, file_path: Path) -> LoadResult:
        # Implement loading logic for new format
        pass

class CustomSplitter(BaseSplitter):
    def split(self, text: str) -> list[Chunk]:
        # Implement custom splitting strategy
        pass

# Use with DocumentBuilder
builder = DocumentBuilder(loader=CustomLoader(), splitter=CustomSplitter())
doc = builder.from_file("custom.ext")
```

### CLI Usage Examples

The sciread package provides a comprehensive command-line interface with three analysis modes:

```bash
# Coordinate mode - Multi-agent comprehensive analysis (recommended for academic papers)
python -msciread coordinate paper.pdf
python -msciread coordinate paper.pdf deepseek/reasoner

# Simple mode - Single agent basic analysis
python -msciread simple paper.pdf
python -msciread simple paper.txt

# ReAct mode - Intelligent iterative analysis with custom tasks
python -msciread react paper.pdf
python -msciread react paper.pdf "What are the main contributions?"
python -msciread react paper.pdf "Custom analysis task" deepseek-chat --max-loops 6
```

**Debug Logging**: Enable detailed agent interactions and prompts:

```bash
# Enable debug logging to see all agent interactions
LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf

# Debug logging works with all modes
LOG_LEVEL=DEBUG python -msciread simple paper.pdf
LOG_LEVEL=DEBUG python -msciread react paper.pdf "Custom task"
```

**Available Models**:
- `deepseek/deepseek-chat` (default)
- `deepseek/deepseek-reasoner`
- `glm-4`, `glm-4.5`
- `ollama/qwen3:4b` (local models)

The CLI automatically handles document loading, text splitting, and agent coordination. Debug logging shows detailed prompts, outputs, and agent decision-making processes.
