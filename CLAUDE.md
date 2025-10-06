# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package called `sciread` designed to understand papers with LLM-driven agents. It provides comprehensive document processing capabilities including loading academic papers from various formats, intelligent text splitting, and chunk-based processing for LLM analysis.

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
├── cli.py          # Command-line interface entry point
├── config.py       # Configuration management for API keys and provider settings
├── core.py         # Core functionality (compute function)
├── document/       # Document processing module
│   ├── __init__.py # Document module exports
│   ├── document.py # Main Document class for managing document lifecycle
│   ├── models.py   # Core data models (Chunk, DocumentMetadata, etc.)
│   ├── loaders/    # Document loaders for different file formats
│   │   ├── __init__.py
│   │   ├── base.py      # Base loader interface
│   │   ├── pdf_loader.py # PDF file loader
│   │   └── txt_loader.py # Text file loader
│   └── splitters/   # Text splitters for chunking documents
│       ├── __init__.py
│       ├── base.py         # Base splitter interface
│       ├── fixed_size.py   # Fixed-size text splitter
│       ├── rule_based.py   # Academic paper rule-based splitter
│       └── hybrid.py       # Hybrid splitter combining multiple approaches
├── llm_provider/   # LLM provider module with factory pattern
│   ├── __init__.py  # Main interface exports get_model() function
│   ├── factory.py   # ModelFactory for creating model instances
│   ├── deepseek.py  # DeepSeek provider implementation
│   ├── zhipu.py     # Zhipu GLM provider implementation
│   └── ollama.py    # Ollama local provider implementation
└── __main__.py     # Module execution entry point
```

### Key Components
- **Core Function**: `compute()` in `src/sciread/core.py` - currently returns the longest string from input arguments
- **CLI Entry**: `run()` in `src/sciread/cli.py` - handles command-line execution
- **Package Interface**: `__init__.py` exports the `compute` function as the main API
- **Configuration**: `config.py` manages API keys and provider settings via TOML configuration
- **Document Module**: `document/` provides comprehensive document processing capabilities
- **LLM Provider Module**: `llm_provider/` provides unified interface for multiple LLM providers

#### Document Processing System
The `document` module provides a complete pipeline for processing academic papers:

**Main Document Class**: `Document` in `src/sciread/document/document.py`
- Factory methods: `Document.from_file(path)` and `Document.from_text(text)`
- Lifecycle management: `load()` → `split()` → processing
- State tracking: loading status, splitting status, processing history
- Chunk management: get chunks, filter by type/processing status

**Core Data Models** in `src/sciread/document/models.py`:
- **Chunk**: Text chunk with metadata (content, type, position, confidence, processing status)
- **DocumentMetadata**: Document metadata (file info, timestamps, title, author, page count)
- **ProcessingState**: Processing lifecycle tracking (timestamps, notes, version)
- **CoverageStats**: Coverage statistics for processed chunks and words

**Document Loaders** in `src/sciread/document/loaders/`:
- **BaseLoader**: Abstract interface with common functionality
- **PdfLoader**: PDF loading with PyPDF2 and pdfplumber fallbacks
- **TxtLoader**: Text file loading with encoding detection
- LoadResult: Standardized result format with text, metadata, warnings, and errors

**Text Splitters** in `src/sciread/document/splitters/`:
- **BaseSplitter**: Abstract interface for text splitting strategies
- **RegexSectionSplitter**: Advanced regex-based academic paper section detection
- **TopicFlowSplitter**: Bottom-up sentence splitter that grows segments based on semantic continuity

**Key Features**:
- Multi-format document loading (PDF, TXT, MD, RST)
- Intelligent text splitting optimized for academic papers
- Comprehensive metadata tracking and state management
- Error handling with detailed warnings and extraction statistics
- Processing pipeline with chunk-level operations
- Coverage tracking and progress monitoring

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

### Document Module Usage Examples

```python
from sciread.document import Document
from pathlib import Path

# Load and process a document
doc = Document.from_file("paper.pdf")
result = doc.load()  # Returns LoadResult with text and metadata

if result.success:
    # Split into chunks (uses TopicFlowSplitter by default)
    chunks = doc.split()

    # Get all chunks
    all_chunks = doc.get_chunks()

    # Get chunks by type
    abstract_chunks = doc.get_chunks(chunk_type="abstract")
    unprocessed_chunks = doc.get_unprocessed_chunks()

    # Check processing state
    coverage_stats = doc.get_coverage_stats()
    print(f"Processed {coverage_stats.chunk_coverage}% of chunks")
```

### Adding Custom Loaders and Splitters

The document module is designed to be extensible:

```python
from sciread.document.loaders import BaseLoader
from sciread.document.splitters import BaseSplitter

class CustomLoader(BaseLoader):
    # Implement loading logic for new format
    pass

class CustomSplitter(BaseSplitter):
    # Implement custom splitting strategy
    pass

# Use with Document class
doc = Document.from_file("custom.ext")
doc.add_loader(CustomLoader())
doc.set_splitter(CustomSplitter())
```
