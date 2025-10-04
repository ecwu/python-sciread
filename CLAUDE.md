# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package called `sciread` designed to understand papers with LLM-driven agents. It's a lightweight package currently in early development stages with a simple command-line interface.

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
├── __init__.py      # Package initialization, exports compute function
├── cli.py          # Command-line interface entry point
├── core.py         # Core functionality (compute function)
└── __main__.py     # Module execution entry point
```

### Key Components
- **Core Function**: `compute()` in `src/sciread/core.py` - currently returns the longest string from input arguments
- **CLI Entry**: `run()` in `src/sciread/cli.py` - handles command-line execution
- **Package Interface**: `__init__.py` exports the `compute` function as the main API

### Configuration
- **pyproject.toml**: Modern Python packaging configuration with ruff linting/formatting rules
- **tox.ini**: Multi-environment testing setup (Python 3.9-3.13, PyPy)
- **pytest.ini**: Test configuration with doctest support and strict warnings
- **GitHub Actions**: CI/CD pipeline testing across multiple Python versions and platforms

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
- Currently has minimal functionality - serves as a foundation for LLM-driven paper analysis
