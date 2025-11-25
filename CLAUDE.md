# CLAUDE.md

<!-- ========================================================================== -->
<!-- PROJECT OVERVIEW -->
<!-- ========================================================================== -->

**Project**: `sciread` - Python package for understanding academic papers using LLM-driven agents

**Core Capabilities**: Document processing, intelligent text splitting, chunk-based LLM analysis, multi-agent coordination

**Key Technologies**: pydantic-ai, ChromaDB, async/await patterns, factory pattern for providers

<!-- ========================================================================== -->
<!-- CONTEXT & BOUNDARIES -->
<!-- ========================================================================== -->

## When to Use Context7

Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

## Boundaries & Things NOT to Do

**Protected Areas** - DO NOT MODIFY without explicit permission:
- Tests in `tests/` directory - maintain test integrity
- Database migrations or schema changes
- Security-critical code and authentication mechanisms
- API contracts and interface definitions
- Configuration files with secrets or production settings

**Code Style Boundaries**:
- Follow existing patterns (don't introduce new frameworks without discussion)
- Maintain async/await patterns in core modules
- Use existing factory patterns for providers
- Preserve type hints and error handling patterns

**When to Ask**:
- Before adding new dependencies to pyproject.toml
- Before making breaking changes to public APIs
- Before modifying core agent coordination logic
- When uncertain about security implications

<!-- ========================================================================== -->
<!-- DEVELOPMENT WORKFLOW -->
<!-- ========================================================================== -->

## Quick Start Commands

```bash
# Development setup
pip install -e ".[test]"

# Testing
tox                           # All environments
pytest tests/test_core.py     # Specific file
pytest --cov src/             # With coverage

# Code quality
tox -e check                  # Lint + format
ruff format src/ tests/       # Format
ruff check src/ tests/        # Lint

# Documentation
tox -e docs && open dist/docs/index.html
```

## Key Files to Understand

**Core Architecture**:
- `src/sciread/core.py` - Main async analysis functions
- `src/sciread/cli.py` - CLI entry point with mode routing
- `src/sciread/config.py` - TOML-based configuration management

@src/sciread/core.py
@src/sciread/cli.py
@src/sciread/config.py

**Agent System**:
- `src/sciread/agent/coordinate_agent.py` - Multi-agent coordination
- `src/sciread/agent/rag_react_agent.py` - RAG+ReAct implementation
- `src/sciread/agent/discussion_agent.py` - Multi-agent discussions
- `src/sciread/agent/simple_agent.py` - Basic single-agent analysis
- `src/sciread/agent/personality_agents.py` - Personality-based expert agents
- `src/sciread/agent/consensus_builder.py` - Consensus building from discussions
- `src/sciread/agent/task_queue.py` - Task queue management system

@src/sciread/agent/react_agent.py
@src/sciread/agent/coordinate_agent.py
@src/sciread/agent/rag_react_agent.py
@src/sciread/agent/discussion_agent.py
@src/sciread/agent/simple_agent.py

**Document Processing**:
- `src/sciread/document/document.py` - Main Document class
- `src/sciread/document/vector_index.py` - ChromaDB integration
- `src/sciread/document/splitters/` - Text chunking strategies
- `src/sciread/document/document_builder.py` - Document builder pattern
- `src/sciread/document/external_clients.py` - External API clients (Mineru)
- `src/sciread/document/models.py` - Core data models

@src/sciread/document/document.py
@src/sciread/document/vector_index.py
@src/sciread/document/document_builder.py
@src/sciread/document/external_clients.py
@src/sciread/document/models.py

**Text Splitters**:
- `src/sciread/document/splitters/semantic_splitter.py` - Embedding-based chunking
- `src/sciread/document/splitters/consecutive_flow.py` - Flow-based splitting
- `src/sciread/document/splitters/cumulative_flow.py` - Cumulative flow splitting
- `src/sciread/document/splitters/markdown_splitter.py` - Markdown-aware splitting
- `src/sciread/document/splitters/regex_section_splitter.py` - Academic paper sections

@src/sciread/document/splitters/semantic_splitter.py
@src/sciread/document/splitters/consecutive_flow.py
@src/sciread/document/splitters/cumulative_flow.py
@src/sciread/document/splitters/markdown_splitter.py

**Provider Systems**:
- `src/sciread/llm_provider/factory.py` - LLM factory pattern
- `src/sciread/embedding_provider/factory.py` - Embedding factory pattern
- `src/sciread/llm_provider/deepseek.py` - DeepSeek provider
- `src/sciread/llm_provider/zhipu.py` - Zhipu provider
- `src/sciread/embedding_provider/siliconflow.py` - SiliconFlow provider

@src/sciread/llm_provider/factory.py
@src/sciread/embedding_provider/factory.py
@src/sciread/llm_provider/deepseek.py
@src/sciread/embedding_provider/siliconflow.py

<!-- ========================================================================== -->
<!-- ARCHITECTURE OVERVIEW -->
<!-- ========================================================================== -->

## Architecture Overview

**Core Design Patterns**:
- Factory pattern for LLM/embedding providers
- Async-first design throughout core modules
- Agent-based architecture with pydantic-ai
- Builder pattern for document processing pipelines

**Main Components**:

```python
# Core interfaces - main entry points
src/sciread/core.py         # Main analysis functions
src/sciread/cli.py          # CLI routing
src/sciread/config.py       # Configuration management

# Agent system - LLM-driven analysis
src/sciread/agent/
├── coordinate_agent.py    # Multi-agent coordination
├── rag_react_agent.py     # RAG+ReAct with ChromaDB
├── discussion_agent.py    # Personality-based discussions
└── prompts/               # Organized prompt templates

# Document processing - pipeline for papers
src/sciread/document/
├── document.py            # Main Document class
├── vector_index.py        # ChromaDB semantic search
├── splitters/             # Text chunking strategies
└── loaders/               # File format support

# Provider systems - pluggable integrations
src/sciread/llm_provider/      # LLM factory (DeepSeek, Zhipu, Ollama)
src/sciread/embedding_provider/ # Embedding factory (Ollama, SiliconFlow)
```

**Agent Types**:
- **SimpleAgent**: Basic single-agent analysis
- **CoordinateAgent**: Multi-agent expert coordination
- **ReActAgent**: Reasoning+Acting iterative analysis
- **RAG+ReActAgent**: Semantic search + ReAct pattern
- **DiscussionAgent**: Multi-personality consensus building

**Text Splitters**:
- **Flow-based**: `consecutive_flow.py`, `cumulative_flow.py` - sentence similarity
- **Semantic**: `semantic_splitter.py` - embedding-based chunking
- **Structure-based**: `markdown_splitter.py`, `regex_section_splitter.py`

<!-- ========================================================================== -->
<!-- CONFIGURATION & USAGE -->
<!-- ========================================================================== -->

## Configuration

**Main Config**: `config/sciread.toml` - API keys, provider settings, splitter defaults

**Environment Variables**:
- `LOG_LEVEL=DEBUG` - Enable detailed agent interaction logging
- API keys set in config file or environment

**Supported Models**:
- `deepseek-chat`, `deepseek-reasoner` (default)
- `glm-4`, `glm-4.5` (Zhipu)
- `ollama/qwen3:4b` (local models)

## CLI Usage

```bash
# Multi-agent comprehensive analysis (recommended)
python -msciread coordinate paper.pdf

# Single-agent basic analysis
python -msciread simple paper.pdf

# Intelligent iterative analysis
python -msciread react paper.pdf "Custom analysis task"

# Debug logging (shows all agent interactions)
LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf
```

<!-- ========================================================================== -->
<!-- CODE EXAMPLES -->
<!-- ========================================================================== -->

## Common Patterns

**Document Processing**:

```python
from sciread.document import Document

# Load and process document
doc = Document.from_file("paper.pdf", to_markdown=True)
chunks = doc.get_chunks(confidence_threshold=0.7)
```

**Custom Agent Usage**:

```python
from sciread.agent import CoordinateAgent
from sciread.llm_provider import get_model

agent = CoordinateAgent(model=get_model("deepseek-chat"))
result = await agent.analyze_document(doc)
```

**Provider Factory**:

```python
from sciread.embedding_provider import get_embedding_client
from sciread.llm_provider import get_model

# Automatic provider detection
llm = get_model("deepseek-chat")
embeddings = get_embedding_client("siliconflow/Qwen/Qwen3-Embedding-8B")
```

<!-- ========================================================================== -->
<!-- TESTING & DEBUGGING -->
<!-- ========================================================================== -->

## Testing Strategy

**Test Structure**: `tests/` mirrors `src/sciread/` structure
- Unit tests for individual components
- Integration tests for agent workflows
- Provider tests for external APIs

**Debug Commands**:
```bash
# Agent interaction debugging
LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf

# Specific test debugging
pytest -v -s tests/test_agent.py::test_coordinate_agent
```

**Common Debug Points**:
- Agent prompt/response flow (use LOG_LEVEL=DEBUG)
- Document chunking quality (check chunk confidence scores)
- Provider API connectivity (verify config settings)