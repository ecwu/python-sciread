========
sciread
========

Understand academic papers with LLM-driven agents.

* Free software: MIT license

Features
========

- ``simple``: single-agent paper explanation and summarization
- ``coordinate``: multi-agent comprehensive analysis across metadata, methods, experiments, and future work
- ``react``: iterative ReAct-style document reasoning
- ``search-react``: retrieval-driven ReAct analysis with lexical, semantic, tree, and hybrid retrievers
- ``discussion``: multi-agent discussion with personality-driven collaboration
- Structured document ingestion, section splitting, and retrieval utilities

Installation
============

Install with ``uv``::

    uv add sciread

Or with ``pip``::

    pip install sciread

Install the development version::

    uv add git+https://github.com/ecwu/python-sciread.git@main

Quick Start
===========

CLI usage::

    uv run sciread simple paper.pdf
    uv run sciread coordinate paper.pdf
    uv run sciread react paper.pdf "What are the main contributions?"
    uv run sciread search-react paper.pdf "What are the main contributions?"
    uv run sciread discussion paper.pdf

Specify a model explicitly when needed::

    uv run sciread coordinate paper.pdf --model deepseek/deepseek-v4-flash
    uv run sciread discussion paper.pdf --model deepseek/deepseek-v4-flash
    uv run sciread search-react paper.pdf "What changed?" --retriever hybrid --top-k 6
    uv run sciread search-react paper.pdf "What changed?" --compare lexical,semantic,tree,hybrid

You can also run the package directly::

    uv run python -m sciread --help

Configuration
=============

Project settings are managed via ``pydantic-settings`` and can be defined in a TOML file or environment variables.

**Configuration File**

The default configuration file is looked for in several standard locations:

1. ``./config/sciread.toml``
2. ``./sciread.toml``
3. ``~/.config/sciread/config.toml``
4. ``~/.sciread.toml``

You should copy the example in ``config/sciread.toml`` to one of these locations and adjust it for your environment.

**Environment Variables**

You can also configure the system using environment variables (standard names for API keys, or prefixed with ``SCIREAD_`` for project-specific settings).

Create a ``.env`` file based on the provided ``.env.example``:

.. code-block:: bash

    cp .env.example .env

Key Environment Variables:

- ``DEEPSEEK_API_KEY``: API key for DeepSeek (used by default).
- ``VOLCES_API``: API key for Volcengine / Doubao (Ark).
- ``SILICONFLOW_API_KEY``: API key for SiliconFlow embeddings and reranking, used by the default RAG embedding/rerank models.
- ``MINERU_TOKEN``: Required for converting PDF files to high-quality Markdown via the Mineru API.
- ``SCIREAD_LOG_LEVEL``: Control logging verbosity (DEBUG, INFO, WARNING, ERROR).

**Mineru PDF Conversion**

By default, ``sciread`` uses a simple PDF loader. For much better results (with extracted tables and formulas), set ``MINERU_TOKEN`` and configuration in ``sciread.toml`` to use the Mineru provider.

**RAG Embeddings and Reranking**

RAG uses SiliconFlow ``BAAI/bge-m3`` embeddings by default. Set ``SILICONFLOW_API_KEY`` before building vector indices. You can still use local models explicitly with LM Studio's OpenAI-compatible local server on ``http://localhost:1234/v1`` or with Ollama. For second-stage ranking, use ``Document.retrieve_chunks(..., strategy="rerank")`` or ``Document.rerank_search(...)``; this reranks semantic candidates with the configured ``providers.rerank.default.model``.

Python API
==========

Canonical public modules:

- ``sciread.agent``: agent systems and agent-specific models
- ``sciread.document``: document-facing public API
- ``sciread.platform``: configuration and logging
- ``sciread.application``: use-case orchestration

Example::

    from sciread.agent import SimpleAgent
    from sciread.document import Document

    document = Document.from_file("paper.pdf", to_markdown=False, auto_split=True)
    agent = SimpleAgent("deepseek/deepseek-v4-flash")
    report = await agent.run_analysis(document, "Explain the paper clearly for an engineer.")

You can enable overlap-aware chunking for RAG-oriented retrieval without changing the default behavior::

    document = Document.from_text(markdown_text, is_markdown=True, chunk_overlap=150)

The same overlap behavior can be configured globally in ``config/sciread.toml`` under
``[document_splitters.markdown]`` or ``[document_splitters.semantic]``. These are the supported built-in splitter
configuration sections.

For retrieval-oriented workflows, ``search-react`` can compare multiple retrievers in one run while reusing the same
document chunking and vector-store configuration. Agent-facing retrieval should use ``EvidenceRetriever`` from
``sciread.document.retrieval`` so callers receive citation-ready evidence blocks with section labels, citation keys,
scores, and optional same-section neighbor context. ``Document.semantic_search()``, ``Document.rerank_search()``, and ``Document.retrieve_chunks()``
remain lower-level APIs for chunk/vector retrieval internals.

Project Layout
==============

The package is organized by subsystem:

- ``src/sciread/agent``: agent implementations, prompts, and agent-local models
- ``src/sciread/document``: canonical document subsystem root and public entrypoints
- ``src/sciread/document/ingestion``: loaders, external document clients, and ingestion cache helpers
- ``src/sciread/document/structure``: chunking, section resolution, rendering, and splitters
- ``src/sciread/document/retrieval``: vector indexing, semantic search, and evidence retrieval
- ``src/sciread/platform``: config and logging
- ``src/sciread/application``: top-level analysis workflows
- ``src/sciread/entrypoints``: CLI entrypoints

Documentation
=============

https://python-sciread.readthedocs.io/

Development
===========

This project uses `uv <https://docs.astral.sh/uv/>`_ for dependency management.

Set up the local environment::

    uv sync --group test --group dev

Run the default stable test suite. This excludes tests that require external services or intentionally slow checks::

    uv run pytest -m "not live and not slow" tests/

Run a focused test file::

    uv run pytest tests/test_cli.py
    uv run pytest tests/test_document/test_document.py

Run layered integration, system, and contract tests explicitly::

    uv run pytest -m "integration or system or contracts" tests/

Run optional live provider tests only when credentials and services are configured::

    SCIREAD_RUN_LIVE=1 uv run pytest -m live tests/

Run tests with coverage::

    uv run pytest -m "not live and not slow" --cov=sciread --cov-fail-under=84 tests/

Run linting::

    uv run ruff check src/ tests/

Run formatting::

    uv run ruff format src/ tests/

If you change documentation and want to build it locally::

    uv pip install -r docs/requirements.txt
    uv run sphinx-build -b html docs docs/_build/html
