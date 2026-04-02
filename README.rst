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
    uv run sciread discussion paper.pdf

Specify a model explicitly when needed::

    uv run sciread coordinate paper.pdf --model deepseek/deepseek-chat
    uv run sciread discussion paper.pdf --model deepseek/deepseek-chat

You can also run the package directly::

    uv run python -m sciread --help

Configuration
=============

Project and local settings are defined in ``config/sciread.toml``.

Typical local setup:

- copy the example config to ``~/.config/sciread/config.toml`` and adjust it for your environment
- or set provider credentials with environment variables such as ``DEEPSEEK_API_KEY`` and ``MINERU_TOKEN``
- choose a model explicitly with ``--model`` when testing different providers

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
    agent = SimpleAgent("deepseek/deepseek-chat")
    report = await agent.run_analysis(document, "Explain the paper clearly for an engineer.")

Project Layout
==============

The package is organized by subsystem:

- ``src/sciread/agent``: agent implementations, prompts, and agent-local models
- ``src/sciread/document``: canonical document subsystem root and public entrypoints
- ``src/sciread/document/ingestion``: loaders, external document clients, and ingestion cache helpers
- ``src/sciread/document/structure``: chunking, section resolution, rendering, persistence, and splitters
- ``src/sciread/document/retrieval``: vector indexing and semantic search
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

Run tests::

    uv run pytest tests/

Run a focused test file::

    uv run pytest tests/test_cli.py
    uv run pytest tests/test_document/test_document.py

Run tests with coverage::

    uv run pytest --cov src/ tests/

Run linting::

    uv run ruff check src/ tests/

Run formatting::

    uv run ruff format src/ tests/

If you change documentation and want to build it locally::

    uv pip install -r docs/requirements.txt
    uv run sphinx-build -b html docs docs/_build/html
