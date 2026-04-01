========
Overview
========

Understand paper with LLM-driven agents.

* Free software: MIT license

Installation
============

You can install ``sciread`` using ``uv``::

    uv add sciread

Or with standard ``pip``::

    pip install sciread

Development version::

    uv add git+https://gitea.ecwu.xyz/ecwu/python-sciread.git@main

Documentation
=============

https://python-sciread.readthedocs.io/

Development
===========

This project uses `uv <https://docs.astral.sh/uv/>`_ for dependency management.

To set up the development environment::

    uv sync --group test --group dev

To install the git hooks::

    uv run pre-commit install --install-hooks

To run all tests::

    uv run pytest tests/

To run tests with coverage::

    uv run pytest --cov src/ tests/

To run linting (Ruff)::

    uv run ruff check src/ tests/

To run formatting (Ruff)::

    uv run ruff format src/ tests/

``pre-commit`` runs ``uv run ruff format`` for Python files before commit.
