# CLAUDE.md

Quick orientation for working in this repo.

## Project Snapshot
- `sciread` is a Python toolkit for reading academic papers with LLM-driven agents.
- Core flows: document ingestion/splitting, provider-backed LLM/embedding calls, agent orchestration (simple/coordinate/react/rag_react/discussion).
- Async-first codebase; factories wire up pluggable providers.

## Repo Map (edit with care)
- `src/sciread/core.py` – async entry points used by CLI modes.
- `src/sciread/cli.py` – argparse CLI wrapper exposing `simple`, `coordinate`, `react`, `rag_react`, `discussion`.
- Agents: `src/sciread/agent/` (`simple_agent.py`, `react_agent.py`, `rag_react_agent.py`, `coordinate_agent.py`, `discussion_agent.py`, `consensus_builder.py`, `task_queue.py`, `personality_agents.py`, prompts/ and tools/ helpers).
- Documents: `src/sciread/document/` (`document.py`, `document_builder.py`, `vector_index.py`, `splitters/`, `loaders/`, `external_clients.py`, `factory.py`, `mineru_cache.py`, `models.py`).
- Providers: `src/sciread/llm_provider/` (deepseek, zhipu, ollama + `factory.py`); `src/sciread/embedding_provider/` (siliconflow, ollama + `factory.py`).
- Config and docs: `config/sciread.toml`, `README.rst`, `docs/`.
- Tests mirror `src/` under `tests/`.

## Guardrails
- Do not edit tests in `tests/` unless asked; avoid schema/migration changes and any security-sensitive logic.
- Keep async patterns, factory-based provider wiring, and existing error-handling intact.
- Ask before adding dependencies or making public API changes in core modules or CLI behavior.
- Use Context7 MCP tools whenever you need library/API docs, code generation scaffolds, or setup/config steps; resolve library ids and pull docs rather than guessing.

## Setup & Tooling
- Install: `pip install -e ".[test]"`.
- Lint/format: `ruff check src/ tests/`, `ruff format src/ tests/`.
- Tests: `tox` for full matrix; `pytest tests/test_core.py` or `pytest --cov src/` for targeted runs.
- Docs: `tox -e docs && open dist/docs/index.html`.

## Config & Env
- Main config: `config/sciread.toml` (providers, splitter defaults, API keys).
- Useful envs: `LOG_LEVEL=DEBUG` to surface agent interactions; provider keys from env or config.

## CLI Usage (from project root)
- `python -msciread coordinate paper.pdf [--model deepseek/deepseek-chat]`
- `python -msciread simple paper.pdf`
- `python -msciread react paper.pdf "task" [--max-loops 8]`
- `python -msciread rag_react paper.pdf "task" [--max-loops 8]`
- `python -msciread discussion paper.pdf`

## Debug Pointers
- For chunking/sections: inspect `Document.get_section_names()` output (logged in `core.py` flows).
- Provider issues: confirm model ids resolved via `llm_provider/factory.py` or `embedding_provider/factory.py`.
- Agent behavior: check prompts under `src/sciread/agent/prompts/` and coordination logic in `coordinate_agent.py`.
