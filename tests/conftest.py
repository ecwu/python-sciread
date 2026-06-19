"""Shared fixtures for layered test suites."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sciread.document.ingestion.loaders.base import LoadResult
from sciread.document.models import DocumentMetadata
from tests.fakes import FakeEmbeddingClient
from tests.fakes import FakeMineruClient
from tests.fakes import FakeRerankClient

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_fixture_path() -> Callable[[str], Path]:
    """Return a path from tests/fixtures."""

    def resolve(name: str) -> Path:
        return FIXTURE_DIR / name

    return resolve


@pytest.fixture
def sample_markdown_file(sample_fixture_path: Callable[[str], Path]) -> Path:
    """Small markdown paper fixture."""
    return sample_fixture_path("sample_paper.md")


@pytest.fixture
def sample_text_file(sample_fixture_path: Callable[[str], Path]) -> Path:
    """Small text paper fixture."""
    return sample_fixture_path("sample_paper.txt")


@pytest.fixture
def fake_embedding_client() -> FakeEmbeddingClient:
    """Deterministic embedding client fixture."""
    return FakeEmbeddingClient()


@pytest.fixture
def fake_rerank_client() -> FakeRerankClient:
    """Deterministic rerank client fixture."""
    return FakeRerankClient()


@pytest.fixture
def fake_mineru_client() -> FakeMineruClient:
    """Successful fake Mineru client."""
    return FakeMineruClient(markdown="# Converted Paper\n\n## Abstract\n\nMineru markdown text.")


@pytest.fixture
def empty_load_result(tmp_path: Path) -> LoadResult:
    """Reusable empty loader result for loader contract tests."""
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    return LoadResult(text="", metadata=DocumentMetadata(source_path=source, file_type="txt"))
