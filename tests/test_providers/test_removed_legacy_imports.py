"""Tests for removed legacy provider namespaces."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "sciread.llm_provider",
        "sciread.embedding_provider",
        "sciread.rerank_provider",
    ],
)
def test_legacy_provider_namespaces_are_removed(module_name: str) -> None:
    """Old provider namespaces should not remain as compatibility shims."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
