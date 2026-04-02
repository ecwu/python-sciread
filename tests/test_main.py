"""Tests for the package __main__ entrypoint."""

import runpy


def test_python_m_sciread_delegates_to_cli_run(monkeypatch) -> None:
    """Running the package as a module should invoke the CLI entrypoint."""
    captured: dict[str, bool] = {"called": False}

    def fake_run() -> None:
        captured["called"] = True

    monkeypatch.setattr("sciread.entrypoints.cli.run", fake_run)

    runpy.run_module("sciread", run_name="__main__")

    assert captured["called"] is True
