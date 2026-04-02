"""Tests for logging configuration helpers."""

import sys
from pathlib import Path
from unittest.mock import Mock

from sciread.platform import logging as logging_module


def test_setup_logging_adds_console_and_file_handlers(monkeypatch, tmp_path: Path) -> None:
    """setup_logging should reset handlers and configure both sinks."""
    remove_mock = Mock()
    add_mock = Mock()

    monkeypatch.setattr(logging_module.logger, "remove", remove_mock)
    monkeypatch.setattr(logging_module.logger, "add", add_mock)

    log_file = tmp_path / "logs" / "sciread.log"
    logging_module.setup_logging(level="DEBUG", log_file=log_file, rotation="1 MB", retention="2 days")

    remove_mock.assert_called_once_with()
    assert add_mock.call_count == 2

    console_call = add_mock.call_args_list[0]
    assert console_call.args[0] is sys.stderr
    assert console_call.kwargs["level"] == "DEBUG"
    assert console_call.kwargs["colorize"] is True
    assert "HH:mm:ss.SSS" in console_call.kwargs["format"]

    file_call = add_mock.call_args_list[1]
    assert file_call.args[0] == log_file
    assert file_call.kwargs["rotation"] == "1 MB"
    assert file_call.kwargs["retention"] == "2 days"
    assert file_call.kwargs["compression"] == "zip"
    assert log_file.parent.exists()


def test_setup_logging_uses_custom_format_without_file(monkeypatch) -> None:
    """Custom formats should be passed through unchanged."""
    remove_mock = Mock()
    add_mock = Mock()

    monkeypatch.setattr(logging_module.logger, "remove", remove_mock)
    monkeypatch.setattr(logging_module.logger, "add", add_mock)

    logging_module.setup_logging(level="WARNING", format_string="{message}")

    remove_mock.assert_called_once_with()
    add_mock.assert_called_once()
    assert add_mock.call_args.args[0] is sys.stderr
    assert add_mock.call_args.kwargs["level"] == "WARNING"
    assert add_mock.call_args.kwargs["format"] == "{message}"


def test_get_logger_returns_global_logger_and_bound_logger(monkeypatch) -> None:
    """Named loggers should bind metadata while unnamed calls return the base logger."""
    bound_logger = object()
    bind_mock = Mock(return_value=bound_logger)

    monkeypatch.setattr(logging_module.logger, "bind", bind_mock)

    assert logging_module.get_logger() is logging_module.logger
    assert logging_module.get_logger("sciread.test") is bound_logger
    bind_mock.assert_called_once_with(name="sciread.test")


def test_logging_convenience_functions_delegate_to_logger(monkeypatch) -> None:
    """Top-level helper functions should forward messages to loguru."""
    debug_mock = Mock()
    info_mock = Mock()
    warning_mock = Mock()
    error_mock = Mock()
    critical_mock = Mock()

    monkeypatch.setattr(logging_module.logger, "debug", debug_mock)
    monkeypatch.setattr(logging_module.logger, "info", info_mock)
    monkeypatch.setattr(logging_module.logger, "warning", warning_mock)
    monkeypatch.setattr(logging_module.logger, "error", error_mock)
    monkeypatch.setattr(logging_module.logger, "critical", critical_mock)

    logging_module.debug("debug message", user="alice")
    logging_module.info("info message", user="alice")
    logging_module.warning("warning message", user="alice")
    logging_module.error("error message", user="alice")
    logging_module.critical("critical message", user="alice")

    debug_mock.assert_called_once_with("debug message", user="alice")
    info_mock.assert_called_once_with("info message", user="alice")
    warning_mock.assert_called_once_with("warning message", user="alice")
    error_mock.assert_called_once_with("error message", user="alice")
    critical_mock.assert_called_once_with("critical message", user="alice")
