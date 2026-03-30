"""
Logging configuration for sciread package using loguru.

This module provides centralized logging configuration with support for
different log levels and output formats.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str | None = None,
) -> None:
    """
    Configure loguru logging for the sciread package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
        rotation: Log file rotation size/interval
        retention: Log file retention period
        format_string: Custom format string for log messages

    Returns:
        None
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<level>{message}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            level=level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )


def get_logger(name: str | None = None) -> logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    if name is None:
        return logger

    # Create a bound logger with the specified name
    return logger.bind(name=name)


# Convenience functions for common logging patterns
def debug(message: str, **kwargs) -> None:
    """Log debug message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs) -> None:
    """Log info message."""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs) -> None:
    """Log warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs) -> None:
    """Log error message."""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs) -> None:
    """Log critical message."""
    logger.critical(message, **kwargs)


# Configure default logging on import
setup_logging()
