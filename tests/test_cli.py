"""Tests for CLI command line interface."""

import subprocess

import pytest


def test_main():
    """Test that invalid CLI commands fail with proper error codes."""
    # Test that the old CLI interface (without subcommands) now fails
    with pytest.raises(subprocess.CalledProcessError):
        # Invalid command should fail since new CLI requires subcommands
        subprocess.check_output(["sciread", "foo", "foobar"], text=True)
