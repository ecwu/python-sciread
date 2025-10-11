"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -msciread` python will execute
    ``__main__.py`` as a script. That means there will not be any
    ``sciread.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there"s no ``sciread.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import sys

from .core import compute, run_main
from .logging_config import logger


def run(argv=sys.argv):
    """
    Args:
        argv (list): List of arguments

    Returns:
        int: A return code

    Does stuff.
    """
    logger.info(f"Running sciread with arguments: {argv[1:]}")

    # Check if agent mode is requested
    if len(argv) > 1 and argv[1] == "agent":
        if len(argv) < 3:
            logger.error("Agent mode requires a txt file path")
            print("Usage: python -msciread agent <txt_file_path> [model]")
            sys.exit(1)

        txt_file_path = argv[2]
        model = argv[3] if len(argv) > 3 else "deepseek/deepseek-chat"

        logger.info(f"Running agent mode with file: {txt_file_path}, model: {model}")

        try:
            result = run_main(txt_file_path, model)
            print("=" * 50)
            print("ANALYSIS RESULT:")
            print("=" * 50)
            print(result)
            print("=" * 50)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Default compute behavior
        result = compute(argv[1:])
        logger.info(f"Compute result: {result}")
        print(result)
        sys.exit(0)
