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

import argparse
import sys

import logfire

from .core import compute
from .core import run_comprehensive_analysis
from .core import run_main
from .core import run_react_analysis
from .logging_config import logger

logfire.configure()
logfire.instrument_pydantic_ai()


def run(argv=sys.argv):
    """
    Command line interface for sciread.

    Args:
        argv (list): List of arguments

    Returns:
        int: A return code
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        prog="sciread",
        description="Academic Paper Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  simple - Uses a single SimpleAgent for basic analysis
  coordinate - Uses CoordinateAgent with multiple expert agents for comprehensive analysis
         (metadata, methodology, experiments, future directions)
  react  - Uses ReAct agent for intelligent iterative analysis
         with reasoning and acting pattern

DEBUG LOGGING:
  Set log level to DEBUG to see detailed agent interactions, prompts,
  and outputs. Use: LOG_LEVEL=DEBUG python -msciread coordinate paper.pdf

EXAMPLES:
  python -msciread coordinate paper.pdf
  python -msciread coordinate paper.pdf deepseek/reasoner
  python -msciread simple paper.pdf
  python -msciread simple paper.txt
  python -msciread react paper.pdf
  python -msciread react paper.pdf "What are the main contributions?"
  python -msciread react paper.pdf "Custom analysis task" deepseek-chat --max-loops 6

MODELS:
  deepseek/deepseek-chat     (default)
  deepseek/deepseek-reasoner
  glm-4, glm-4.5
  ollama/qwen3:4b
        """,
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simple mode parser
    simple_parser = subparsers.add_parser(
        "simple",
        help="Single agent analysis",
        description="Use a single SimpleAgent for basic analysis",
    )
    simple_parser.add_argument("document_file", help="Path to the document file to analyze (PDF or TXT)")
    simple_parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )

    # Coordinate mode parser
    coordinate_parser = subparsers.add_parser(
        "coordinate",
        help="Multi-agent comprehensive analysis",
        description="Use multiple expert agents for comprehensive analysis",
    )
    coordinate_parser.add_argument("pdf_file", help="Path to the PDF file to analyze")
    coordinate_parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )

    # ReAct mode parser
    react_parser = subparsers.add_parser(
        "react",
        help="ReAct agent iterative analysis",
        description="Use ReAct agent for intelligent iterative analysis with reasoning and acting pattern",
    )
    react_parser.add_argument("document_file", help="Path to the document file to analyze (PDF or TXT)")
    react_parser.add_argument(
        "task",
        nargs="?",
        default="Analyze this academic paper focusing on: 1) What are the research questions and objectives? 2) What methodology and approach did the researchers use? 3) What are the key findings and results? 4) What are the main contributions and significance of this work?",
        help="Analysis task or question about the document (default: comprehensive academic analysis)",
    )
    react_parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )
    react_parser.add_argument(
        "--max-loops",
        type=int,
        default=8,
        metavar="N",
        help="Maximum number of analysis iterations (default: 8)",
    )
    react_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress display during analysis",
    )

    # Parse arguments (skip the script name)
    args = parser.parse_args(argv[1:])

    logger.info(f"Running sciread with command: {args.command}")

    # Handle different commands
    if args.command == "coordinate":
        logger.info(f"Running coordinate mode with file: {args.pdf_file}, model: {args.model}")

        try:
            result = run_comprehensive_analysis(args.pdf_file, args.model)

            print("=" * 60)
            print("COMPREHENSIVE ANALYSIS RESULT:")
            print("=" * 60)
            print(f"Analysis Plan: {result.analysis_plan.reasoning}")
            print(f"Total Execution Time: {result.total_execution_time:.2f} seconds")
            print(f"Agents Executed: {result.execution_summary['total_agents_executed']}")
            print(f"Successful Agents: {result.execution_summary['successful_agents']}")
            print(f"Failed Agents: {result.execution_summary['failed_agents']}")

            if hasattr(result, "interaction_log"):
                print(f"Interactions Logged: {len(result.interaction_log)}")

            print()
            print("FINAL REPORT:")
            print(result.final_report)

            print("=" * 60)
            return 0
        except Exception as e:
            logger.error(f"Coordinate analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "react":
        task_display = args.task[:100] + "..." if len(args.task) > 100 else args.task
        logger.info(
            f"Running react mode with file: {args.document_file}, task: {task_display}, model: {args.model}, max_loops: {args.max_loops}, show_progress: {not args.no_progress}"
        )

        try:
            result = run_react_analysis(
                args.document_file, args.task, model=args.model, max_loops=args.max_loops, show_progress=not args.no_progress
            )
            # The final report is already printed if show_progress=True
            if args.no_progress:
                print("=" * 60)
                print("REACT ANALYSIS RESULT:")
                print("=" * 60)
                print(result)
                print("=" * 60)
            return 0
        except Exception as e:
            logger.error(f"ReAct analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "simple":
        logger.info(f"Running simple mode with file: {args.document_file}, model: {args.model}")

        try:
            result = run_main(args.document_file, args.model)
            print("=" * 50)
            print("ANALYSIS RESULT:")
            print("=" * 50)
            print(result)
            print("=" * 50)
            return 0
        except Exception as e:
            logger.error(f"Simple analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    else:
        # Default compute behavior (no command specified)
        result = compute(argv[1:])
        logger.info(f"Compute result: {result}")
        print(result)
        return 0
