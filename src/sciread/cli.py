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

from .core import (
    compute,
    run_main,
    run_comprehensive_analysis,
    run_comprehensive_analysis_with_debug,
)
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
  agent  - Uses a single DocumentAgent for basic analysis
  tool   - Uses multiple expert agents for comprehensive analysis
         (metadata, methodology, experiments, future directions)

EXAMPLES:
  python -msciread tool paper.pdf
  python -msciread tool paper.pdf deepseek/reasoner
  python -msciread tool paper.pdf deepseek/reasoner --debug-output debug.json
  python -msciread agent paper.txt

MODELS:
  deepseek/deepseek-chat     (default)
  deepseek/deepseek-reasoner
  glm-4, glm-4.5
  ollama/qwen3:4b
        """,
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Agent mode parser
    agent_parser = subparsers.add_parser(
        "agent",
        help="Single agent analysis",
        description="Use a single DocumentAgent for basic analysis",
    )
    agent_parser.add_argument("txt_file", help="Path to the text file to analyze")
    agent_parser.add_argument(
        "model",
        nargs="?",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )

    # Tool mode parser
    tool_parser = subparsers.add_parser(
        "tool",
        help="Multi-agent comprehensive analysis",
        description="Use multiple expert agents for comprehensive analysis",
    )
    tool_parser.add_argument("pdf_file", help="Path to the PDF file to analyze")
    tool_parser.add_argument(
        "model",
        nargs="?",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )
    tool_parser.add_argument(
        "--debug-output",
        metavar="FILE",
        help="Save interaction log to the specified JSON file for debugging",
    )

    # Parse arguments (skip the script name)
    args = parser.parse_args(argv[1:])

    logger.info(f"Running sciread with command: {args.command}")

    # Handle different commands
    if args.command == "tool":
        logger.info(
            f"Running tool mode with file: {args.pdf_file}, model: {args.model}, debug_output: {args.debug_output}"
        )

        try:
            # Use debug version if debug output is requested
            if args.debug_output:
                result = run_comprehensive_analysis_with_debug(
                    args.pdf_file, args.model
                )
                # Save the interaction log
                from .agent import ToolAgent

                tool_agent = ToolAgent(args.model)
                tool_agent.interaction_log = result.interaction_log
                tool_agent.save_interaction_log(args.debug_output)
                print(f"Debug interaction log saved to: {args.debug_output}")
                print(f"Total interactions logged: {len(result.interaction_log)}")
            else:
                result = run_comprehensive_analysis(args.pdf_file, args.model)

            print("=" * 60)
            print("COMPREHENSIVE ANALYSIS RESULT:")
            print("=" * 60)
            print(f"Analysis Plan: {result.analysis_plan.reasoning}")
            print(f"Total Execution Time: {result.total_execution_time:.2f} seconds")
            print(
                f"Agents Executed: {result.execution_summary['total_agents_executed']}"
            )
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
            logger.error(f"Tool analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "agent":
        logger.info(
            f"Running agent mode with file: {args.txt_file}, model: {args.model}"
        )

        try:
            result = run_main(args.txt_file, args.model)
            print("=" * 50)
            print("ANALYSIS RESULT:")
            print("=" * 50)
            print(result)
            print("=" * 50)
            return 0
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    else:
        # Default compute behavior (no command specified)
        result = compute(argv[1:])
        logger.info(f"Compute result: {result}")
        print(result)
        return 0
