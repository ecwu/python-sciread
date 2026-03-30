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
import asyncio
import sys

import logfire

from .core import comprehensive_analysis
from .core import discussion_analysis
from .core import main
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
  discussion - Uses DiscussionAgent with multiple personality-driven agents
         for collaborative analysis through discussion and consensus-building

DEBUG LOGGING:
  Set log level to DEBUG to see detailed agent interactions, prompts,
  and outputs. Use: LOG_LEVEL=DEBUG sciread coordinate paper.pdf

EXAMPLES:
  sciread coordinate paper.pdf
  sciread coordinate paper.pdf deepseek/reasoner
  sciread simple paper.pdf
  sciread simple paper.txt
  sciread react paper.pdf
  sciread react paper.pdf "What are the main contributions?"
  sciread react paper.pdf "Custom analysis task" deepseek-chat --max-loops 6
  sciread discussion paper.pdf
  sciread discussion paper.pdf deepseek-reasoner

MODELS:
  deepseek/deepseek-chat     (default)
  deepseek/deepseek-reasoner
    volcengine/doubao-seed-2.0-code
    volcengine/glm-4.7
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
    simple_parser.add_argument(
        "document_file", help="Path to the document file to analyze (PDF or TXT)"
    )
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
    react_parser.add_argument(
        "document_file", help="Path to the document file to analyze (PDF or TXT)"
    )
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

    # Discussion mode parser
    discussion_parser = subparsers.add_parser(
        "discussion",
        help="Multi-agent discussion-based analysis",
        description="Use multiple personality-driven agents for collaborative analysis through discussion and consensus-building",
    )
    discussion_parser.add_argument(
        "document_file", help="Path to the document file to analyze (PDF or TXT)"
    )
    discussion_parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat",
        help="Model identifier for the LLM provider (default: deepseek/deepseek-chat)",
    )

    # Parse arguments (skip the script name)
    args = parser.parse_args(argv[1:])

    logger.info(f"Running sciread with command: {args.command}")

    # Handle different commands
    if args.command == "coordinate":
        logger.debug(
            f"Running coordinate mode with file: {args.pdf_file}, model: {args.model}"
        )

        try:
            result = asyncio.run(comprehensive_analysis(args.pdf_file, args.model))

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
        logger.debug(
            f"Running react mode with file: {args.document_file}, task: {task_display}, model: {args.model}, max_loops: {args.max_loops}, show_progress: {not args.no_progress}"
        )

        try:
            result = asyncio.run(
                run_react_analysis(
                    args.document_file,
                    args.task,
                    model=args.model,
                    max_loops=args.max_loops,
                    show_progress=not args.no_progress,
                )
            )
            # The final report is already printed if show_progress=True
            if args.no_progress:
                print("=" * 60)
                print("REACT ANALYSIS RESULT:")
                print("=" * 60)
                print(result.final_report)
                print()
                print("SUMMARY:")
                print(result.summary)
                print()
                print("SECTIONS COVERED:")
                print(
                    ", ".join(result.sections_covered)
                    if result.sections_covered
                    else "None"
                )
                print("=" * 60)
            return 0
        except Exception as e:
            logger.error(f"ReAct analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "simple":
        logger.debug(
            f"Running simple mode with file: {args.document_file}, model: {args.model}"
        )

        try:
            result = asyncio.run(main(args.document_file, args.model))
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

    elif args.command == "discussion":
        logger.debug(
            f"Running discussion mode with file: {args.document_file}, model: {args.model}"
        )

        try:
            result = asyncio.run(discussion_analysis(args.document_file, args.model))

            print("=" * 80)
            print("DISCUSSION-BASED ANALYSIS RESULT:")
            print("=" * 80)
            print(f"Document: {result.document_title}")
            print(f"Overall Confidence: {result.confidence_score:.2f}")
            print(f"Total Insights: {len(result.final_insights)}")
            print(f"Consensus Points: {len(result.consensus_points)}")
            print(f"Divergent Views: {len(result.divergent_views)}")

            if result.discussion_metadata:
                print("\nDiscussion Metadata:")
                for key, value in result.discussion_metadata.items():
                    if key != "error":
                        print(f"  {key.replace('_', ' ').title()}: {value}")

            print()
            print("ANALYSIS SUMMARY:")
            print(result.summary)

            if result.key_contributions:
                print()
                print("KEY CONTRIBUTIONS:")
                for i, contribution in enumerate(result.key_contributions, 1):
                    print(f"  {i}. {contribution}")

            if result.significance:
                print()
                print("SIGNIFICANCE ASSESSMENT:")
                print(result.significance)

            if result.final_insights:
                print()
                print("FINAL INSIGHTS FROM DISCUSSION:")
                for i, insight in enumerate(result.final_insights, 1):
                    print(
                        f"  {i}. From {insight.agent_id} (Confidence: {insight.confidence:.2f}, Importance: {insight.importance_score:.2f})"
                    )
                    # Print full content with proper indentation for multi-line text
                    content_lines = insight.content.split("\n")
                    for line in content_lines:
                        print(f"     {line}")
                    if insight.supporting_evidence:
                        print("     Supporting Evidence:")
                        for evidence in insight.supporting_evidence:
                            print(f"       - {evidence}")
                    if insight.related_sections:
                        print(
                            f"     Related Sections: {', '.join(insight.related_sections)}"
                        )
                    if i < len(result.final_insights):
                        print()

            if result.consensus_points:
                print()
                print("CONSENSUS POINTS:")
                for i, point in enumerate(result.consensus_points, 1):
                    print(f"  {i}. {point.topic} (Strength: {point.strength:.2f})")
                    # Print full content with proper indentation for multi-line text
                    content_lines = point.content.split("\n")
                    for line in content_lines:
                        print(f"     {line}")
                    print(f"     Supporting agents: {point.supporting_agents}")
                    if i < len(result.consensus_points):
                        print()

            if result.divergent_views:
                print()
                print("DIVERGENT VIEWS:")
                for i, view in enumerate(result.divergent_views, 1):
                    print(f"  {i}. {view.topic} (Held by: {view.holding_agent})")
                    # Print full content with proper indentation for multi-line text
                    content_lines = view.content.split("\n")
                    for line in content_lines:
                        print(f"     {line}")
                    if i < len(result.divergent_views):
                        print()

            print("=" * 80)
            return 0
        except Exception as e:
            logger.error(f"Discussion analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    else:
        parser.print_help()
        return 1
