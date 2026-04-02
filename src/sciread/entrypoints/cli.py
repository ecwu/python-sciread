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
from rich.console import Console

from ..agent.coordinate import ComprehensiveAnalysisResult
from ..application import run_coordinate_analysis
from ..application import run_discussion_analysis
from ..application import run_react_analysis
from ..application import run_simple_analysis
from ..platform.logging import logger
from ..platform.rich_output import TableColumnSpec
from ..platform.rich_output import build_data_table
from ..platform.rich_output import build_key_value_table
from ..platform.rich_output import build_markdown_panel
from ..platform.rich_output import build_mode_banner
from ..platform.rich_output import build_sections_table

logfire.configure()
logfire.instrument_pydantic_ai()
console = Console()


COORDINATE_PLAN_SPECS = (
    ("metadata", "Metadata", "analyze_metadata", None),
    (
        "previous_methods",
        "Previous Methods",
        "analyze_previous_methods",
        "previous_methods_sections",
    ),
    (
        "research_questions",
        "Research Questions",
        "analyze_research_questions",
        "research_questions_sections",
    ),
    ("methodology", "Methodology", "analyze_methodology", "methodology_sections"),
    ("experiments", "Experiments", "analyze_experiments", "experiments_sections"),
    (
        "future_directions",
        "Future Directions",
        "analyze_future_directions",
        "future_directions_sections",
    ),
)


def _render_discussion_overview(overview: dict[str, object]) -> None:
    """Render discussion mode document overview."""
    section_names = overview["section_names"]
    if not isinstance(section_names, list):
        section_names = []

    console.print()
    console.print(
        build_mode_banner(
            "Discussion Analysis",
            subtitle="Multi-agent discussion with normalized progress output",
        )
    )
    overview_rows = [
        ("Document", str(overview["document_title"])),
        ("Total Content", f"{overview['total_content_chars']} characters"),
        ("Chunks", str(overview["chunk_count"])),
    ]

    if section_names:
        overview_rows.append(("Sections", f"{len(section_names)} main sections identified"))
    else:
        overview_rows.append(("Sections", "No named sections found (continuous text analysis)"))

    console.print(build_key_value_table("Document Overview", overview_rows))

    if section_names:
        console.print(
            build_sections_table(
                "Available Sections for Analysis",
                [str(section_name).title() for section_name in section_names[:10]],
                caption=(f"... and {len(section_names) - 10} more sections" if len(section_names) > 10 else None),
            )
        )

    console.print()


def _resolve_plan_sections(
    result: ComprehensiveAnalysisResult,
    analysis_type: str,
    plan_field: str,
    sections_field: str | None,
) -> str:
    """Return a display string for the sections assigned to one sub-agent."""
    plan = result.analysis_plan
    if not getattr(plan, plan_field):
        return "-"

    executed_sections = result.sections_analyzed.get(analysis_type)
    if executed_sections:
        return "\n".join(executed_sections)

    if sections_field is None:
        return "First 3 chunks"

    planned_sections = getattr(plan, sections_field)
    if planned_sections:
        return "\n".join(planned_sections)

    return "All sections"


def _render_coordinate_plan(result: ComprehensiveAnalysisResult, target_console: Console | None = None) -> None:
    """Render the coordinate analysis plan and section allocation."""
    active_console = target_console or console
    plan = result.analysis_plan

    active_console.print()
    active_console.print(
        build_mode_banner(
            "Coordinate Analysis Plan",
            subtitle="Planned sub-agent coverage and final synthesis",
        )
    )

    if plan.reasoning:
        active_console.print(build_markdown_panel("Planner Reasoning", plan.reasoning, border_style="blue"))

    rows: list[tuple[str, ...]] = []
    for analysis_type, label, plan_field, sections_field in COORDINATE_PLAN_SPECS:
        enabled = getattr(plan, plan_field)
        enabled_text = "Yes" if enabled else "No"
        sections = _resolve_plan_sections(result, analysis_type, plan_field, sections_field)
        rows.append((label, enabled_text, sections))

    active_console.print(
        build_data_table(
            title="Sub-Agent Section Plan",
            columns=[
                TableColumnSpec("Sub-Agent", style="cyan", no_wrap=True),
                TableColumnSpec("Enabled", style="green", no_wrap=True),
                TableColumnSpec("Sections to Read", style="yellow"),
            ],
            rows=rows,
        )
    )
    active_console.print()


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
        default=5,
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
    discussion_parser.add_argument("document_file", help="Path to the document file to analyze (PDF or TXT)")
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
        logger.debug(f"Running coordinate mode with file: {args.pdf_file}, model: {args.model}")

        try:
            result = asyncio.run(run_coordinate_analysis(args.pdf_file, args.model))
            _render_coordinate_plan(result)
            console.print(build_markdown_panel("Final Report", result.final_report, border_style="green"))
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
            return 0
        except Exception as e:
            logger.error(f"ReAct analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "simple":
        logger.debug(f"Running simple mode with file: {args.document_file}, model: {args.model}")

        try:
            result = asyncio.run(run_simple_analysis(args.document_file, args.model))
            return 0
        except Exception as e:
            logger.error(f"Simple analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    elif args.command == "discussion":
        logger.debug(f"Running discussion mode with file: {args.document_file}, model: {args.model}")

        try:
            overview, result = asyncio.run(run_discussion_analysis(args.document_file, args.model))
            _render_discussion_overview(overview)

            markdown_lines = [
                "# Discussion-Based Analysis Result",
                "",
                "## Overview",
                f"- **Document:** {result.document_title}",
                f"- **Overall Confidence:** {result.confidence_score:.2f}",
                f"- **Total Insights:** {len(result.final_insights)}",
                f"- **Consensus Points:** {len(result.consensus_points)}",
                f"- **Divergent Views:** {len(result.divergent_views)}",
            ]

            if result.discussion_metadata:
                markdown_lines.extend(["", "## Discussion Metadata"])
                for key, value in result.discussion_metadata.items():
                    if key != "error":
                        markdown_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

            markdown_lines.extend(["", "## Analysis Summary", result.summary])

            if result.key_contributions:
                markdown_lines.extend(["", "## Key Contributions"])
                for contribution in result.key_contributions:
                    markdown_lines.append(f"- {contribution}")

            if result.significance:
                markdown_lines.extend(["", "## Significance Assessment", result.significance])

            if result.final_insights:
                markdown_lines.extend(["", "## Final Insights From Discussion"])
                for i, insight in enumerate(result.final_insights, 1):
                    markdown_lines.extend(
                        [
                            "",
                            f"### {i}. From {insight.agent_id}",
                            f"- **Confidence:** {insight.confidence:.2f}",
                            f"- **Importance:** {insight.importance_score:.2f}",
                            "",
                            insight.content,
                        ]
                    )
                    if insight.supporting_evidence:
                        markdown_lines.append("\n**Supporting Evidence:**")
                        for evidence in insight.supporting_evidence:
                            markdown_lines.append(f"- {evidence}")
                    if insight.related_sections:
                        markdown_lines.append(f"\n**Related Sections:** {', '.join(insight.related_sections)}")

            if result.consensus_points:
                markdown_lines.extend(["", "## Consensus Points"])
                for i, point in enumerate(result.consensus_points, 1):
                    markdown_lines.extend(
                        [
                            "",
                            f"### {i}. {point.topic}",
                            f"- **Strength:** {point.strength:.2f}",
                            f"- **Supporting Agents:** {point.supporting_agents}",
                            "",
                            point.content,
                        ]
                    )

            if result.divergent_views:
                markdown_lines.extend(["", "## Divergent Views"])
                for i, view in enumerate(result.divergent_views, 1):
                    markdown_lines.extend(
                        [
                            "",
                            f"### {i}. {view.topic}",
                            f"- **Held By:** {view.holding_agent}",
                            "",
                            view.content,
                        ]
                    )

            console.print(build_markdown_panel("Final Report", "\n".join(markdown_lines), border_style="green"))
            return 0
        except Exception as e:
            logger.error(f"Discussion analysis failed: {e}")
            print(f"Error: {e}")
            return 1

    else:
        parser.print_help()
        return 1
