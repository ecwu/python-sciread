"""Shared rich-rendering helpers for CLI and agent progress output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(slots=True)
class TableColumnSpec:
    """Declarative table-column configuration for shared rich tables."""

    header: str
    style: str = "white"
    justify: str = "left"
    no_wrap: bool = False
    ratio: int | None = None
    width: int | None = None
    max_width: int | None = None
    overflow: str = "fold"


def build_mode_banner(title: str, subtitle: str | None = None, border_style: str = "cyan") -> Panel:
    """Build a consistent banner for one analysis mode."""
    lines: list[Text] = [Text(title, style="bold white")]
    if subtitle:
        lines.append(Text(subtitle, style="dim"))

    return Panel(
        Group(*lines),
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def build_stage_banner(title: str, summary_lines: list[str] | None = None, border_style: str = "blue") -> Panel:
    """Build a compact panel for phase or iteration progress."""
    lines: list[Text] = [Text(title, style="bold white")]
    for line in summary_lines or []:
        lines.append(Text(line, style="dim"))

    return Panel(
        Group(*lines),
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def build_data_table(
    title: str,
    columns: list[TableColumnSpec],
    rows: list[tuple[str, ...]] | list[list[str]],
    caption: str | None = None,
    show_lines: bool = True,
    expand: bool = True,
    show_header: bool = True,
) -> Table:
    """Build a table with consistent visual styling."""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_lines=show_lines,
        expand=expand,
        show_header=show_header,
        header_style="bold cyan",
    )

    for column in columns:
        table.add_column(
            column.header,
            style=column.style,
            justify=column.justify,
            no_wrap=column.no_wrap,
            ratio=column.ratio,
            width=column.width,
            max_width=column.max_width,
            overflow=column.overflow,
        )

    for row in rows:
        table.add_row(*[str(value) for value in row])

    if caption:
        table.caption = caption

    return table


def build_key_value_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """Build a two-column summary table."""
    return build_data_table(
        title=title,
        columns=[
            TableColumnSpec("Field", style="cyan", no_wrap=True),
            TableColumnSpec("Value", style="white"),
        ],
        rows=rows,
        show_lines=False,
        show_header=False,
    )


def build_sections_table(
    title: str,
    sections: list[str] | list[tuple[str, int | str | None]],
    caption: str | None = None,
) -> Table:
    """Build a normalized section table without content previews."""
    has_lengths = any(isinstance(section, tuple) for section in sections)

    columns = [
        TableColumnSpec("#", style="green", justify="right", no_wrap=True, width=4),
        TableColumnSpec("Section", style="yellow", ratio=4),
    ]
    if has_lengths:
        columns.append(TableColumnSpec("Chars", style="green", justify="right", no_wrap=True, width=8))

    rows: list[tuple[str, ...]] = []
    for index, section in enumerate(sections, start=1):
        if isinstance(section, tuple):
            name, length = section
            length_text = "-" if length is None else str(length)
            rows.append((str(index), str(name), length_text))
            continue

        if has_lengths:
            rows.append((str(index), str(section), "-"))
        else:
            rows.append((str(index), str(section)))

    return build_data_table(
        title=title,
        columns=columns,
        rows=rows,
        caption=caption,
    )


def build_markdown_panel(
    title: str,
    content: str,
    border_style: str = "green",
    subtitle: str | None = None,
) -> Panel:
    """Build a consistent markdown panel for reasoning and final reports."""
    safe_content = content.strip() if content and content.strip() else "No content generated."
    return Panel(
        Markdown(safe_content),
        title=title,
        subtitle=subtitle,
        title_align="left",
        subtitle_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def build_discussion_report(result: Any) -> Panel:
    """Build a comprehensive markdown panel for discussion-based analysis results."""
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
                name = key.replace("_", " ").title()
                markdown_lines.append(f"- **{name}:** {value}")

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
            if hasattr(insight, "supporting_evidence") and insight.supporting_evidence:
                markdown_lines.append("\n**Supporting Evidence:**")
                for evidence in insight.supporting_evidence:
                    markdown_lines.append(f"- {evidence}")
            if hasattr(insight, "related_sections") and insight.related_sections:
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

    return build_markdown_panel("Final Report", "\n".join(markdown_lines), border_style="green")
