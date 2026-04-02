"""Tests for shared rich output helpers."""

from rich.console import Console

from sciread.platform.rich_output import TableColumnSpec
from sciread.platform.rich_output import build_data_table
from sciread.platform.rich_output import build_key_value_table
from sciread.platform.rich_output import build_markdown_panel
from sciread.platform.rich_output import build_mode_banner
from sciread.platform.rich_output import build_sections_table
from sciread.platform.rich_output import build_stage_banner


def test_build_mode_and_stage_banners_render_subtitles() -> None:
    """Banners should render both titles and optional summary text."""
    console = Console(record=True, width=80)

    console.print(build_mode_banner("Coordinate", subtitle="Sub-agents ready"))
    console.print(build_stage_banner("Iteration 1", summary_lines=["2 questions", "1 answer"]))

    rendered = console.export_text()
    assert "Coordinate" in rendered
    assert "Sub-agents ready" in rendered
    assert "Iteration 1" in rendered
    assert "2 questions" in rendered
    assert "1 answer" in rendered


def test_build_data_and_key_value_tables_render_rows() -> None:
    """Table builders should preserve row values and captions."""
    console = Console(record=True, width=80)
    table = build_data_table(
        title="Summary",
        columns=[
            TableColumnSpec("Name", style="cyan"),
            TableColumnSpec("Value", style="green"),
        ],
        rows=[("alpha", "1"), ("beta", "2")],
        caption="Two rows",
    )
    key_value = build_key_value_table("Document", [("Chunks", "3"), ("Sections", "2")])

    console.print(table)
    console.print(key_value)

    rendered = console.export_text()
    assert "Summary" in rendered
    assert "alpha" in rendered
    assert "beta" in rendered
    assert "Two rows" in rendered
    assert "Document" in rendered
    assert "Chunks" in rendered
    assert "Sections" in rendered


def test_build_sections_table_handles_named_sections_and_lengths() -> None:
    """Section table should include a character column when lengths are provided."""
    console = Console(record=True, width=80)

    console.print(build_sections_table("Sections", ["Introduction", "Method"]))
    console.print(build_sections_table("Sections With Lengths", [("Results", 250), ("Appendix", None)]))

    rendered = console.export_text()
    assert "Introduction" in rendered
    assert "Method" in rendered
    assert "Sections With Lengths" in rendered
    assert "Results" in rendered
    assert "250" in rendered
    assert "Appendix" in rendered


def test_build_markdown_panel_uses_fallback_for_blank_content() -> None:
    """Blank markdown panels should render a stable fallback message."""
    console = Console(record=True, width=80)

    console.print(build_markdown_panel("Final Report", "   "))

    rendered = console.export_text()
    assert "Final Report" in rendered
    assert "No content generated." in rendered
