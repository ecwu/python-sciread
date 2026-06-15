"""Tests for shared rich output helpers."""

from types import SimpleNamespace

from rich.console import Console

from sciread.platform.rich_output import TableColumnSpec
from sciread.platform.rich_output import build_data_table
from sciread.platform.rich_output import build_discussion_report
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


def test_build_mode_banner_without_subtitle() -> None:
    """A banner without a subtitle should only render the title."""
    console = Console(record=True, width=80)
    console.print(build_mode_banner("Simple Analysis"))

    rendered = console.export_text()
    assert "Simple Analysis" in rendered
    assert "subtitle" not in rendered.lower()


def test_build_sections_table_with_mixed_tuple_and_string() -> None:
    """A mixed section list should fill missing lengths with '-'."""
    console = Console(record=True, width=80)
    console.print(build_sections_table("Sections", [("Results", 120), "Appendix"]))

    rendered = console.export_text()
    assert "Results" in rendered
    assert "120" in rendered
    assert "Appendix" in rendered
    assert "-" in rendered


def test_build_discussion_report_handles_empty_fields() -> None:
    """Discussion report rendering should tolerate missing optional fields."""
    result = SimpleNamespace(
        document_title="Paper",
        confidence_score=0.75,
        final_insights=[
            SimpleNamespace(
                agent_id="agent-1",
                confidence=0.8,
                importance_score=0.7,
                content="Insight one.",
            )
        ],
        consensus_points=[],
        divergent_views=[],
        summary="Short summary.",
        key_contributions=[],
        significance="",
        discussion_metadata={},
    )

    panel = build_discussion_report(result)
    console = Console(record=True, width=120)
    console.print(panel)
    rendered = console.export_text()

    assert "Paper" in rendered
    assert "Short summary." in rendered
    assert "Insight one." in rendered
    assert "Key Contributions" not in rendered
    assert "Significance Assessment" not in rendered
