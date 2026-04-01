"""ReAct models and state containers."""

from dataclasses import dataclass
from dataclasses import field

from pydantic import BaseModel
from pydantic import Field

from ...document_structure import Document


class ReActIterationInput(BaseModel):
    """Input to a single ReAct analysis iteration (one loop)."""

    task: str = Field(description="The overall analysis task or question")
    previous_thoughts: str = Field(default="", description="Thoughts from the previous iteration to guide this one")
    processed_sections: list[str] = Field(
        default_factory=list,
        description="Section names already read in previous iterations",
    )
    available_sections: list[str] = Field(description="All available section names in the document")


class ReActIterationOutput(BaseModel):
    """Output of a single ReAct analysis iteration (one tool call + thoughts)."""

    thoughts: str = Field(description="Agent's reasoning, observations, and decisions for the next action")
    should_continue: bool = Field(default=True, description="Whether analysis should continue to the next iteration")
    report: str = Field(default="", description="Final report text. Usually empty in normal iterations; populated when finishing analysis.")


@dataclass
class ReActIterationDeps:
    """Immutable dependencies for a single ReAct iteration."""

    document: Document
    task: str
    iteration_input: ReActIterationInput
    current_loop: int = 1
    max_loops: int = 8
    accumulated_memory: str = ""
    show_progress: bool = True


@dataclass
class ReActIterationState:
    """Mutable per-iteration state used by tools."""

    read_section_called: bool = False
    add_memory_called: bool = False
    sections_read: list[str] = field(default_factory=list)
    section_content: str = ""
    memory_text: str = ""


@dataclass
class ReActAnalysisState:
    """Aggregated state across ReAct iterations."""

    task: str
    available_sections: list[str]
    processed_sections: list[str] = field(default_factory=list)
    accumulated_memory_fragments: list[str] = field(default_factory=list)
    current_thoughts: str = ""
    last_iteration_output: ReActIterationOutput | None = None

    @property
    def accumulated_memory(self) -> str:
        """Return the accumulated memory as a single string."""
        return "\n\n".join(fragment for fragment in self.accumulated_memory_fragments if fragment)

    @property
    def remaining_sections(self) -> list[str]:
        """Return section names that have not been processed yet."""
        return [section for section in self.available_sections if section not in self.processed_sections]

    def build_iteration_input(self) -> ReActIterationInput:
        """Create the next iteration input from the current session state."""
        return ReActIterationInput(
            task=self.task,
            previous_thoughts=self.current_thoughts,
            processed_sections=self.processed_sections.copy(),
            available_sections=self.available_sections,
        )

    def apply_iteration(self, iteration_output: ReActIterationOutput, iteration_state: ReActIterationState) -> None:
        """Merge one iteration's output into the session state."""
        self.last_iteration_output = iteration_output
        self.current_thoughts = iteration_output.thoughts

        for section in iteration_state.sections_read:
            if section not in self.processed_sections:
                self.processed_sections.append(section)

        if iteration_state.memory_text:
            self.accumulated_memory_fragments.append(iteration_state.memory_text)

    def build_final_output(self) -> ReActIterationOutput:
        """Create the public final output from the aggregated state."""
        final_report = (
            self.last_iteration_output.report.strip() if self.last_iteration_output and self.last_iteration_output.report.strip() else ""
        )
        if not final_report:
            final_report = self.accumulated_memory.strip() or "No memory content was generated during analysis."

        return ReActIterationOutput(
            thoughts=self.current_thoughts or "Analysis complete.",
            should_continue=False,
            report=final_report,
        )
