"""Models for the search-react agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from pydantic import BaseModel
from pydantic import Field

from ...document import Document
from ...document.retrieval.models import RetrievedChunk


class SearchReactIterationInput(BaseModel):
    """Input to one search-react iteration."""

    task: str = Field(description="The overall analysis task or question")
    previous_thoughts: str = Field(default="", description="Thoughts from the previous iteration")
    processed_queries: list[str] = Field(default_factory=list, description="Queries already issued by previous iterations")
    strategy: str = Field(default="hybrid", description="The default retrieval strategy for this run")
    top_k: int = Field(default=5, description="Maximum retrieval results per search")
    neighbor_window: int = Field(default=1, description="Number of neighbor chunks to include around each hit")


class SearchReactIterationOutput(BaseModel):
    """Output of a single search-react iteration."""

    thoughts: str = Field(description="Agent reasoning and planning for the next action")
    should_continue: bool = Field(default=True, description="Whether analysis should continue")
    report: str = Field(default="", description="Final report text for the completed analysis")


@dataclass
class SearchReactDeps:
    """Immutable dependencies for a search-react iteration."""

    document: Document
    task: str
    iteration_input: SearchReactIterationInput
    current_loop: int = 1
    max_loops: int = 5
    accumulated_memory: str = ""
    show_progress: bool = True


@dataclass
class SearchReactIterationState:
    """Mutable per-iteration state used by tools."""

    queries_run: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    memory_text: str = ""
    tree_inspected: bool = False
    all_memory_read: bool = False


@dataclass
class SearchReactAnalysisState:
    """Aggregated state across iterations."""

    task: str
    strategy: str
    top_k: int
    neighbor_window: int
    processed_queries: list[str] = field(default_factory=list)
    accumulated_memory_fragments: list[str] = field(default_factory=list)
    current_thoughts: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    last_iteration_output: SearchReactIterationOutput | None = None

    @property
    def accumulated_memory(self) -> str:
        """Return the accumulated memory as one text block."""
        return "\n\n".join(fragment for fragment in self.accumulated_memory_fragments if fragment)

    def build_iteration_input(self) -> SearchReactIterationInput:
        """Create the next iteration input."""
        return SearchReactIterationInput(
            task=self.task,
            previous_thoughts=self.current_thoughts,
            processed_queries=self.processed_queries.copy(),
            strategy=self.strategy,
            top_k=self.top_k,
            neighbor_window=self.neighbor_window,
        )

    def apply_iteration(
        self,
        iteration_output: SearchReactIterationOutput,
        iteration_state: SearchReactIterationState,
    ) -> None:
        """Merge one iteration into the analysis state."""
        self.last_iteration_output = iteration_output
        self.current_thoughts = iteration_output.thoughts

        for query in iteration_state.queries_run:
            if query not in self.processed_queries:
                self.processed_queries.append(query)

        existing_ids = {result.chunk.chunk_id for result in self.retrieved_chunks}
        for result in iteration_state.retrieved_chunks:
            if result.chunk.chunk_id not in existing_ids:
                self.retrieved_chunks.append(result)
                existing_ids.add(result.chunk.chunk_id)

        if iteration_state.memory_text:
            self.accumulated_memory_fragments.append(iteration_state.memory_text)

    def build_final_output(self) -> SearchReactIterationOutput:
        """Create the final user-facing output."""
        final_report = ""
        if self.last_iteration_output and self.last_iteration_output.report.strip():
            final_report = self.last_iteration_output.report.strip()
        if not final_report:
            final_report = self.accumulated_memory.strip() or "No memory content was generated during analysis."

        return SearchReactIterationOutput(
            thoughts=self.current_thoughts or "Analysis complete.",
            should_continue=False,
            report=final_report,
        )


@dataclass
class SearchReactStrategyRun:
    """One executed strategy run, used by compare mode."""

    strategy: str
    output: SearchReactIterationOutput
    retrieved_chunks: list[RetrievedChunk]
    total_time_seconds: float
    error: str = ""


@dataclass
class SearchReactAnalysisResult:
    """Top-level result for search-react, including compare mode."""

    task: str
    primary_strategy: str
    runs: list[SearchReactStrategyRun]
