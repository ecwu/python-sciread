"""ReActAgent result models for structured outputs."""

from pydantic import BaseModel
from pydantic import Field


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
    should_continue: bool = Field(
        default=True,
        description="Whether analysis should continue to the next iteration",
    )
    report: str = Field(
        default="",
        description="Final report text. Usually empty in normal iterations; populated when finishing analysis.",
    )
