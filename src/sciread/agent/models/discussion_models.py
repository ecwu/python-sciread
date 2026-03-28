"""Data models for multi-agent discussion system."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class AgentPersonality(StrEnum):
    """Different personality types for discussion agents."""

    CRITICAL_EVALUATOR = "critical_evaluator"
    INNOVATIVE_INSIGHTER = "innovative_insighter"
    PRACTICAL_APPLICATOR = "practical_applicator"
    THEORETICAL_INTEGRATOR = "theoretical_integrator"


class DiscussionPhase(StrEnum):
    """Phases of the discussion process."""

    INITIAL_ANALYSIS = "initial_analysis"
    QUESTIONING = "questioning"
    RESPONDING = "responding"
    CONVERGENCE = "convergence"
    CONSENSUS = "consensus"
    COMPLETED = "completed"


class AgentInsight(BaseModel):
    """Represents an insight or finding from an agent."""

    agent_id: AgentPersonality = Field(..., description="The agent personality type")
    content: str = Field(..., description="The insight content")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Importance score from 0 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level from 0 to 1")
    supporting_evidence: list[str] = Field(default_factory=list, description="Evidence supporting the insight")
    related_sections: list[str] = Field(default_factory=list, description="Related document sections")
    questions_raised: list[str] = Field(default_factory=list, description="Questions this insight raises")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the insight was generated")

    model_config = ConfigDict(use_enum_values=True)


class Question(BaseModel):
    """Represents a question from one agent to another."""

    question_id: str = Field(..., description="Unique identifier for the question")
    from_agent: AgentPersonality = Field(..., description="Agent asking the question")
    to_agent: AgentPersonality = Field(..., description="Agent being questioned")
    content: str = Field(..., description="The question content")
    target_insight: str = Field(..., description="ID of the insight being questioned")
    question_type: str = Field(..., description="Type of question (clarification, challenge, extension)")
    priority: float = Field(..., ge=0.0, le=1.0, description="Priority score from 0 to 1")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the question was asked")
    requires_response: bool = Field(default=True, description="Whether this question requires a response")

    model_config = ConfigDict(use_enum_values=True)


class Response(BaseModel):
    """Represents a response to a question."""

    response_id: str = Field(..., description="Unique identifier for the response")
    question_id: str = Field(..., description="ID of the question being responded to")
    from_agent: AgentPersonality = Field(..., description="Agent providing the response")
    content: str = Field(..., description="The response content")
    stance: str = Field(..., description="Response stance (agree, disagree, clarify, modify)")
    revised_insight: str | None = Field(None, description="Revised insight if applicable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the response")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was given")

    model_config = ConfigDict(use_enum_values=True)


class DiscussionState(BaseModel):
    """Represents the current state of the discussion."""

    current_phase: DiscussionPhase = Field(..., description="Current phase of discussion")
    iteration_count: int = Field(default=0, description="Number of iterations completed")
    max_iterations: int = Field(default=5, description="Maximum allowed iterations")
    insights: list[AgentInsight] = Field(default_factory=list, description="All insights generated")
    questions: list[Question] = Field(default_factory=list, description="All questions asked")
    responses: list[Response] = Field(default_factory=list, description="All responses given")
    phase_history: list[dict[str, Any]] = Field(default_factory=list, description="History of phase transitions")
    convergence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How converged the discussion is")
    start_time: datetime = Field(default_factory=datetime.now, description="Discussion start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")

    model_config = ConfigDict(use_enum_values=True)


class ConsensusPoint(BaseModel):
    """Represents a point of consensus among agents."""

    topic: str = Field(..., description="The consensus topic")
    content: str = Field(..., description="The consensus content")
    supporting_agents: list[AgentPersonality] = Field(..., description="Agents supporting this consensus")
    opposing_agents: list[AgentPersonality] = Field(default_factory=list, description="Agents opposing this consensus")
    strength: float = Field(..., ge=0.0, le=1.0, description="Strength of consensus")
    evidence: list[str] = Field(default_factory=list, description="Evidence supporting the consensus")

    model_config = ConfigDict(use_enum_values=True)


class DivergentView(BaseModel):
    """Represents a divergent view that couldn't be resolved."""

    topic: str = Field(..., description="The topic of divergence")
    content: str = Field(..., description="The divergent view")
    holding_agent: AgentPersonality = Field(..., description="Agent holding this view")
    reasoning: str = Field(..., description="Reasoning behind this view")
    counter_arguments: list[str] = Field(default_factory=list, description="Counter arguments from other agents")

    model_config = ConfigDict(use_enum_values=True)


class DiscussionResult(BaseModel):
    """Final result of the discussion."""

    document_title: str = Field(..., description="Title of the analyzed document")
    summary: str = Field(..., description="Summary of the discussion and findings")
    key_contributions: list[str] = Field(default_factory=list, description="Key contributions identified")
    significance: str = Field(..., description="Overall significance assessment")
    consensus_points: list[ConsensusPoint] = Field(default_factory=list, description="Points of consensus")
    divergent_views: list[DivergentView] = Field(default_factory=list, description="Unresolved divergent views")
    final_insights: list[AgentInsight] = Field(default_factory=list, description="Most important insights")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in results")
    discussion_metadata: dict[str, Any] = Field(default_factory=dict, description="Discussion metadata")
    completion_time: datetime = Field(default_factory=datetime.now, description="When discussion completed")

    model_config = ConfigDict(use_enum_values=True)
