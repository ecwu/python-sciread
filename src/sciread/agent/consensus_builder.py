"""Consensus builder for multi-agent discussion system."""

import re
from collections import defaultdict
from datetime import datetime
from typing import Any
from typing import Optional

from pydantic_ai import Agent

from ..document import Document
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.discussion_models import AgentInsight
from .models.discussion_models import AgentPersonality
from .models.discussion_models import ConsensusPoint
from .models.discussion_models import DiscussionResult
from .models.discussion_models import DiscussionState
from .models.discussion_models import DivergentView
from .models.discussion_models import Question
from .models.discussion_models import Response

logger = get_logger(__name__)


class ConsensusBuilder:
    """Builds consensus from multi-agent discussion results."""

    def __init__(self, model_name: str = "deepseek-chat"):
        """Initialize the consensus builder."""
        self.model_name = model_name
        self.model = get_model(model_name)
        self.agent = Agent(
            self.model,
            system_prompt="You are an expert consensus builder for academic research analysis. Your task is to synthesize insights from multiple expert agents into a comprehensive, balanced assessment.",
        )
        self.logger = get_logger(__name__)

    async def build_consensus_result(
        self,
        document: Document,
        discussion_state: Optional[DiscussionState],
        agent_insights: dict[AgentPersonality, list[AgentInsight]],
        questions: list[Question],
        responses: list[Response],
    ) -> DiscussionResult:
        """Build final consensus result from discussion data."""
        try:
            self.logger.info("Building consensus from discussion results")

            # Extract most important insights
            top_insights = self._extract_top_insights(agent_insights)

            # Identify consensus points
            consensus_points = await self._identify_consensus_points(agent_insights, questions, responses)

            # Identify divergent views
            divergent_views = await self._identify_divergent_views(agent_insights, questions, responses)

            # Generate summary and significance assessment
            summary, significance = await self._generate_summary_and_significance(document, top_insights, consensus_points, divergent_views)

            # Extract key contributions
            key_contributions = await self._extract_key_contributions(document, top_insights, consensus_points)

            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(top_insights, consensus_points)

            # Build result
            result = DiscussionResult(
                document_title=document.metadata.title or "Untitled",
                summary=summary,
                key_contributions=key_contributions,
                significance=significance,
                consensus_points=consensus_points,
                divergent_views=divergent_views,
                final_insights=top_insights,
                confidence_score=confidence_score,
                discussion_metadata={
                    "total_iterations": (discussion_state.iteration_count if discussion_state else 0),
                    "final_phase": (discussion_state.current_phase if discussion_state else "unknown"),
                    "total_insights": sum(len(insights) for insights in agent_insights.values()),
                    "total_questions": len(questions),
                    "total_responses": len(responses),
                    "convergence_score": (discussion_state.convergence_score if discussion_state else 0.0),
                    "agent_insight_counts": {k.value: len(v) for k, v in agent_insights.items()},
                },
                completion_time=datetime.now(),
            )

            self.logger.info(f"Consensus building completed. Confidence: {confidence_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Consensus building failed: {e}")
            # Return minimal result on error
            return DiscussionResult(
                document_title=document.metadata.title or "Untitled",
                summary=f"Error building consensus: {e!s}",
                key_contributions=[],
                significance="Analysis failed",
                confidence_score=0.0,
                discussion_metadata={"error": str(e)},
                completion_time=datetime.now(),
            )

    def _extract_top_insights(
        self,
        agent_insights: dict[AgentPersonality, list[AgentInsight]],
        limit: int = 10,
    ) -> list[AgentInsight]:
        """Extract the most important insights across all agents."""
        all_insights = []

        for personality, insights in agent_insights.items():
            all_insights.extend(insights)

        # Sort by importance score and confidence
        sorted_insights = sorted(
            all_insights,
            key=lambda i: (i.importance_score + i.confidence) / 2,
            reverse=True,
        )

        return sorted_insights[:limit]

    async def _identify_consensus_points(
        self,
        agent_insights: dict[AgentPersonality, list[AgentInsight]],
        questions: list[Question],
        responses: list[Response],
    ) -> list[ConsensusPoint]:
        """Identify points of consensus among agents."""
        try:
            # Group insights by topics/themes
            topic_groups = self._group_insights_by_topic(agent_insights)

            consensus_points = []

            for topic, insights in topic_groups.items():
                # Check if there's consensus on this topic
                consensus_data = await self._evaluate_topic_consensus(topic, insights, questions, responses)

                if consensus_data["has_consensus"]:
                    point = ConsensusPoint(
                        topic=topic,
                        content=consensus_data["content"],
                        supporting_agents=consensus_data["supporting_agents"],
                        opposing_agents=consensus_data["opposing_agents"],
                        strength=consensus_data["strength"],
                        evidence=consensus_data["evidence"],
                    )
                    consensus_points.append(point)

            # Sort by strength
            consensus_points.sort(key=lambda p: p.strength, reverse=True)

            return consensus_points

        except Exception as e:
            self.logger.error(f"Error identifying consensus points: {e}")
            return []

    async def _identify_divergent_views(
        self,
        agent_insights: dict[AgentPersonality, list[AgentInsight]],
        questions: list[Question],
        responses: list[Response],
    ) -> list[DivergentView]:
        """Identify significant divergent views."""
        try:
            # Build lookup so we can recover the original insight from a question reference
            insight_lookup_by_id: dict[str, AgentInsight] = {}
            insight_lookup_by_snippet: dict[str, list[AgentInsight]] = defaultdict(list)

            for insights in agent_insights.values():
                for insight in insights:
                    insight_id = getattr(insight, "insight_id", None)
                    if insight_id:
                        insight_lookup_by_id[insight_id] = insight

                    # Questions currently reference the first 50 chars; store both the clipped and full strings
                    snippet_key = insight.content[:50]
                    insight_lookup_by_snippet[snippet_key].append(insight)
                    insight_lookup_by_snippet[insight.content].append(insight)

            # Find insights that were strongly challenged
            challenged_insights = []

            for question in questions:
                for response in responses:
                    if response.question_id != question.question_id:
                        continue

                    stance = (response.stance or "").lower()
                    if stance not in ["disagree", "challenge", "modify"]:
                        continue

                    target_insight: Optional[AgentInsight] = None

                    target_id = getattr(question, "target_insight_id", None)
                    if target_id:
                        target_insight = insight_lookup_by_id.get(target_id)

                    if not target_insight:
                        target_key = question.target_insight or ""
                        matches = insight_lookup_by_snippet.get(target_key)
                        if matches:
                            target_insight = matches[0]

                    if not target_insight:
                        continue

                    challenged_insights.append(
                        {
                            "question": question,
                            "response": response,
                            "insight": target_insight,
                        }
                    )

            # Group conflicts by topic
            conflicts = self._group_conflicts_by_topic(challenged_insights)

            divergent_views = []

            for topic, conflict_data in conflicts.items():
                view = DivergentView(
                    topic=topic,
                    content=conflict_data["content"],
                    holding_agent=conflict_data["holding_agent"],
                    reasoning=conflict_data["reasoning"],
                    counter_arguments=conflict_data["counter_arguments"],
                )
                divergent_views.append(view)

            return divergent_views

        except Exception as e:
            self.logger.error(f"Error identifying divergent views: {e}")
            return []

    async def _generate_summary_and_significance(
        self,
        document: Document,
        top_insights: list[AgentInsight],
        consensus_points: list[ConsensusPoint],
        divergent_views: list[DivergentView],
    ) -> tuple[str, str]:
        """Generate overall summary and significance assessment."""
        try:
            prompt = f"""
Based on the following analysis of an academic paper, generate a comprehensive summary and significance assessment.

**Paper Title:** {document.metadata.title}

**Key Insights from Analysis:**
{chr(10).join(f"- {insight.content[:200]}..." for insight in top_insights[:5])}

**Points of Consensus:**
{chr(10).join(f"- {point.topic}: {point.content[:150]}..." for point in consensus_points[:3]) if consensus_points else "No major consensus points identified."}

**Areas of Disagreement:**
{chr(10).join(f"- {view.topic}: {view.content[:150]}..." for view in divergent_views[:2]) if divergent_views else "No significant disagreements identified."}

**Instructions:**
1. Generate a concise yet comprehensive summary (300-500 words) that captures the main findings
2. Assess the overall significance of the paper considering the consensus and disagreements
3. Highlight both strengths and limitations identified in the analysis

**Format your response as:**
SUMMARY:
[Your detailed summary here]

SIGNIFICANCE:
[Your significance assessment here]
"""

            result = await self.agent.run(prompt)

            # Parse response
            summary_match = re.search(
                r"SUMMARY:\s*(.+?)(?=SIGNIFICANCE:|$)",
                result.output,
                re.DOTALL | re.IGNORECASE,
            )
            significance_match = re.search(r"SIGNIFICANCE:\s*(.+)", result.output, re.DOTALL | re.IGNORECASE)

            summary = summary_match.group(1).strip() if summary_match else "Summary generation failed."
            significance = significance_match.group(1).strip() if significance_match else "Significance assessment failed."

            return summary, significance

        except Exception as e:
            self.logger.error(f"Error generating summary and significance: {e}")
            return "Summary generation failed.", "Significance assessment failed."

    async def _extract_key_contributions(
        self,
        document: Document,
        top_insights: list[AgentInsight],
        consensus_points: list[ConsensusPoint],
    ) -> list[str]:
        """Extract key contributions identified in the analysis."""
        try:
            contributions = set()

            # From insights
            for insight in top_insights:
                # Look for contribution-related keywords
                content_lower = insight.content.lower()
                if any(
                    keyword in content_lower
                    for keyword in [
                        "contribution",
                        "novel",
                        "innovation",
                        "advancement",
                        "breakthrough",
                    ]
                ):
                    contributions.add(insight.content[:200] + "..." if len(insight.content) > 200 else insight.content)

            # From consensus points
            for point in consensus_points:
                if any(keyword in point.content.lower() for keyword in ["contribution", "novel", "innovation"]):
                    contributions.add(point.content[:200] + "..." if len(point.content) > 200 else point.content)

            # If no clear contributions, use top insights
            if not contributions and top_insights:
                for insight in top_insights[:5]:
                    contributions.add(insight.content[:200] + "..." if len(insight.content) > 200 else insight.content)

            return list(contributions)

        except Exception as e:
            self.logger.error(f"Error extracting key contributions: {e}")
            return []

    def _calculate_overall_confidence(self, top_insights: list[AgentInsight], consensus_points: list[ConsensusPoint]) -> float:
        """Calculate overall confidence in the analysis results."""
        if not top_insights:
            return 0.0

        # Average confidence from insights
        insight_confidence = sum(insight.confidence for insight in top_insights) / len(top_insights)

        # Weight by consensus strength
        consensus_weight = sum(point.strength for point in consensus_points) / (len(consensus_points) + 1) / 2

        # Combined confidence
        overall_confidence = (insight_confidence * 0.7) + (consensus_weight * 0.3)

        return min(overall_confidence, 1.0)

    def _group_insights_by_topic(self, agent_insights: dict[AgentPersonality, list[AgentInsight]]) -> dict[str, list[AgentInsight]]:
        """Group insights by topics/themes."""
        topic_groups = defaultdict(list)

        for personality, insights in agent_insights.items():
            for insight in insights:
                # Simple topic extraction based on keywords
                topic = self._extract_topic_from_insight(insight.content)
                topic_groups[topic].append(insight)

        return dict(topic_groups)

    def _extract_topic_from_insight(self, content: str) -> str:
        """Extract a simple topic from insight content."""
        content_lower = content.lower()

        # Simple keyword-based topic extraction
        if any(keyword in content_lower for keyword in ["method", "approach", "methodology"]):
            return "Methodology"
        elif any(keyword in content_lower for keyword in ["result", "finding", "outcome"]):
            return "Results"
        elif any(keyword in content_lower for keyword in ["limitation", "weakness", "drawback"]):
            return "Limitations"
        elif any(keyword in content_lower for keyword in ["application", "use case", "practical"]):
            return "Applications"
        elif any(keyword in content_lower for keyword in ["theory", "framework", "model"]):
            return "Theoretical Contributions"
        elif any(keyword in content_lower for keyword in ["innovation", "novel", "breakthrough"]):
            return "Innovation"
        else:
            return "General Analysis"

    async def _evaluate_topic_consensus(
        self,
        topic: str,
        insights: list[AgentInsight],
        questions: list[Question],
        responses: list[Response],
    ) -> dict[str, Any]:
        """Evaluate if there's consensus on a specific topic."""
        try:
            if len(insights) < 2:  # Need at least 2 insights for consensus
                return {"has_consensus": False}

            # Check for conflicting responses related to this topic
            conflicting_responses = [
                resp
                for resp in responses
                if resp.stance.lower() in ["disagree", "challenge", "modify"] and topic.lower() in resp.content.lower()
            ]

            if len(conflicting_responses) > len(insights) / 2:
                return {"has_consensus": False}

            # Build consensus content
            supporting_agents = list(set(insight.agent_id for insight in insights))
            avg_importance = sum(insight.importance_score for insight in insights) / len(insights)
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)

            # Synthesize consensus content
            consensus_content = f"Analysis of {topic}: " + "; ".join(
                [(insight.content[:100] + "..." if len(insight.content) > 100 else insight.content) for insight in insights[:3]]
            )

            return {
                "has_consensus": True,
                "content": consensus_content,
                "supporting_agents": supporting_agents,
                "opposing_agents": [],
                "strength": avg_confidence,
                "evidence": [insight.content for insight in insights],
            }

        except Exception as e:
            self.logger.error(f"Error evaluating topic consensus: {e}")
            return {"has_consensus": False}

    def _group_conflicts_by_topic(self, challenged_insights: list[dict]) -> dict[str, dict]:
        """Group conflicts by topic."""
        conflicts = defaultdict(list)

        for challenge in challenged_insights:
            insight: AgentInsight = challenge["insight"]
            question = challenge["question"]
            response = challenge["response"]

            topic = self._extract_topic_from_insight(insight.content)

            conflicts[topic].append(
                {
                    "question": question,
                    "response": response,
                    "insight": insight,
                }
            )

        # Convert to simpler format
        result = {}
        for topic, conflict_list in conflicts.items():
            if conflict_list:
                main_conflict = conflict_list[0]
                main_insight: AgentInsight = main_conflict["insight"]
                result[topic] = {
                    "content": main_insight.content,
                    "holding_agent": main_insight.agent_id,
                    "reasoning": main_conflict["response"].content,
                    "counter_arguments": [c["response"].content for c in conflict_list[1:]],
                }

        return result
