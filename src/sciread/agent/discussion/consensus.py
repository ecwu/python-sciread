"""Consensus builder for multi-agent discussion system."""

import re
from collections import defaultdict
from datetime import UTC
from datetime import datetime
from typing import Any

from pydantic_ai import Agent

from ...document_structure import Document
from ...llm_provider import get_model
from ...platform.logging import get_logger
from .models import AgentInsight
from .models import AgentPersonality
from .models import ConsensusPoint
from .models import DiscussionResult
from .models import DiscussionState
from .models import DivergentView
from .models import Question
from .models import Response

logger = get_logger(__name__)


class ConsensusBuilder:
    """Builds consensus from multi-agent discussion results."""

    CONTRIBUTION_ACTION_KEYWORDS = (
        "introduces",
        "proposes",
        "presents",
        "develops",
        "provides",
        "demonstrates",
        "establishes",
        "offers",
        "enables",
        "achieves",
    )

    CONTRIBUTION_TOPIC_KEYWORDS = (
        "contribution",
        "contributions",
        "novel",
        "innovation",
        "innovative",
        "framework",
        "method",
        "methodology",
        "approach",
        "technique",
        "algorithm",
        "model",
        "system",
        "architecture",
        "benchmark",
        "dataset",
        "finding",
        "findings",
        "result",
        "results",
    )

    NON_CONTRIBUTION_KEYWORDS = (
        "limitation",
        "limitations",
        "weakness",
        "weaknesses",
        "drawback",
        "drawbacks",
        "concern",
        "concerns",
        "problem",
        "problems",
        "issue",
        "issues",
        "risk",
        "risks",
        "lacks",
        "lack of",
        "lacking",
        "omitted",
        "omission",
        "failed",
        "failure",
        "unclear",
        "questionable",
        "critic",
        "critique",
        "relies heavily",
    )

    def __init__(self, model_name: str = "deepseek-chat"):
        """Initialize the consensus builder."""
        self.model_name = model_name
        self.model = get_model(model_name)
        self.agent = Agent(
            self.model,
            system_prompt="你是一名学术研究分析领域的共识整合专家。你的任务是综合多个专家智能体的洞见，形成全面、平衡的评估。",
        )
        self.logger = get_logger(__name__)

    async def build_consensus_result(
        self,
        document: Document,
        discussion_state: DiscussionState | None,
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
                document_title=document.metadata.title or "未命名论文",
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
                completion_time=datetime.now(UTC),
            )

            self.logger.info(f"Consensus building completed. Confidence: {confidence_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Consensus building failed: {e}")
            # Return minimal result on error
            return DiscussionResult(
                document_title=document.metadata.title or "未命名论文",
                summary=f"构建共识结果时出错：{e!s}",
                key_contributions=[],
                significance="分析失败",
                confidence_score=0.0,
                discussion_metadata={"error": str(e)},
                completion_time=datetime.now(UTC),
            )

    def _extract_top_insights(
        self,
        agent_insights: dict[AgentPersonality, list[AgentInsight]],
        limit: int = 10,
    ) -> list[AgentInsight]:
        """Extract the most important insights across all agents."""
        all_insights = []

        for _personality, insights in agent_insights.items():
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

                    target_insight: AgentInsight | None = None

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
请基于以下学术论文分析结果，生成一份综合摘要和整体意义评估。

**论文标题：** {document.metadata.title}

**分析提炼出的关键洞见：**
{chr(10).join(f"- {insight.content[:200]}..." for insight in top_insights[:5])}

**已形成的主要共识：**
{chr(10).join(f"- {point.topic}: {point.content[:150]}..." for point in consensus_points[:3]) if consensus_points else "尚未识别出明显的共识点。"}

**仍存在分歧的方面：**
{chr(10).join(f"- {view.topic}: {view.content[:150]}..." for view in divergent_views[:2]) if divergent_views else "目前未识别出显著分歧。"}

**要求：**
1. 生成一段 300-500 字的中文摘要，简洁但完整地概括主要发现
2. 结合共识与分歧，对论文整体意义进行评估
3. 同时强调分析中识别出的优势与局限

**输出格式要求：**
- 摘要和意义评估的正文必须使用中文。
- 为兼容现有解析逻辑，请保留以下两个英文标签：
SUMMARY:
[请在这里写中文摘要]

SIGNIFICANCE:
[请在这里写中文意义评估]
"""

            result = await self.agent.run(prompt)

            # Parse response
            summary_match = re.search(
                r"SUMMARY:\s*(.+?)(?=SIGNIFICANCE:|$)",
                result.output,
                re.DOTALL | re.IGNORECASE,
            )
            significance_match = re.search(r"SIGNIFICANCE:\s*(.+)", result.output, re.DOTALL | re.IGNORECASE)

            summary = summary_match.group(1).strip() if summary_match else "摘要生成失败。"
            significance = significance_match.group(1).strip() if significance_match else "意义评估生成失败。"

            return summary, significance

        except Exception as e:
            self.logger.error(f"Error generating summary and significance: {e}")
            return "摘要生成失败。", "意义评估生成失败。"

    async def _extract_key_contributions(
        self,
        document: Document,
        top_insights: list[AgentInsight],
        consensus_points: list[ConsensusPoint],
    ) -> list[str]:
        """Extract key contributions identified in the analysis."""
        try:
            scored_candidates: list[tuple[int, int, str]] = []

            for insight in top_insights:
                candidate = self._normalize_contribution_candidate(insight.content)
                score = self._score_contribution_candidate(candidate)
                if score > 0:
                    scored_candidates.append((score, int(insight.importance_score * 100), candidate))

            for point in consensus_points:
                candidate = self._normalize_contribution_candidate(point.content)
                score = self._score_contribution_candidate(candidate)
                if score > 0:
                    scored_candidates.append((score + 1, int(point.strength * 100), candidate))

            if not scored_candidates:
                for insight in top_insights:
                    candidate = self._normalize_contribution_candidate(insight.content)
                    if self._is_viable_fallback_contribution(candidate):
                        scored_candidates.append((1, int(insight.importance_score * 100), candidate))

            ranked_candidates = sorted(scored_candidates, key=lambda item: (item[0], item[1], len(item[2])), reverse=True)

            contributions: list[str] = []
            seen_normalized: set[str] = set()
            for _score, _importance, candidate in ranked_candidates:
                normalized = self._dedupe_contribution_candidate(candidate)
                if normalized in seen_normalized:
                    continue
                contributions.append(candidate)
                seen_normalized.add(normalized)
                if len(contributions) == 5:
                    break

            return contributions

        except Exception as e:
            self.logger.error(f"Error extracting key contributions: {e}")
            return []

    def _normalize_contribution_candidate(self, content: str) -> str:
        """Normalize contribution text for display and deduplication."""
        candidate = re.sub(r"\s+", " ", content).strip()
        candidate = re.sub(r"^\d+\.\s*", "", candidate)

        return candidate

    def _score_contribution_candidate(self, candidate: str) -> int:
        """Score whether a statement is likely to be a real paper contribution."""
        if not candidate:
            return 0

        candidate_lower = candidate.lower()

        if any(
            phrase in candidate_lower
            for phrase in [
                "analysis of ",
                "evaluation of ",
                "key contributions is omitted",
                "key contribution is omitted",
                "main contribution is omitted",
            ]
        ):
            return 0

        action_hits = sum(keyword in candidate_lower for keyword in self.CONTRIBUTION_ACTION_KEYWORDS)
        topic_hits = sum(keyword in candidate_lower for keyword in self.CONTRIBUTION_TOPIC_KEYWORDS)
        negative_hits = sum(keyword in candidate_lower for keyword in self.NON_CONTRIBUTION_KEYWORDS)

        if action_hits == 0 and topic_hits == 0:
            return 0

        if negative_hits >= action_hits + topic_hits and action_hits < 2:
            return 0

        if negative_hits > 0 and any(
            phrase in candidate_lower
            for phrase in [
                "raises significant concerns",
                "lacks public validation",
                "relies heavily on",
            ]
        ):
            return 0

        score = (action_hits * 3) + topic_hits - (negative_hits * 2)

        if any(candidate_lower.startswith(keyword) for keyword in self.CONTRIBUTION_ACTION_KEYWORDS):
            score += 2

        return max(score, 0)

    def _is_viable_fallback_contribution(self, candidate: str) -> bool:
        """Use only affirmative, non-critical insight text as fallback contributions."""
        candidate_lower = candidate.lower()
        if any(keyword in candidate_lower for keyword in self.NON_CONTRIBUTION_KEYWORDS):
            return False
        return bool(
            any(keyword in candidate_lower for keyword in self.CONTRIBUTION_ACTION_KEYWORDS)
            or any(keyword in candidate_lower for keyword in ("framework", "method", "approach", "system", "model"))
        )

    def _dedupe_contribution_candidate(self, candidate: str) -> str:
        """Create a normalized key so near-identical contribution strings collapse."""
        normalized = candidate.lower()
        normalized = normalized.replace("...", "")
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

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

        for _personality, insights in agent_insights.items():
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
            return "方法"
        elif any(keyword in content_lower for keyword in ["result", "finding", "outcome"]):
            return "结果"
        elif any(keyword in content_lower for keyword in ["limitation", "weakness", "drawback"]):
            return "局限性"
        elif any(keyword in content_lower for keyword in ["application", "use case", "practical"]):
            return "应用"
        elif any(keyword in content_lower for keyword in ["theory", "framework", "model"]):
            return "理论贡献"
        elif any(keyword in content_lower for keyword in ["innovation", "novel", "breakthrough"]):
            return "创新"
        else:
            return "综合分析"

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
            supporting_agents = list({insight.agent_id for insight in insights})
            sum(insight.importance_score for insight in insights) / len(insights)
            avg_confidence = sum(insight.confidence for insight in insights) / len(insights)

            # Synthesize consensus content
            consensus_content = f"围绕“{topic}”的综合判断：" + "；".join(
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
