"""Personality-based agents for multi-agent discussion system."""

import re
import uuid
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ..document import Document
from ..document.document_renderers import get_sections_content
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.discussion_models import AgentInsight
from .models.discussion_models import AgentPersonality
from .models.discussion_models import Question
from .models.discussion_models import Response
from .prompts.personalities import build_insight_generation_prompt
from .prompts.personalities import get_personality_system_prompt

logger = get_logger(__name__)


class PersonalityAgent:
    """Base class for personality-based agents."""

    def __init__(
        self, personality: AgentPersonality, model_name: str = "deepseek-chat"
    ):
        """Initialize the personality agent."""
        self.personality = personality
        self.model_name = model_name
        self.model = get_model(model_name)
        self.agent = Agent(
            self.model, system_prompt=get_personality_system_prompt(personality)
        )
        self.logger = get_logger(f"{__name__}.{personality.value}")
        self.message_history: list[ModelMessage] = []

    async def _run_with_history(self, prompt: str):
        """Run agent with message history persistence."""
        if self.message_history:
            # Continue existing conversation
            result = await self.agent.run(prompt, message_history=self.message_history)
        else:
            # Start new conversation
            result = await self.agent.run(prompt)

        # Update history with all messages from this run
        self.message_history = result.all_messages()
        return result

    async def generate_insights(
        self, document: Document, discussion_context: dict[str, Any]
    ) -> list[AgentInsight]:
        """Generate insights based on document and personality."""
        try:
            self.logger.info(f"Generating insights for {self.personality.value}")

            # Get abstract from document
            abstract_chunks = document.get_sections_by_name(["abstract"])
            abstract_text = (
                " ".join(chunk.content for chunk in abstract_chunks)
                if abstract_chunks
                else "No abstract available"
            )

            # Step 1: Let agent select which sections to read based on personality
            section_names = document.get_section_names()
            selected_sections = await self._select_sections_to_read(
                document.metadata.title or "Untitled", abstract_text, section_names
            )

            self.logger.info(
                f"{self.personality.value} selected {len(selected_sections)} sections to read: {selected_sections}"
            )

            # Step 2: Get content of selected sections
            selected_content = self._get_section_content(document, selected_sections)

            # Step 3: Build the prompt with actual content
            prompt = build_insight_generation_prompt(
                personality=self.personality,
                document_title=document.metadata.title or "Untitled",
                document_abstract=abstract_text,
                key_sections=section_names,
                selected_sections_content=selected_content,
                discussion_context=discussion_context,
            )

            # Execute the agent with history
            result = await self._run_with_history(prompt)

            # Parse the response to extract insights
            insights = self._parse_insights_response(result.output, document)

            self.logger.info(
                f"Generated {len(insights)} insights for {self.personality.value}"
            )
            return insights

        except Exception as e:
            self.logger.error(
                f"Error generating insights for {self.personality.value}: {e}"
            )
            return []

    async def _select_sections_to_read(
        self, title: str, abstract: str, available_sections: list[str]
    ) -> list[str]:
        """Select which sections to read based on personality and paper overview."""
        try:
            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, you need to select which sections of a paper to read carefully.

**Paper Title:** {title}
**Abstract:** {abstract}

**Available Sections:**
{chr(10).join(f"{i + 1}. {section}" for i, section in enumerate(available_sections))}

**Your Task:**
Based on your analytical focus as a {self.personality.value.replace("_", " ").title()}, select 3-5 sections that are most relevant to your perspective:
- Critical Evaluator: Focus on methodology, results, limitations
- Innovative Insighter: Focus on novel approaches, innovations, future work
- Practical Applicator: Focus on applications, experiments, real-world impact
- Theoretical Integrator: Focus on theoretical framework, related work, conclusions

Respond with ONLY the section names you want to read, one per line, exactly as they appear in the list above.
Select sections that will help you provide the most valuable insights from your unique perspective.
"""

            result = await self._run_with_history(prompt)

            # Parse section names from response
            selected = []
            response_lines = result.output.strip().split("\n")

            for line in response_lines:
                line = line.strip().strip("-").strip("*").strip()
                # Remove numbering if present
                line = line.split(".", 1)[-1].strip() if "." in line else line

                # Match against available sections (case-insensitive, flexible matching)
                for section in available_sections:
                    if line and (
                        line.lower() == section.lower()
                        or line.lower() in section.lower()
                        or section.lower() in line.lower()
                    ):
                        if section not in selected:
                            selected.append(section)
                            break

            # Fallback: if no sections selected or too few, use defaults based on personality
            if len(selected) < 2:
                selected = self._get_default_sections(available_sections)

            return selected[:5]  # Limit to 5 sections max

        except Exception as e:
            self.logger.error(
                f"Error selecting sections for {self.personality.value}: {e}"
            )
            return self._get_default_sections(available_sections)

    def _get_default_sections(self, available_sections: list[str]) -> list[str]:
        """Get default sections based on personality if selection fails."""
        try:
            # Use unified section matching for better results
            defaults = []

            # Since we don't have document access here, use pattern-based matching
            # Always include abstract and introduction if available
            for section in available_sections:
                section_lower = section.lower()
                if "abstract" in section_lower and not any(
                    "abstract" in d.lower() for d in defaults
                ):
                    defaults.append(section)
                elif "introduction" in section_lower and not any(
                    "introduction" in d.lower() for d in defaults
                ):
                    defaults.append(section)

            # Add personality-specific defaults using pattern matching
            if self.personality == AgentPersonality.CRITICAL_EVALUATOR:
                targets = [
                    "methodology",
                    "experiments",
                    "results",
                    "evaluation",
                    "limitations",
                ]
            elif self.personality == AgentPersonality.INNOVATIVE_INSIGHTER:
                targets = [
                    "approach",
                    "innovation",
                    "novelty",
                    "contributions",
                    "future",
                ]
            elif self.personality == AgentPersonality.PRACTICAL_APPLICATOR:
                targets = [
                    "applications",
                    "experiments",
                    "implementation",
                    "deployment",
                ]
            else:  # THEORETICAL_INTEGRATOR
                targets = [
                    "related work",
                    "background",
                    "theory",
                    "framework",
                    "conclusion",
                ]

            # Use pattern matching to find sections
            for target in targets:
                for section in available_sections:
                    if section not in defaults:
                        section_lower = section.lower()
                        target_lower = target.lower()
                        if (
                            target_lower in section_lower
                            or section_lower in target_lower
                            or any(
                                word in section_lower for word in target_lower.split()
                            )
                        ):
                            defaults.append(section)
                            break

            return defaults[:5] if defaults else available_sections[:3]

        except Exception as e:
            self.logger.warning(
                f"Enhanced section matching failed, using fallback approach: {e}"
            )

            # Fallback to original approach
            defaults = []
            for section in available_sections:
                section_lower = section.lower()
                if "abstract" in section_lower or "introduction" in section_lower:
                    defaults.append(section)

            # Add personality-specific defaults
            if self.personality == AgentPersonality.CRITICAL_EVALUATOR:
                keywords = [
                    "method",
                    "result",
                    "experiment",
                    "evaluation",
                    "limitation",
                ]
            elif self.personality == AgentPersonality.INNOVATIVE_INSIGHTER:
                keywords = ["approach", "innovation", "novel", "future", "contribution"]
            elif self.personality == AgentPersonality.PRACTICAL_APPLICATOR:
                keywords = ["application", "experiment", "implementation", "practical"]
            else:  # THEORETICAL_INTEGRATOR
                keywords = ["related", "work", "theory", "framework", "conclusion"]

            for section in available_sections:
                section_lower = section.lower()
                if any(keyword in section_lower for keyword in keywords):
                    if section not in defaults:
                        defaults.append(section)

            return defaults[:5] if defaults else available_sections[:3]

    def _get_section_content(
        self, document: Document, section_names: list[str]
    ) -> dict[str, str]:
        """Get the actual content of selected sections using unified section handling."""
        content_dict: dict[str, str] = {}

        sections = get_sections_content(
            document,
            section_names=section_names,
            clean_text=True,
            max_chars_per_section=3000,
        )
        for name, content in sections:
            content_dict[name] = content

        available = document.get_section_names()
        missing = [s for s in section_names if s not in content_dict]
        if missing:
            fallback_names = self._get_default_sections(available)
            fallback_sections = get_sections_content(
                document,
                section_names=fallback_names,
                clean_text=True,
                max_chars_per_section=3000,
            )
            for name, content in fallback_sections:
                if name not in content_dict:
                    content_dict[name] = content

        return content_dict

    async def ask_question(
        self,
        target_insight: AgentInsight,
        target_agent: AgentPersonality,
        discussion_context: dict[str, Any],
    ) -> Any | None:
        """Ask a question about another agent's insight."""
        try:
            self.logger.info(
                f"{self.personality.value} asking question to {target_agent.value}"
            )

            insight_author = getattr(target_insight, "agent_id", target_agent)
            author_value = getattr(insight_author, "value", insight_author)
            author_name = str(author_value).replace("_", " ").title()

            prior_qa = discussion_context.get("prior_qa_for_insight", [])
            my_prior_questions = discussion_context.get("my_prior_questions", [])

            prior_qa_text = self._format_prior_qa_for_prompt(prior_qa)
            my_prior_text = self._format_my_prior_questions_for_prompt(
                my_prior_questions
            )

            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, review the following insight and decide whether a follow-up question is truly necessary.

**Insight Author:** {author_name}
**You Are Asking To:** {target_agent.value.replace("_", " ").title()}

**Target Insight:**
{target_insight.content}
**Importance Score:** {target_insight.importance_score}
**Confidence:** {target_insight.confidence}
**Supporting Evidence:** {", ".join(target_insight.supporting_evidence)}

{prior_qa_text}

{my_prior_text}

**Discussion Context:**
Current Phase: {discussion_context.get("phase", "questioning")}
Iteration: {discussion_context.get("iteration", 1)}
Questions Asked So Far: {discussion_context.get("total_questions", 0)}

**Before deciding to ask anything, reflect on:**
- Does this insight contain unresolved risks, contradictions, or missing evidence from your perspective?
- Have your previous questions been answered satisfactorily? If yes, no need to ask again.
- If your question was answered, does the response address your concern or create new ones?
- Would asking a NEW question materially change the shared understanding or convergence?

Only ask a question if it will meaningfully advance the discussion. If the insight already satisfies your concerns OR your previous question has been answered satisfactorily, choose to skip.

Provide your response in *exactly* this format:
```
Decision: [ask|skip]
Reason: [brief justification for your decision from your personality's viewpoint]
Question: [your specific question or "None" if skipping]
Priority: [0.0-1.0 importance score, use 0.0 when skipping]
Type: [clarification/challenge/extension/none]
```
When you choose `Decision: skip`, you must set `Question: None`, `Priority: 0.0`, and `Type: none`.
When you choose `Decision: ask`, craft one precise question that reflects your personality and advances the dialogue.
"""

            result = await self._run_with_history(prompt)
            parsed = self._parse_question_response(
                result.output, target_insight, target_agent
            )

            if not parsed:
                return None

            if parsed.get("decision") == "skip":
                self.logger.info(
                    f"{self.personality.value} chose to skip questioning {target_agent.value}: {parsed.get('reason', 'no reason provided')}"
                )
                return parsed

            question_obj = parsed.get("question")
            if question_obj:
                self.logger.debug(
                    f"Generated question from {self.personality.value} to {target_agent.value}"
                )
            return question_obj

        except Exception as e:
            self.logger.error(
                f"Error asking question from {self.personality.value}: {e}"
            )
            return None

    async def answer_question(
        self,
        question: Question,
        my_insights: list[AgentInsight],
        discussion_context: dict[str, Any],
    ) -> Response | None:
        """Answer a question from another agent."""
        try:
            # Handle from_agent which might be string or enum
            from_agent_str = (
                question.from_agent
                if isinstance(question.from_agent, str)
                else question.from_agent.value
            )

            self.logger.info(
                f"{self.personality.value} answering question from {from_agent_str}"
            )

            relevant_insights = self._find_relevant_insights_for_question(
                my_insights, question
            )

            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, answer the following question from {from_agent_str.replace("_", " ").title()}:

**Question:**
{question.content}

**Your Relevant Insights:**
{chr(10).join(f"- {insight.content}" for insight in relevant_insights) if relevant_insights else "No direct relevant insights found."}

**Discussion Context:**
Current Phase: {discussion_context.get("phase", "responding")}
Question Priority: {question.priority}

**Your Task:**
Provide a thoughtful response that:
1. Directly addresses the question
2. Maintains your personality's perspective
3. Provides clear reasoning and evidence
4. Suggests revisions to your insights if appropriate

Provide your response in this format:
```
Response: [Your detailed response]
Stance: [agree/disagree/clarify/modify]
Revised Insight: [If modifying, provide revised insight text, otherwise "None"]
Confidence: [0.0-1.0 confidence in your response]
```
"""

            result = await self._run_with_history(prompt)
            parsed = self._parse_answer_response(result.output, question)

            if parsed:
                self.logger.debug(f"Generated response from {self.personality.value}")
            return parsed

        except Exception as e:
            self.logger.error(
                f"Error answering question for {self.personality.value}: {e}"
            )
            return None

    async def evaluate_convergence(
        self,
        all_insights: list[AgentInsight],
        all_questions: list[Question],
        all_responses: list[Response],
        discussion_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate if the discussion has reached convergence."""
        try:
            self.logger.debug(f"{self.personality.value} evaluating convergence")

            qa_thread = self._build_qa_thread_summary(all_questions, all_responses)
            my_questions_answered = self._count_my_answered_questions(
                all_questions, all_responses
            )

            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, evaluate whether the discussion has reached sufficient convergence:

**Current State:**
- Total Insights: {len(all_insights)}
- Total Questions: {len(all_questions)}
- Total Responses: {len(all_responses)}
- Current Iteration: {discussion_context.get("iteration", 1)}
- Your questions answered: {my_questions_answered["answered"]}/{my_questions_answered["total"]}

**Recent Insights from All Agents:**
{chr(10).join(f"{insight.agent_id}: {insight.content[:150]}..." for insight in all_insights[-8:])}

**Recent Q&A Discussion Thread:**
{qa_thread}

**Evaluation Criteria:**
1. Are insights becoming more consistent and aligned?
2. Are your questions being answered? Were the answers satisfactory?
3. Are major disagreements being resolved through the Q&A?
4. From your perspective, is further discussion likely to yield significant new insights?

**Provide your evaluation in this format:**
```
Convergence Score: [0.0-1.0]
Continue Discussion: [yes/no]
Key Issues Remaining: [List any major unresolved issues]
Recommendations: [Any suggestions for next steps]
```
"""

            result = await self._run_with_history(prompt)
            evaluation = self._parse_convergence_evaluation(result.output)

            self.logger.info(
                f"{self.personality.value} convergence evaluation: {evaluation.get('convergence_score', 0.0)}"
            )
            return evaluation

        except Exception as e:
            self.logger.error(
                f"Error evaluating convergence for {self.personality.value}: {e}"
            )
            return {"convergence_score": 0.5, "continue_discussion": True}

    def _build_qa_thread_summary(
        self, all_questions: list[Question], all_responses: list[Response]
    ) -> str:
        """Build a summary of Q&A threads for convergence evaluation."""
        if not all_questions:
            return "(No questions asked yet)"

        response_map = {r.question_id: r for r in all_responses}
        lines = []

        for q in all_questions[-10:]:
            from_agent = (
                q.from_agent if isinstance(q.from_agent, str) else q.from_agent.value
            )
            to_agent = q.to_agent if isinstance(q.to_agent, str) else q.to_agent.value
            response = response_map.get(q.question_id)

            q_summary = q.content[:100] + "..." if len(q.content) > 100 else q.content
            lines.append(f"Q ({from_agent} → {to_agent}): {q_summary}")

            if response:
                r_summary = (
                    response.content[:100] + "..."
                    if len(response.content) > 100
                    else response.content
                )
                lines.append(f"  A ({response.stance}): {r_summary}")
            else:
                lines.append("  A: (Pending)")

        return "\n".join(lines) if lines else "(No Q&A yet)"

    def _count_my_answered_questions(
        self, all_questions: list[Question], all_responses: list[Response]
    ) -> dict[str, int]:
        """Count how many of this agent's questions have been answered."""
        response_ids = {r.question_id for r in all_responses}
        my_questions = [
            q
            for q in all_questions
            if (q.from_agent if isinstance(q.from_agent, str) else q.from_agent.value)
            == self.personality.value
        ]
        answered = sum(1 for q in my_questions if q.question_id in response_ids)
        return {"total": len(my_questions), "answered": answered}

    def _parse_insights_response(
        self, response: str, document: Document
    ) -> list[AgentInsight]:
        """Parse the agent's response to extract AgentInsight objects."""
        insights = []

        try:
            lines = response.split("\n")
            current_insight: dict[str, Any] = {}
            insight_count = 0

            for line in lines:
                line = line.strip()
                if line.startswith("```"):
                    continue

                if line.lower().startswith("insight:") or line.lower().startswith(
                    "finding:"
                ):
                    if current_insight and "content" in current_insight:
                        insights.append(
                            self._create_insight_from_dict(current_insight, document)
                        )
                        insight_count += 1
                        if insight_count >= 3:
                            break

                    current_insight = {"content": line.split(":", 1)[1].strip()}

                elif line.lower().startswith("importance:"):
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        match = re.search(r"(\d+\.?\d*)", value_str)
                        if match:
                            score = float(match.group(1))
                            current_insight["importance_score"] = min(
                                max(score, 0.0), 1.0
                            )
                        else:
                            current_insight["importance_score"] = 0.5
                    except Exception:
                        current_insight["importance_score"] = 0.5

                elif line.lower().startswith("confidence:"):
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        match = re.search(r"(\d+\.?\d*)", value_str)
                        if match:
                            score = float(match.group(1))
                            current_insight["confidence"] = min(max(score, 0.0), 1.0)
                        else:
                            current_insight["confidence"] = 0.5
                    except Exception:
                        current_insight["confidence"] = 0.5

                elif line.lower().startswith("evidence:"):
                    current_insight["evidence"] = line.split(":", 1)[1].strip()

                elif line.lower().startswith("questions:"):
                    current_insight["questions"] = line.split(":", 1)[1].strip()

                elif current_insight and "evidence" in current_insight:
                    if (
                        line
                        and not line.startswith("-")
                        and not any(
                            line.lower().startswith(f)
                            for f in [
                                "insight:",
                                "finding:",
                                "importance:",
                                "confidence:",
                                "questions:",
                            ]
                        )
                    ):
                        current_insight["evidence"] = (
                            f"{current_insight['evidence']} {line}"
                        )

            if current_insight and "content" in current_insight:
                insights.append(
                    self._create_insight_from_dict(current_insight, document)
                )

            if insights:
                self.logger.debug(f"Parsed {len(insights)} insights from response")
                for i, insight in enumerate(insights):
                    has_evidence = bool(
                        insight.supporting_evidence and insight.supporting_evidence[0]
                    )
                    self.logger.debug(
                        f"  Insight {i + 1}: importance={insight.importance_score}, "
                        f"confidence={insight.confidence}, has_evidence={has_evidence}"
                    )
            else:
                self.logger.warning(
                    "No structured insights parsed from response, using fallback"
                )

            if not insights:
                insights.append(
                    AgentInsight(
                        agent_id=self.personality,
                        content=response[:500],
                        importance_score=0.5,
                        confidence=0.5,
                        supporting_evidence=[
                            "(Unstructured response - no specific evidence extracted)"
                        ],
                        related_sections=document.get_section_names()[:3],
                    )
                )

        except Exception as e:
            self.logger.error(f"Error parsing insights response: {e}")
            insights.append(
                AgentInsight(
                    agent_id=self.personality,
                    content=response[:500],
                    importance_score=0.5,
                    confidence=0.5,
                    supporting_evidence=["(Parse error - no evidence extracted)"],
                    related_sections=[],
                )
            )

        return insights

    def _create_insight_from_dict(
        self, insight_dict: dict[str, Any], document: Document
    ) -> AgentInsight:
        """Create an AgentInsight from a dictionary."""
        return AgentInsight(
            agent_id=self.personality,
            content=insight_dict.get("content", ""),
            importance_score=insight_dict.get("importance_score", 0.5),
            confidence=insight_dict.get("confidence", 0.5),
            supporting_evidence=(
                [insight_dict.get("evidence", "")]
                if insight_dict.get("evidence")
                else []
            ),
            related_sections=document.get_section_names()[:3],
            questions_raised=(
                [insight_dict.get("questions", "")]
                if insight_dict.get("questions")
                else []
            ),
        )

    def _format_prior_qa_for_prompt(self, prior_qa: list[dict[str, Any]]) -> str:
        """Format prior Q&A pairs for inclusion in prompt."""
        if not prior_qa:
            return ""

        lines = ["**Prior Questions & Answers about this insight:**"]
        for qa in prior_qa:
            from_agent = qa.get("from_agent", "Unknown")
            question = qa.get("question", "")
            response = qa.get("response")
            stance = qa.get("response_stance")

            lines.append(f"- Q from {from_agent}: {question}")
            if response:
                lines.append(
                    f"  A: {response[:200]}..."
                    if len(response) > 200
                    else f"  A: {response}"
                )
                if stance:
                    lines.append(f"  Stance: {stance}")
            else:
                lines.append("  A: (No response yet)")

        return "\n".join(lines)

    def _format_my_prior_questions_for_prompt(
        self, my_prior_questions: list[dict[str, Any]]
    ) -> str:
        """Format the agent's own prior questions about this insight."""
        if not my_prior_questions:
            return ""

        lines = ["**Your previous questions about this insight:**"]
        for qa in my_prior_questions:
            question = qa.get("question", "")
            response = qa.get("response")
            stance = qa.get("response_stance")

            lines.append(f"- Your question: {question}")
            if response:
                lines.append(
                    f"  Response received: {response[:200]}..."
                    if len(response) > 200
                    else f"  Response received: {response}"
                )
                if stance:
                    lines.append(f"  Their stance: {stance}")
            else:
                lines.append("  (Awaiting response)")

        return "\n".join(lines)

    def _parse_question_response(
        self,
        response: str,
        target_insight: AgentInsight,
        target_agent: AgentPersonality,
    ) -> dict[str, Any] | None:
        """Parse LLM output into a structured decision about questioning."""
        try:
            fields: dict[str, str] = {}
            current_key: str | None = None

            for raw_line in response.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    current_key = key.strip().lower()
                    fields[current_key] = value.strip()
                elif current_key:
                    # Allow the model to spill into the next line; keep it compact
                    fields[current_key] = f"{fields[current_key]} {line}"

            decision = fields.get("decision", "ask").lower()
            reason = fields.get("reason", "").strip()
            question_text = fields.get("question", "").strip()
            priority_raw = fields.get("priority", "").strip()
            question_type = fields.get("type", "").strip() or "clarification"

            try:
                priority = float(priority_raw) if priority_raw else 0.0
            except ValueError:
                priority = 0.0

            skip_requested = decision == "skip" or question_text.lower() in {
                "none",
                "no",
                "n/a",
            }

            if skip_requested:
                return {
                    "decision": "skip",
                    "reason": reason or "No further clarification needed.",
                    "question": None,
                    "priority": 0.0,
                    "type": "none",
                }

            if not question_text:
                return None

            question_obj = Question(
                question_id=str(uuid.uuid4()),
                from_agent=self.personality,
                to_agent=target_agent,
                content=question_text,
                target_insight=target_insight.content[:50],  # Use first 50 chars as ID
                question_type=question_type if question_type else "clarification",
                priority=min(max(priority, 0.0), 1.0) or 0.5,
                requires_response=True,
            )

            return {
                "decision": "ask",
                "reason": reason,
                "question": question_obj,
                "priority": question_obj.priority,
                "type": question_obj.question_type,
            }
        except Exception as e:
            self.logger.error(f"Error parsing question response: {e}")
            return None

    def _find_relevant_insights_for_question(
        self, my_insights: list[AgentInsight], question: Question
    ) -> list[AgentInsight]:
        """Find insights relevant to a given question using word overlap and evidence matching."""
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

        relevant_insights = []
        question_words = set(question.content.lower().split()) - stop_words

        for insight in my_insights:
            insight_words = set(insight.content.lower().split()) - stop_words
            meaningful_common = insight_words & question_words

            if len(meaningful_common) >= 3:
                relevant_insights.append(insight)
                continue

            for evidence in insight.supporting_evidence:
                if evidence and len(evidence) > 10:
                    evidence_lower = evidence.lower()
                    question_lower = question.content.lower()
                    if (
                        evidence_lower in question_lower
                        or question_lower in evidence_lower
                    ):
                        relevant_insights.append(insight)
                        break

        return relevant_insights

    def _parse_answer_response(
        self, response: str, question: Question
    ) -> Response | None:
        """Parse response to create a Response object."""
        try:
            response_match = re.search(
                r"Response:\s*(.+)", response, re.IGNORECASE | re.DOTALL
            )
            stance_match = re.search(r"Stance:\s*(\w+)", response, re.IGNORECASE)
            revised_match = re.search(
                r"Revised Insight:\s*(.+)", response, re.IGNORECASE
            )
            confidence_match = re.search(
                r"Confidence:\s*([0-9.]+)", response, re.IGNORECASE
            )

            if response_match:
                return Response(
                    response_id=str(uuid.uuid4()),
                    question_id=question.question_id,
                    from_agent=self.personality,
                    content=response_match.group(1).strip(),
                    stance=stance_match.group(1).strip() if stance_match else "clarify",
                    revised_insight=(
                        revised_match.group(1).strip()
                        if revised_match
                        and revised_match.group(1).strip().lower() != "none"
                        else None
                    ),
                    confidence=(
                        float(confidence_match.group(1)) if confidence_match else 0.5
                    ),
                )
        except Exception as e:
            self.logger.error(f"Error parsing answer response: {e}")
            return None

    def _parse_convergence_evaluation(self, response: str) -> dict[str, Any]:
        """Parse response to extract convergence evaluation."""
        try:
            evaluation = {
                "convergence_score": 0.5,
                "continue_discussion": True,
                "key_issues": [],
                "recommendations": [],
            }

            score_match = re.search(
                r"Convergence Score:\s*([0-9.]+)", response, re.IGNORECASE
            )
            continue_match = re.search(
                r"Continue Discussion:\s*(\w+)", response, re.IGNORECASE
            )
            issues_match = re.search(
                r"Key Issues Remaining:\s*(.+)", response, re.IGNORECASE | re.DOTALL
            )
            rec_match = re.search(
                r"Recommendations:\s*(.+)", response, re.IGNORECASE | re.DOTALL
            )

            if score_match:
                evaluation["convergence_score"] = float(score_match.group(1))
            if continue_match:
                evaluation["continue_discussion"] = (
                    continue_match.group(1).lower().startswith("y")
                )
            if issues_match:
                evaluation["key_issues"] = [
                    issue.strip()
                    for issue in issues_match.group(1).split("\n")
                    if issue.strip()
                ]
            if rec_match:
                evaluation["recommendations"] = [
                    rec.strip() for rec in rec_match.group(1).split("\n") if rec.strip()
                ]

            return evaluation
        except Exception as e:
            self.logger.error(f"Error parsing convergence evaluation: {e}")
            return {"convergence_score": 0.5, "continue_discussion": True}


# Factory function to create personality agents
def create_personality_agent(
    personality: AgentPersonality, model_name: str = "deepseek-chat"
) -> PersonalityAgent:
    """Create a personality agent of the specified type."""
    return PersonalityAgent(personality, model_name)
