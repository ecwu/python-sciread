"""Personality-based agents for multi-agent discussion system."""

import re
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ..document import Document
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

    def __init__(self, personality: AgentPersonality, model_name: str = "deepseek-chat"):
        """Initialize the personality agent."""
        self.personality = personality
        self.model_name = model_name
        self.model = get_model(model_name)
        self.agent = Agent(self.model, system_prompt=get_personality_system_prompt(personality))
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

    async def generate_insights(self, document: Document, discussion_context: Dict[str, Any]) -> List[AgentInsight]:
        """Generate insights based on document and personality."""
        try:
            self.logger.info(f"Generating insights for {self.personality.value}")

            # Get abstract from document
            abstract_chunks = document.get_sections_by_name(["abstract"])
            abstract_text = " ".join(chunk.content for chunk in abstract_chunks) if abstract_chunks else "No abstract available"

            # Step 1: Let agent select which sections to read based on personality
            section_names = document.get_section_names()
            selected_sections = await self._select_sections_to_read(document.metadata.title or "Untitled", abstract_text, section_names)

            self.logger.info(f"{self.personality.value} selected {len(selected_sections)} sections to read: {selected_sections}")

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

            self.logger.info(f"Generated {len(insights)} insights for {self.personality.value}")
            return insights

        except Exception as e:
            self.logger.error(f"Error generating insights for {self.personality.value}: {e}")
            return []

    async def _select_sections_to_read(self, title: str, abstract: str, available_sections: List[str]) -> List[str]:
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
                    if line and (line.lower() == section.lower() or line.lower() in section.lower() or section.lower() in line.lower()):
                        if section not in selected:
                            selected.append(section)
                            break

            # Fallback: if no sections selected or too few, use defaults based on personality
            if len(selected) < 2:
                selected = self._get_default_sections(available_sections)

            return selected[:5]  # Limit to 5 sections max

        except Exception as e:
            self.logger.error(f"Error selecting sections for {self.personality.value}: {e}")
            return self._get_default_sections(available_sections)

    def _get_default_sections(self, available_sections: List[str]) -> List[str]:
        """Get default sections based on personality if selection fails."""
        try:
            # Use unified section matching for better results
            defaults = []

            # Since we don't have document access here, use pattern-based matching
            # Always include abstract and introduction if available
            for section in available_sections:
                section_lower = section.lower()
                if "abstract" in section_lower and not any("abstract" in d.lower() for d in defaults):
                    defaults.append(section)
                elif "introduction" in section_lower and not any("introduction" in d.lower() for d in defaults):
                    defaults.append(section)

            # Add personality-specific defaults using pattern matching
            if self.personality == AgentPersonality.CRITICAL_EVALUATOR:
                targets = ["methodology", "experiments", "results", "evaluation", "limitations"]
            elif self.personality == AgentPersonality.INNOVATIVE_INSIGHTER:
                targets = ["approach", "innovation", "novelty", "contributions", "future"]
            elif self.personality == AgentPersonality.PRACTICAL_APPLICATOR:
                targets = ["applications", "experiments", "implementation", "deployment"]
            else:  # THEORETICAL_INTEGRATOR
                targets = ["related work", "background", "theory", "framework", "conclusion"]

            # Use pattern matching to find sections
            for target in targets:
                for section in available_sections:
                    if section not in defaults:
                        section_lower = section.lower()
                        target_lower = target.lower()
                        if (target_lower in section_lower or
                            section_lower in target_lower or
                            any(word in section_lower for word in target_lower.split())):
                            defaults.append(section)
                            break

            return defaults[:5] if defaults else available_sections[:3]

        except Exception as e:
            self.logger.warning(f"Enhanced section matching failed, using fallback approach: {e}")

            # Fallback to original approach
            defaults = []
            for section in available_sections:
                section_lower = section.lower()
                if "abstract" in section_lower or "introduction" in section_lower:
                    defaults.append(section)

            # Add personality-specific defaults
            if self.personality == AgentPersonality.CRITICAL_EVALUATOR:
                keywords = ["method", "result", "experiment", "evaluation", "limitation"]
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

    def _get_section_content(self, document: Document, section_names: List[str]) -> Dict[str, str]:
        """Get the actual content of selected sections using unified section handling."""
        try:
            # Use unified personality-based section selection
            personality_sections = document.get_sections_for_personality(
                personality_type=self.personality.value,
                max_sections=len(section_names) + 2,  # Allow a few extra sections
                max_chars_per_section=3000,  # Match current limit
                include_fallback=True,
            )

            content_dict = {}

            # Start with personality-selected sections
            for section_name, content in personality_sections:
                content_dict[section_name] = content

            # Add any explicitly requested sections that weren't included
            for section_name in section_names:
                if section_name not in content_dict:
                    chunks = document.get_sections_by_name([section_name])
                    if chunks:
                        # Combine all chunks for this section
                        section_text = "\n\n".join(chunk.content for chunk in chunks)
                        # Limit content length to avoid token overflow
                        if len(section_text) > 3000:
                            section_text = section_text[:3000] + "\n... (content truncated)"
                        content_dict[section_name] = section_text

            return content_dict

        except Exception as e:
            self.logger.warning(f"Unified personality section selection failed, falling back to legacy approach: {e}")

            # Fallback to original approach
            content_dict = {}

            for section_name in section_names:
                chunks = document.get_sections_by_name([section_name])
                if chunks:
                    # Combine all chunks for this section
                    section_text = "\n\n".join(chunk.content for chunk in chunks)
                    # Limit content length to avoid token overflow
                    if len(section_text) > 3000:
                        section_text = section_text[:3000] + "\n... (content truncated)"
                    content_dict[section_name] = section_text

            return content_dict

    async def ask_question(
        self,
        target_insight: AgentInsight,
        target_agent: AgentPersonality,
        discussion_context: Dict[str, Any],
    ) -> Optional[Any]:
        """Ask a question about another agent's insight."""
        try:
            self.logger.info(f"{self.personality.value} asking question to {target_agent.value}")

            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, review the following insight from {target_agent.value.replace("_", " ").title()} and decide whether a follow-up question is truly necessary.

**Target Insight:**
{target_insight.content}
**Importance Score:** {target_insight.importance_score}
**Confidence:** {target_insight.confidence}
**Supporting Evidence:** {", ".join(target_insight.supporting_evidence)}

**Discussion Context:**
Current Phase: {discussion_context.get("phase", "questioning")}
Questions Asked So Far: {discussion_context.get("total_questions", 0)}

**Before deciding to ask anything, reflect on:**
- Does this insight contain unresolved risks, contradictions, or missing evidence from your perspective?
- Would asking a question materially change the shared understanding or convergence?
- Have similar questions already been asked in this discussion?

Only ask a question if it will meaningfully advance the discussion. If the insight already satisfies your concerns, choose to skip.

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
            parsed = self._parse_question_response(result.output, target_insight, target_agent)

            if not parsed:
                return None

            if parsed.get("decision") == "skip":
                self.logger.info(
                    f"{self.personality.value} chose to skip questioning {target_agent.value}: {parsed.get('reason', 'no reason provided')}"
                )
                return parsed

            question_obj = parsed.get("question")
            if question_obj:
                self.logger.debug(f"Generated question from {self.personality.value} to {target_agent.value}")
            return question_obj

        except Exception as e:
            self.logger.error(f"Error asking question from {self.personality.value}: {e}")
            return None

    async def answer_question(
        self,
        question: Question,
        my_insights: List[AgentInsight],
        discussion_context: Dict[str, Any],
    ) -> Optional[Response]:
        """Answer a question from another agent."""
        try:
            # Handle from_agent which might be string or enum
            from_agent_str = question.from_agent if isinstance(question.from_agent, str) else question.from_agent.value

            self.logger.info(f"{self.personality.value} answering question from {from_agent_str}")

            # Find relevant insights
            relevant_insights = [
                insight
                for insight in my_insights
                if insight.content.lower() in question.content.lower()
                or any(evidence.lower() in question.content.lower() for evidence in insight.supporting_evidence)
            ]

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
                self.logger.info(f"Generated response from {self.personality.value}")
            return parsed

        except Exception as e:
            self.logger.error(f"Error answering question for {self.personality.value}: {e}")
            return None

    async def evaluate_convergence(
        self,
        all_insights: List[AgentInsight],
        all_questions: List[Question],
        all_responses: List[Response],
        discussion_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate if the discussion has reached convergence."""
        try:
            self.logger.debug(f"{self.personality.value} evaluating convergence")

            prompt = f"""
As a {self.personality.value.replace("_", " ").title()}, evaluate whether the discussion has reached sufficient convergence:

**Current State:**
- Total Insights: {len(all_insights)}
- Total Questions: {len(all_questions)}
- Total Responses: {len(all_responses)}
- Current Iteration: {discussion_context.get("iteration", 1)}

**Recent Insights from All Agents:**
{chr(10).join(f"{insight.agent_id}: {insight.content}" for insight in all_insights[-10:])}

**Evaluation Criteria:**
1. Are insights becoming more consistent and aligned?
2. Are major disagreements being resolved?
3. Are questions and answers leading to deeper understanding?
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

            self.logger.info(f"{self.personality.value} convergence evaluation: {evaluation.get('convergence_score', 0.0)}")
            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating convergence for {self.personality.value}: {e}")
            return {"convergence_score": 0.5, "continue_discussion": True}

    def _parse_insights_response(self, response: str, document: Document) -> List[AgentInsight]:
        """Parse the agent's response to extract AgentInsight objects."""
        insights = []

        try:
            # Try to extract structured information
            lines = response.split("\n")
            current_insight = {}
            insight_count = 0

            for line in lines:
                line = line.strip()
                if line.lower().startswith("insight:") or line.lower().startswith("finding:"):
                    if current_insight and "content" in current_insight:
                        # Save previous insight
                        insights.append(self._create_insight_from_dict(current_insight, document))
                        insight_count += 1
                        if insight_count >= 3:  # Limit to 3 insights
                            break

                    current_insight = {"content": line.split(":", 1)[1].strip()}

                elif line.lower().startswith("importance:"):
                    try:
                        current_insight["importance_score"] = float(line.split(":", 1)[1].strip())
                    except:
                        current_insight["importance_score"] = 0.5

                elif line.lower().startswith("confidence:"):
                    try:
                        current_insight["confidence"] = float(line.split(":", 1)[1].strip())
                    except:
                        current_insight["confidence"] = 0.5

                elif line.lower().startswith("evidence:"):
                    current_insight["evidence"] = line.split(":", 1)[1].strip()

                elif line.lower().startswith("questions:"):
                    current_insight["questions"] = line.split(":", 1)[1].strip()

            # Add the last insight
            if current_insight and "content" in current_insight:
                insights.append(self._create_insight_from_dict(current_insight, document))

            # If no structured insights found, create from full response
            if not insights:
                insights.append(
                    AgentInsight(
                        agent_id=self.personality,
                        content=response[:500],  # Limit length
                        importance_score=0.5,
                        confidence=0.5,
                        supporting_evidence=[],
                        related_sections=document.get_section_names()[:3],
                    )
                )

        except Exception as e:
            self.logger.error(f"Error parsing insights response: {e}")
            # Fallback: create simple insight
            insights.append(
                AgentInsight(
                    agent_id=self.personality,
                    content=response[:500],
                    importance_score=0.5,
                    confidence=0.5,
                    supporting_evidence=[],
                    related_sections=[],
                )
            )

        return insights

    def _create_insight_from_dict(self, insight_dict: Dict[str, Any], document: Document) -> AgentInsight:
        """Create an AgentInsight from a dictionary."""
        return AgentInsight(
            agent_id=self.personality,
            content=insight_dict.get("content", ""),
            importance_score=insight_dict.get("importance_score", 0.5),
            confidence=insight_dict.get("confidence", 0.5),
            supporting_evidence=([insight_dict.get("evidence", "")] if insight_dict.get("evidence") else []),
            related_sections=document.get_section_names()[:3],
            questions_raised=([insight_dict.get("questions", "")] if insight_dict.get("questions") else []),
        )

    def _parse_question_response(
        self,
        response: str,
        target_insight: AgentInsight,
        target_agent: AgentPersonality,
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM output into a structured decision about questioning."""
        try:
            fields: Dict[str, str] = {}
            current_key: Optional[str] = None

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

    def _parse_answer_response(self, response: str, question: Question) -> Optional[Response]:
        """Parse response to create a Response object."""
        try:
            response_match = re.search(r"Response:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            stance_match = re.search(r"Stance:\s*(\w+)", response, re.IGNORECASE)
            revised_match = re.search(r"Revised Insight:\s*(.+)", response, re.IGNORECASE)
            confidence_match = re.search(r"Confidence:\s*([0-9.]+)", response, re.IGNORECASE)

            if response_match:
                return Response(
                    response_id=str(uuid.uuid4()),
                    question_id=question.question_id,
                    from_agent=self.personality,
                    content=response_match.group(1).strip(),
                    stance=stance_match.group(1).strip() if stance_match else "clarify",
                    revised_insight=(
                        revised_match.group(1).strip() if revised_match and revised_match.group(1).strip().lower() != "none" else None
                    ),
                    confidence=(float(confidence_match.group(1)) if confidence_match else 0.5),
                )
        except Exception as e:
            self.logger.error(f"Error parsing answer response: {e}")
            return None

    def _parse_convergence_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse response to extract convergence evaluation."""
        try:
            evaluation = {
                "convergence_score": 0.5,
                "continue_discussion": True,
                "key_issues": [],
                "recommendations": [],
            }

            score_match = re.search(r"Convergence Score:\s*([0-9.]+)", response, re.IGNORECASE)
            continue_match = re.search(r"Continue Discussion:\s*(\w+)", response, re.IGNORECASE)
            issues_match = re.search(r"Key Issues Remaining:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            rec_match = re.search(r"Recommendations:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

            if score_match:
                evaluation["convergence_score"] = float(score_match.group(1))
            if continue_match:
                evaluation["continue_discussion"] = continue_match.group(1).lower().startswith("y")
            if issues_match:
                evaluation["key_issues"] = [issue.strip() for issue in issues_match.group(1).split("\n") if issue.strip()]
            if rec_match:
                evaluation["recommendations"] = [rec.strip() for rec in rec_match.group(1).split("\n") if rec.strip()]

            return evaluation
        except Exception as e:
            self.logger.error(f"Error parsing convergence evaluation: {e}")
            return {"convergence_score": 0.5, "continue_discussion": True}


# Factory function to create personality agents
def create_personality_agent(personality: AgentPersonality, model_name: str = "deepseek-chat") -> PersonalityAgent:
    """Create a personality agent of the specified type."""
    return PersonalityAgent(personality, model_name)
