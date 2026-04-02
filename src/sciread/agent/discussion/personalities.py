"""Personality-based agents for multi-agent discussion system."""

import re
import uuid
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ...document.document_renderers import choose_best_section_match
from ...document.document_renderers import format_section_choices
from ...document.document_renderers import get_section_length_map
from ...document.document_renderers import get_sections_content
from ...document.document_renderers import is_likely_heading_only
from ...document_structure import Document
from ...llm_provider import get_model
from ...platform.logging import get_logger
from .models import AGENT_ABBREVIATIONS
from .models import AgentInsight
from .models import AgentPersonality
from .models import Question
from .models import Response
from .prompts.personalities import build_insight_generation_prompt
from .prompts.personalities import get_personality_system_prompt

logger = get_logger(__name__)


PERSONALITY_DISPLAY_NAMES = {
    AgentPersonality.CRITICAL_EVALUATOR: "批判性评估者",
    AgentPersonality.INNOVATIVE_INSIGHTER: "创新洞察者",
    AgentPersonality.PRACTICAL_APPLICATOR: "实践应用者",
    AgentPersonality.THEORETICAL_INTEGRATOR: "理论整合者",
}


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
        self.abbrev = AGENT_ABBREVIATIONS.get(personality, "XX")

    def _display_name(self, personality: AgentPersonality | str | None = None) -> str:
        """Return a Chinese display name for a personality."""
        target = personality or self.personality
        if isinstance(target, AgentPersonality):
            return PERSONALITY_DISPLAY_NAMES.get(target, target.value)

        try:
            enum_value = AgentPersonality(target)
            return PERSONALITY_DISPLAY_NAMES.get(enum_value, enum_value.value)
        except Exception:
            return str(target)

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

    async def generate_insights(self, document: Document, discussion_context: dict[str, Any]) -> list[AgentInsight]:
        """Generate insights based on document and personality."""
        try:
            self.logger.debug(f"Generating insights for {self.personality.value}")

            # Get abstract from document
            abstract_chunks = document.get_sections_by_name(["abstract"])
            abstract_text = " ".join(chunk.content for chunk in abstract_chunks) if abstract_chunks else "暂无摘要"

            # Step 1: Let agent select which sections to read based on personality
            section_names = document.get_section_names()
            section_lengths = get_section_length_map(document, section_names)
            selected_sections = await self._select_sections_to_read(
                document.metadata.title or "未命名论文",
                abstract_text,
                section_names,
                section_lengths,
            )

            self.logger.debug(f"{self.personality.value} selected sections: {selected_sections}")

            # Step 2: Get content of selected sections
            selected_content = self._get_section_content(document, selected_sections)

            # Step 3: Build the prompt with actual content
            prompt = build_insight_generation_prompt(
                personality=self.personality,
                document_title=document.metadata.title or "未命名论文",
                document_abstract=abstract_text,
                key_sections=section_names,
                selected_sections_content=selected_content,
                discussion_context=discussion_context,
            )

            # Execute the agent with history
            result = await self._run_with_history(prompt)

            # Parse the response to extract insights
            insights = self._parse_insights_response(result.output, document)

            # Assign short IDs if not present
            for i, insight in enumerate(insights):
                if not getattr(insight, "insight_id", None):
                    insight.insight_id = f"INS-{self.abbrev}-{i + 1:02d}"

            self.logger.debug(f"Generated {len(insights)} insights for {self.personality.value}")
            return insights

        except Exception as e:
            self.logger.error(f"Error generating insights for {self.personality.value}: {e}")
            return []

    async def ask_questions_batch(
        self,
        target_insights: list[AgentInsight],
        discussion_context: dict[str, Any],
    ) -> list[Question]:
        """Ask multiple questions about other agents' insights in one call."""
        # Safety limit to avoid context window issues
        target_insights = target_insights[:12]
        try:
            self.logger.debug(f"{self.personality.value} reviewing {len(target_insights)} insights for questioning")

            insights_text = ""
            for insight in target_insights:
                author_name = self._display_name(insight.agent_id)
                insights_text += f"\n[{insight.insight_id}] 来自 {author_name}（重要性：{insight.importance_score}）：\n"
                insights_text += f'  "{insight.content}"\n'
                if insight.supporting_evidence:
                    insights_text += f'  证据："{insight.supporting_evidence[0][:200]}..."\n'

            qa_summary = self._format_role_qa_summary(discussion_context)

            prompt = f"""
请你以{self._display_name()}的视角，审阅以下来自其他智能体的洞见，并判断哪些值得继续追问。

**待审阅洞见：**
{insights_text}

{qa_summary}

**讨论上下文：**
当前阶段：{discussion_context.get("phase", "questioning")}
当前轮次：{discussion_context.get("iteration", 1)}

**你的任务：**
针对每条洞见，判断是否需要提出具体的追问、质疑或澄清问题。
只有当提问能实质性推进讨论时才选择提问；如果该洞见已经回应了你的关注点，或者你之前的问题已经覆盖了该主题，请选择跳过。

**输出要求：**
- 原因说明和问题内容请使用中文。
- `Question about [...]`、`Decision:`、`Reason:`、`Question:`、`Priority:`、`Type:` 这些字段标签必须保留英文。
- `Decision` 只能使用 `ask` 或 `skip`；`Type` 只能使用 `clarification`、`challenge`、`extension` 或 `none`。

对每条洞见，都必须使用如下格式：
---
Question about [INS-XX-01]:
Decision: [ask|skip]
Reason: [请用中文简要说明理由]
Question: [请用中文写出具体问题；若跳过则填写 "None"]
Priority: [0.0-1.0，若跳过则填写 0.0]
Type: [clarification/challenge/extension/none]
---

上方列出的每条洞见都必须输出一个完整区块。
"""

            result = await self._run_with_history(prompt)
            questions = self._parse_batch_questions_response(result.output, target_insights)

            self.logger.debug(f"{self.personality.value} generated {len(questions)} questions in batch")
            return questions

        except Exception as e:
            self.logger.error(f"Error in batch questioning for {self.personality.value}: {e}")
            return []

    async def answer_questions_batch(
        self,
        questions: list[Question],
        my_insights: list[AgentInsight],
        discussion_context: dict[str, Any],
    ) -> list[Response]:
        """Answer multiple questions directed at this agent in one call."""
        try:
            self.logger.debug(f"{self.personality.value} answering {len(questions)} questions in batch")

            questions_text = ""
            for q in questions:
                from_name = self._display_name(q.from_agent)
                questions_text += f"\n[{q.question_id}] 来自 {from_name}，针对洞见“{q.target_insight}”：\n"
                questions_text += f'  "{q.content}"\n'

            insights_text = "\n".join([f"- [{i.insight_id}] {i.content}" for i in my_insights])
            qa_summary = self._format_role_qa_summary(discussion_context)

            prompt = f"""
请回答以下问题：
{questions_text}

**与你相关的洞见：**
{insights_text}

{qa_summary}

**讨论上下文：**
当前阶段：{discussion_context.get("phase", "responding")}
当前轮次：{discussion_context.get("iteration", 1)}

**你的任务：**
请逐条作答。回答时要保持你的角色视角，给出清晰推理；如有必要，可以修订自己的洞见。

**输出要求：**
- `Answer to [...]`、`Response:`、`Stance:`、`Revised Insight:`、`Confidence:` 这些字段标签必须保留英文。
- 回答正文与修订内容请使用中文。
- `Stance` 必须使用 `agree`、`disagree`、`clarify` 或 `modify` 之一。

每个问题都必须使用以下精确格式：
---
Answer to [Q-XX-01]:
Response: [请用中文写出详细回答]
Stance: [agree/disagree/clarify/modify]
Revised Insight: [若需修订，请用中文写出修订后的洞见；否则填写 "None"]
Confidence: [0.0-1.0]
---

上方列出的每个问题都必须输出一个完整区块。
"""

            result = await self._run_with_history(prompt)
            responses = self._parse_batch_responses_response(result.output, questions)

            self.logger.debug(f"{self.personality.value} generated {len(responses)} responses in batch")
            return responses

        except Exception as e:
            self.logger.error(f"Error in batch answering for {self.personality.value}: {e}")
            return []

    def _format_role_qa_summary(self, discussion_context: dict[str, Any]) -> str:
        """Format a summary of Q&A involving this agent's role."""
        qa_summary = discussion_context.get("role_qa_summary", "")
        if qa_summary:
            return f"**与你相关的问答（{self._display_name()}）：**\n{qa_summary}"
        return ""

    def _parse_batch_questions_response(self, response: str, target_insights: list[AgentInsight]) -> list[Question]:
        """Parse multiple question blocks from LLM response."""
        questions = []
        blocks = re.split(r"---", response)

        # Map insight_id to insight object for easy lookup
        insight_map = {i.insight_id: i for i in target_insights}

        for block in blocks:
            if "Question about [" not in block:
                continue

            try:
                insight_id_match = re.search(r"Question about \[(INS-[A-Z]+-\d+)\]", block)
                if not insight_id_match:
                    continue

                insight_id = insight_id_match.group(1)
                target_insight = insight_map.get(insight_id)

                if not target_insight:
                    continue

                # Use existing single parser logic but adapted for the block
                parsed = self._parse_question_response(block, target_insight, target_insight.agent_id)
                if parsed and parsed.get("decision") == "ask":
                    q = parsed["question"]
                    # Short ID will be assigned by orchestrator/generator,
                    # but we can set a temporary one if needed.
                    # Actually, the orchestrator should use QuestionIdGenerator.
                    questions.append(q)

            except Exception as e:
                self.logger.warning(f"Failed to parse question block: {e}")

        return questions

    def _parse_batch_responses_response(self, response: str, questions: list[Question]) -> list[Response]:
        """Parse multiple answer blocks from LLM response."""
        responses = []
        blocks = re.split(r"---", response)

        # Map question_id to question object
        question_map = {q.question_id: q for q in questions}

        for block in blocks:
            if "Answer to [" not in block:
                continue

            try:
                question_id_match = re.search(r"Answer to \[(Q-[A-Z]+-\d+)\]", block)
                if not question_id_match:
                    continue

                question_id = question_id_match.group(1)
                question = question_map.get(question_id)

                if not question:
                    continue

                parsed = self._parse_answer_response(block, question)
                if parsed:
                    responses.append(parsed)

            except Exception as e:
                self.logger.warning(f"Failed to parse answer block: {e}")

        return responses

    async def _select_sections_to_read(
        self,
        title: str,
        abstract: str,
        available_sections: list[str],
        section_lengths: dict[str, int],
    ) -> list[str]:
        """Select which sections to read based on personality and paper overview."""
        try:
            prompt = f"""
请你以{self._display_name()}的视角，选择这篇论文中最值得重点阅读的章节。

**论文标题：** {title}
**摘要：** {abstract}

**可选章节（格式：章节名 | 正文长度）：**
{format_section_choices(available_sections, section_lengths, numbered=True)}

**你的任务：**
请依据你的分析重点，从中选择 3-5 个最相关的章节：
- 批判性评估者：重点关注方法、结果、局限
- 创新洞察者：重点关注新方法、创新点、未来工作
- 实践应用者：重点关注应用、实验、真实世界影响
- 理论整合者：重点关注理论框架、相关工作、结论

补充规则：
- 每个章节后的 `chars` 表示该 section 的正文长度。
- 若某个章节标注“可能仅标题”，通常说明它可能只有标题或过渡句，真正内容在更下一级子章节。

请只返回你要阅读的章节名，每行一个，并且必须与上方列表中的章节名完全一致。
不要附加 `chars`、编号或注释。
请选择那些最有助于你从自身视角产出高价值洞见的章节。
"""

            result = await self._run_with_history(prompt)

            # Parse section names from response
            selected = []
            response_lines = result.output.strip().split("\n")

            for line in response_lines:
                line = line.strip().strip("-").strip("*").strip()
                # Remove numbering if present
                line = line.split(".", 1)[-1].strip() if "." in line else line
                # Remove appended metadata such as "| 120 chars" or "(120 chars)"
                if "|" in line:
                    line = line.split("|", 1)[0].strip()
                if " (" in line and line.endswith(")"):
                    line = line.rsplit(" (", 1)[0].strip()

                # Match against available sections (case-insensitive, flexible matching)
                for section in available_sections:
                    if line and (line.lower() == section.lower() or line.lower() in section.lower() or section.lower() in line.lower()):
                        if section not in selected:
                            selected.append(section)
                            break

            # Fallback: if no sections selected or too few, use defaults based on personality
            if len(selected) < 2:
                selected = self._get_default_sections(available_sections, section_lengths)

            return selected[:5]  # Limit to 5 sections max

        except Exception as e:
            self.logger.error(f"Error selecting sections for {self.personality.value}: {e}")
            return self._get_default_sections(available_sections, section_lengths)

    def _get_default_sections(self, available_sections: list[str], section_lengths: dict[str, int]) -> list[str]:
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
                targets = [
                    "methodology",
                    "method",
                    "approach",
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
                section = choose_best_section_match(target, available_sections, section_lengths)
                if section and section not in defaults:
                    defaults.append(section)

            if defaults:
                return defaults[:5]

            non_short_sections = [section for section in available_sections if not is_likely_heading_only(section_lengths.get(section, 0))]
            return non_short_sections[:3] if non_short_sections else available_sections[:3]

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

            if defaults:
                return defaults[:5]

            non_short_sections = [section for section in available_sections if not is_likely_heading_only(section_lengths.get(section, 0))]
            return non_short_sections[:3] if non_short_sections else available_sections[:3]

    def _get_section_content(self, document: Document, section_names: list[str]) -> dict[str, str]:
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
            fallback_names = self._get_default_sections(available, get_section_length_map(document, available))
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
            # Removed early info log to avoid redundancy with later print/log

            insight_author = getattr(target_insight, "agent_id", target_agent)
            author_value = getattr(insight_author, "value", insight_author)
            author_name = self._display_name(author_value)

            prior_qa = discussion_context.get("prior_qa_for_insight", [])
            my_prior_questions = discussion_context.get("my_prior_questions", [])

            prior_qa_text = self._format_prior_qa_for_prompt(prior_qa)
            my_prior_text = self._format_my_prior_questions_for_prompt(my_prior_questions)

            prompt = f"""
请你以{self._display_name()}的视角审阅以下洞见，并判断是否真的有必要继续追问。

**洞见作者：** {author_name}
**提问对象：** {self._display_name(target_agent)}

**目标洞见：**
{target_insight.content}
**重要性评分：** {target_insight.importance_score}
**置信度：** {target_insight.confidence}
**支撑证据：** {", ".join(target_insight.supporting_evidence)}

{prior_qa_text}

{my_prior_text}

**讨论上下文：**
当前阶段：{discussion_context.get("phase", "questioning")}
当前轮次：{discussion_context.get("iteration", 1)}
当前累计提问数：{discussion_context.get("total_questions", 0)}

**在决定是否提问前，请先思考：**
- 从你的视角看，这条洞见是否仍存在未解决的风险、矛盾或证据缺口？
- 你之前的问题是否已经得到满意回答？如果是，就不必重复提问。
- 如果之前的问题已被回答，该回答是否真正回应了你的关切，还是引出了新的问题？
- 再提出一个新问题，是否会实质性改变当前的共同理解或收敛状态？

只有在提问能显著推进讨论时才选择提问。如果该洞见已经回应了你的关切，或者你之前的问题已得到充分回答，请直接跳过。

**输出要求：**
- 理由和问题内容请使用中文。
- `Decision:`、`Reason:`、`Question:`、`Priority:`、`Type:` 这些字段标签必须保留英文。
- `Decision` 只能填写 `ask` 或 `skip`；`Question: None`、`Priority: 0.0`、`Type: none` 表示跳过。

请严格使用以下格式：
```
Decision: [ask|skip]
Reason: [请用中文说明你做出该决定的理由]
Question: [请用中文写出具体问题；若跳过则填写 "None"]
Priority: [0.0-1.0，若跳过则填写 0.0]
Type: [clarification/challenge/extension/none]
```
当你选择 `Decision: skip` 时，必须将 `Question` 设为 `None`、`Priority` 设为 `0.0`、`Type` 设为 `none`。
当你选择 `Decision: ask` 时，请提出一个精准、能推进对话的问题。
"""

            result = await self._run_with_history(prompt)
            parsed = self._parse_question_response(result.output, target_insight, target_agent)

            if not parsed:
                return None

            if parsed.get("decision") == "skip":
                self.logger.debug(f"{self.personality.value} skipped asking question to {target_agent.value}")
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
        my_insights: list[AgentInsight],
        discussion_context: dict[str, Any],
    ) -> Response | None:
        """Answer a question from another agent."""
        try:
            # Handle from_agent which might be string or enum
            from_agent_str = question.from_agent if isinstance(question.from_agent, str) else question.from_agent.value

            self.logger.debug(f"{self.personality.value} answering question from {from_agent_str}")

            relevant_insights = self._find_relevant_insights_for_question(my_insights, question)

            prompt = f"""
请你以{self._display_name()}的视角，回答来自 {self._display_name(from_agent_str)} 的以下问题：

**问题：**
{question.content}

**与你相关的洞见：**
{chr(10).join(f"- {insight.content}" for insight in relevant_insights) if relevant_insights else "暂未检索到可直接对应的洞见，请基于你的角色理解作答。"}

**讨论上下文：**
当前阶段：{discussion_context.get("phase", "responding")}
问题优先级：{question.priority}

**你的任务：**
请给出一个经过思考的回答，并满足以下要求：
1. 直接回应问题本身
2. 保持你的角色视角
3. 给出清晰的推理和依据
4. 如有必要，提出对原洞见的修订

**输出要求：**
- 回答内容请使用中文。
- `Response:`、`Stance:`、`Revised Insight:`、`Confidence:` 这些字段标签必须保留英文。
- `Stance` 必须使用 `agree`、`disagree`、`clarify` 或 `modify`。

请按以下格式输出：
```
Response: [请用中文写出详细回答]
Stance: [agree/disagree/clarify/modify]
Revised Insight: [若需修订，请用中文写出修订后的洞见；否则填写 "None"]
Confidence: [0.0-1.0，表示你对该回答的置信度]
```
"""

            result = await self._run_with_history(prompt)
            parsed = self._parse_answer_response(result.output, question)

            if parsed:
                self.logger.debug(f"Generated response from {self.personality.value}")
            return parsed

        except Exception as e:
            self.logger.error(f"Error answering question for {self.personality.value}: {e}")
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
            my_questions_answered = self._count_my_answered_questions(all_questions, all_responses)

            prompt = f"""
请你以{self._display_name()}的视角，判断当前讨论是否已经达到足够的收敛程度：

**当前状态：**
- 总洞见数：{len(all_insights)}
- 总问题数：{len(all_questions)}
- 总回答数：{len(all_responses)}
- 当前轮次：{discussion_context.get("iteration", 1)}
- 你提出的问题中已获回答：{my_questions_answered["answered"]}/{my_questions_answered["total"]}

**各智能体最近的洞见：**
{chr(10).join(f"{self._display_name(insight.agent_id)}：{insight.content[:150]}..." for insight in all_insights[-8:])}

**最近的问答线程：**
{qa_thread}

**评估标准：**
1. 各方洞见是否正在变得更加一致？
2. 你提出的问题是否得到了回答，且回答是否令人满意？
3. 主要分歧是否正在通过问答被消解？
4. 从你的视角看，继续讨论是否还可能产生重要新洞见？

**输出要求：**
- 解释性内容请使用中文。
- 为了兼容现有解析流程，以下字段标签必须保留英文，`Continue Discussion` 只能填写 `yes` 或 `no`。

请按以下格式作答：
```
Convergence Score: [0.0-1.0]
Continue Discussion: [yes/no]
Key Issues Remaining: [请用中文列出尚未解决的关键问题]
Recommendations: [请用中文给出下一步建议]
```
"""

            result = await self._run_with_history(prompt)
            evaluation = self._parse_convergence_evaluation(result.output)

            self.logger.debug(f"{self.personality.value} convergence evaluation: {evaluation.get('convergence_score', 0.0)}")
            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating convergence for {self.personality.value}: {e}")
            return {"convergence_score": 0.5, "continue_discussion": True}

    def _build_qa_thread_summary(self, all_questions: list[Question], all_responses: list[Response]) -> str:
        """Build a summary of Q&A threads for convergence evaluation."""
        if not all_questions:
            return "（目前还没有提出任何问题）"

        response_map = {r.question_id: r for r in all_responses}
        lines = []

        for q in all_questions[-10:]:
            from_agent = q.from_agent if isinstance(q.from_agent, str) else q.from_agent.value
            to_agent = q.to_agent if isinstance(q.to_agent, str) else q.to_agent.value
            response = response_map.get(q.question_id)

            q_summary = q.content[:100] + "..." if len(q.content) > 100 else q.content
            lines.append(f"问（{self._display_name(from_agent)} → {self._display_name(to_agent)}）：{q_summary}")

            if response:
                r_summary = response.content[:100] + "..." if len(response.content) > 100 else response.content
                lines.append(f"  答（立场：{response.stance}）：{r_summary}")
            else:
                lines.append("  答：（待回复）")

        return "\n".join(lines) if lines else "（暂无问答）"

    def _count_my_answered_questions(self, all_questions: list[Question], all_responses: list[Response]) -> dict[str, int]:
        """Count how many of this agent's questions have been answered."""
        response_ids = {r.question_id for r in all_responses}
        my_questions = [
            q for q in all_questions if (q.from_agent if isinstance(q.from_agent, str) else q.from_agent.value) == self.personality.value
        ]
        answered = sum(1 for q in my_questions if q.question_id in response_ids)
        return {"total": len(my_questions), "answered": answered}

    def _parse_insights_response(self, response: str, document: Document) -> list[AgentInsight]:
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

                if line.lower().startswith("insight:") or line.lower().startswith("finding:"):
                    if current_insight and "content" in current_insight:
                        insights.append(self._create_insight_from_dict(current_insight, document))
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
                            current_insight["importance_score"] = min(max(score, 0.0), 1.0)
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
                        current_insight["evidence"] = f"{current_insight['evidence']} {line}"

            if current_insight and "content" in current_insight:
                insights.append(self._create_insight_from_dict(current_insight, document))

            if insights:
                self.logger.debug(f"Parsed {len(insights)} insights from response")
                for i, insight in enumerate(insights):
                    has_evidence = bool(insight.supporting_evidence and insight.supporting_evidence[0])
                    self.logger.debug(
                        f"  Insight {i + 1}: importance={insight.importance_score}, "
                        f"confidence={insight.confidence}, has_evidence={has_evidence}"
                    )
            else:
                self.logger.warning("No structured insights parsed from response, using fallback")

            if not insights:
                insights.append(
                    AgentInsight(
                        agent_id=self.personality,
                        content=response[:500],
                        importance_score=0.5,
                        confidence=0.5,
                        supporting_evidence=["（未解析出结构化字段，未能提取明确证据）"],
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
                    supporting_evidence=["（解析响应失败，未能提取证据）"],
                    related_sections=[],
                )
            )

        return insights

    def _create_insight_from_dict(self, insight_dict: dict[str, Any], document: Document) -> AgentInsight:
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

    def _format_prior_qa_for_prompt(self, prior_qa: list[dict[str, Any]]) -> str:
        """Format prior Q&A pairs for inclusion in prompt."""
        if not prior_qa:
            return ""

        lines = ["**关于该洞见的既往问答：**"]
        for qa in prior_qa:
            from_agent = self._display_name(qa.get("from_agent", "未知角色"))
            question = qa.get("question", "")
            response = qa.get("response")
            stance = qa.get("response_stance")

            lines.append(f"- 来自 {from_agent} 的问题：{question}")
            if response:
                lines.append(f"  回答：{response[:200]}..." if len(response) > 200 else f"  回答：{response}")
                if stance:
                    lines.append(f"  立场：{stance}")
            else:
                lines.append("  回答：（尚未收到回复）")

        return "\n".join(lines)

    def _format_my_prior_questions_for_prompt(self, my_prior_questions: list[dict[str, Any]]) -> str:
        """Format the agent's own prior questions about this insight."""
        if not my_prior_questions:
            return ""

        lines = ["**你此前围绕该洞见提出的问题：**"]
        for qa in my_prior_questions:
            question = qa.get("question", "")
            response = qa.get("response")
            stance = qa.get("response_stance")

            lines.append(f"- 你的问题：{question}")
            if response:
                lines.append(f"  已收到回复：{response[:200]}..." if len(response) > 200 else f"  已收到回复：{response}")
                if stance:
                    lines.append(f"  对方立场：{stance}")
            else:
                lines.append("  （等待回复中）")

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
                    "reason": reason or "当前无需继续澄清。",
                    "question": None,
                    "priority": 0.0,
                    "type": "none",
                }

            if not question_text:
                return None

            question_obj = Question(
                question_id="PENDING",  # Will be assigned a short ID by orchestrator
                from_agent=self.personality,
                to_agent=target_agent,
                content=question_text,
                target_insight=getattr(target_insight, "insight_id", None) or target_insight.content[:50],
                target_insight_id=getattr(target_insight, "insight_id", None),
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

    def _find_relevant_insights_for_question(self, my_insights: list[AgentInsight], question: Question) -> list[AgentInsight]:
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
                    if evidence_lower in question_lower or question_lower in evidence_lower:
                        relevant_insights.append(insight)
                        break

        return relevant_insights

    def _parse_answer_response(self, response: str, question: Question) -> Response | None:
        """Parse response to create a Response object."""
        try:
            fields: dict[str, str] = {}
            current_key: str | None = None

            for raw_line in response.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    current_key = key.strip().lower().replace(" ", "_")
                    fields[current_key] = value.strip()
                elif current_key:
                    fields[current_key] = f"{fields[current_key]} {line}"

            content = fields.get("response", "").strip() or fields.get("answer", "").strip()

            if content:
                stance = fields.get("stance", "clarify").lower()
                revised = fields.get("revised_insight", "None")
                if not revised or revised.lower() == "none":
                    revised = None

                priority_raw = fields.get("confidence", "0.5")
                try:
                    confidence = float(re.search(r"(\d+\.?\d*)", priority_raw).group(1)) if re.search(r"(\d+\.?\d*)", priority_raw) else 0.5
                except (ValueError, AttributeError):
                    confidence = 0.5

                return Response(
                    response_id=str(uuid.uuid4()),
                    question_id=question.question_id,
                    from_agent=self.personality,
                    content=content,
                    stance=stance,
                    revised_insight=revised,
                    confidence=min(max(confidence, 0.0), 1.0),
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
