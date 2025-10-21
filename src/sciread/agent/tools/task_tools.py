"""Task execution tools for multi-agent discussion system."""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..personality_agents import PersonalityAgent, create_personality_agent
from ...document import Document
from ...logging_config import get_logger
from ..models.discussion_models import (
    AgentPersonality,
    AgentInsight,
    Question,
    Response,
)
from ..models.task_models import Task, TaskResult, TaskType

logger = get_logger(__name__)


async def generate_insights_tool(task: Task) -> TaskResult:
    """Tool for generating insights from a personality agent."""
    start_time = datetime.now()

    try:
        # Extract parameters
        personality = task.parameters.get("personality")
        document = task.parameters.get("document")
        discussion_context = task.parameters.get("discussion_context", {})

        if not personality or not document:
            raise ValueError("Missing required parameters: personality and document")

        # Create agent if not provided
        if isinstance(personality, str):
            personality = AgentPersonality(personality)

        agent = create_personality_agent(
            personality, task.context.get("model_name", "deepseek-chat")
        )

        # Generate insights
        insights = await agent.generate_insights(document, discussion_context)

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Generated {len(insights)} insights for {personality.value}")

        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=execution_time,
            insights=insights,
            confidence=(
                sum(insight.confidence for insight in insights) / len(insights)
                if insights
                else 0.0
            ),
            metadata={
                "personality": personality.value,
                "insights_count": len(insights),
                "average_importance": (
                    sum(insight.importance_score for insight in insights)
                    / len(insights)
                    if insights
                    else 0.0
                ),
            },
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Insight generation failed: {str(e)}"
        logger.error(error_msg)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            execution_time=execution_time,
            error_message=error_msg,
            confidence=0.0,
            metadata={"personality": str(personality) if personality else "unknown"},
        )


async def ask_question_tool(task: Task) -> TaskResult:
    """Tool for asking questions between agents."""
    start_time = datetime.now()

    try:
        # Extract parameters
        from_agent = task.parameters.get("from_agent")
        to_agent = task.parameters.get("to_agent")
        target_insight = task.parameters.get("target_insight")
        discussion_context = task.parameters.get("discussion_context", {})

        if not all([from_agent, to_agent, target_insight]):
            raise ValueError(
                "Missing required parameters: from_agent, to_agent, target_insight"
            )

        # Convert to proper types
        if isinstance(from_agent, str):
            from_agent = AgentPersonality(from_agent)
        if isinstance(to_agent, str):
            to_agent = AgentPersonality(to_agent)

        # Create agent
        agent = create_personality_agent(
            from_agent, task.context.get("model_name", "deepseek-chat")
        )

        # Ask question
        question = await agent.ask_question(
            target_insight, to_agent, discussion_context
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        if question:
            logger.info(
                f"Generated question from {from_agent.value} to {to_agent.value}"
            )
            return TaskResult(
                task_id=task.task_id,
                success=True,
                execution_time=execution_time,
                questions=[question],
                confidence=question.priority,
                metadata={
                    "from_agent": from_agent.value,
                    "to_agent": to_agent.value,
                    "question_type": question.question_type,
                    "priority": question.priority,
                },
            )
        else:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                error_message="Failed to generate question",
                confidence=0.0,
                metadata={"from_agent": from_agent.value, "to_agent": to_agent.value},
            )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Question generation failed: {str(e)}"
        logger.error(error_msg)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            execution_time=execution_time,
            error_message=error_msg,
            confidence=0.0,
            metadata={"from_agent": str(from_agent) if from_agent else "unknown"},
        )


async def answer_question_tool(task: Task) -> TaskResult:
    """Tool for answering questions from other agents."""
    start_time = datetime.now()

    try:
        # Extract parameters
        question = task.parameters.get("question")
        my_insights = task.parameters.get("my_insights", [])
        discussion_context = task.parameters.get("discussion_context", {})

        if not question:
            raise ValueError("Missing required parameter: question")

        # Extract personality from question's to_agent
        # Note: question.to_agent might be a string due to use_enum_values=True
        personality = question.to_agent
        if isinstance(personality, str):
            personality = AgentPersonality(personality)

        # Create agent
        agent = create_personality_agent(
            personality, task.context.get("model_name", "deepseek-chat")
        )

        # Answer question
        response = await agent.answer_question(
            question, my_insights, discussion_context
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        if response:
            logger.info(f"Generated response from {personality.value}")
            return TaskResult(
                task_id=task.task_id,
                success=True,
                execution_time=execution_time,
                responses=[response],
                confidence=response.confidence,
                metadata={
                    "personality": personality.value,
                    "stance": response.stance,
                    "has_revised_insight": response.revised_insight is not None,
                },
            )
        else:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                execution_time=execution_time,
                error_message="Failed to generate response",
                confidence=0.0,
                metadata={"personality": personality.value},
            )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Answer generation failed: {str(e)}"
        logger.error(error_msg)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            execution_time=execution_time,
            error_message=error_msg,
            confidence=0.0,
            metadata={"personality": str(personality) if personality else "unknown"},
        )


async def evaluate_convergence_tool(task: Task) -> TaskResult:
    """Tool for evaluating discussion convergence."""
    start_time = datetime.now()

    try:
        # Extract parameters
        personality = task.parameters.get("personality")
        all_insights = task.parameters.get("all_insights", [])
        all_questions = task.parameters.get("all_questions", [])
        all_responses = task.parameters.get("all_responses", [])
        discussion_context = task.parameters.get("discussion_context", {})

        if not personality:
            raise ValueError("Missing required parameter: personality")

        # Convert to proper type
        if isinstance(personality, str):
            personality = AgentPersonality(personality)

        # Create agent
        agent = create_personality_agent(
            personality, task.context.get("model_name", "deepseek-chat")
        )

        # Evaluate convergence
        evaluation = await agent.evaluate_convergence(
            all_insights, all_questions, all_responses, discussion_context
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"{personality.value} convergence evaluation: {evaluation.get('convergence_score', 0.0)}"
        )

        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=execution_time,
            analysis_result=f"Convergence score: {evaluation.get('convergence_score', 0.0)}",
            confidence=evaluation.get("convergence_score", 0.5),
            metadata={
                "personality": personality.value,
                "convergence_score": evaluation.get("convergence_score", 0.0),
                "continue_discussion": evaluation.get("continue_discussion", True),
                "key_issues": evaluation.get("key_issues", []),
                "recommendations": evaluation.get("recommendations", []),
            },
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Convergence evaluation failed: {str(e)}"
        logger.error(error_msg)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            execution_time=execution_time,
            error_message=error_msg,
            confidence=0.0,
            metadata={"personality": str(personality) if personality else "unknown"},
        )


# Registry of task tools
TASK_TOOLS = {
    TaskType.GENERATE_INSIGHTS: generate_insights_tool,
    TaskType.ASK_QUESTION: ask_question_tool,
    TaskType.ANSWER_QUESTION: answer_question_tool,
    TaskType.EVALUATE_RESPONSE: evaluate_convergence_tool,
    TaskType.MONITOR_CONVERGENCE: evaluate_convergence_tool,
}


def get_task_tool(task_type: TaskType):
    """Get the appropriate tool function for a task type."""
    return TASK_TOOLS.get(task_type)
