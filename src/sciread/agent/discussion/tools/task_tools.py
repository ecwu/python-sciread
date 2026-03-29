"""Task execution tools for multi-agent discussion system."""

from datetime import UTC
from datetime import datetime
from typing import Any

from ....logging_config import get_logger
from ..models import AgentPersonality
from ..personalities import PersonalityAgent
from ..personalities import create_personality_agent
from ..task_models import Task
from ..task_models import TaskResult
from ..task_models import TaskType

logger = get_logger(__name__)

# Global cache for personality agents to maintain chat history
AGENT_CACHE: dict[str, PersonalityAgent] = {}


def get_cached_agent(personality: AgentPersonality, model_name: str = "deepseek-chat") -> PersonalityAgent:
    """Get cached agent instance or create new one if not exists."""
    if personality is None:
        raise ValueError("Personality cannot be None")

    cache_key = f"{personality.value}_{model_name}"

    if cache_key not in AGENT_CACHE:
        logger.debug(f"Creating new agent instance for {personality.value} with model {model_name}")
        AGENT_CACHE[cache_key] = create_personality_agent(personality, model_name)
    else:
        logger.debug(f"Reusing cached agent for {personality.value} with model {model_name}")

    return AGENT_CACHE[cache_key]


def clear_agent_cache():
    """Clear all cached agent instances."""
    global AGENT_CACHE
    AGENT_CACHE.clear()
    logger.info("Agent cache cleared")


def get_agent_cache_status() -> dict[str, Any]:
    """Get current agent cache status for debugging."""
    return {
        "cache_size": len(AGENT_CACHE),
        "cached_agents": list(AGENT_CACHE.keys()),
    }


async def generate_insights_tool(task: Task) -> TaskResult:
    """Tool for generating insights from a personality agent."""
    start_time = datetime.now(UTC)

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

        agent = get_cached_agent(personality, task.context.get("model_name", "deepseek-chat"))

        # Generate insights
        insights = await agent.generate_insights(document, discussion_context)

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        logger.debug(f"Generated {len(insights)} insights for {personality.value}")

        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=execution_time,
            insights=insights,
            confidence=(sum(insight.confidence for insight in insights) / len(insights) if insights else 0.0),
            metadata={
                "personality": personality.value,
                "insights_count": len(insights),
                "average_importance": (sum(insight.importance_score for insight in insights) / len(insights) if insights else 0.0),
            },
        )

    except Exception as e:
        execution_time = (datetime.now(UTC) - start_time).total_seconds()
        error_msg = f"Insight generation failed: {e!s}"
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
    """Tool for asking questions between agents (batch-aware)."""
    start_time = datetime.now(UTC)

    try:
        # Extract parameters
        from_agent = task.parameters.get("from_agent")
        target_insights = task.parameters.get("target_insights", [])
        discussion_context = task.parameters.get("discussion_context", {})

        if not from_agent or not target_insights:
            # Fallback for old single-question tasks if they exist
            target_insight = task.parameters.get("target_insight")
            if target_insight:
                target_insights = [target_insight]
            else:
                raise ValueError("Missing required parameters: from_agent and target_insights")

        # Convert to proper types
        if isinstance(from_agent, str):
            from_agent = AgentPersonality(from_agent)

        # Create agent
        agent = get_cached_agent(from_agent, task.context.get("model_name", "deepseek-chat"))

        # Ask questions in batch
        questions = await agent.ask_questions_batch(target_insights, discussion_context)

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        logger.debug(f"Generated {len(questions)} questions from {from_agent.value}")

        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=execution_time,
            questions=questions,
            confidence=(sum(q.priority for q in questions) / len(questions) if questions else 0.0),
            metadata={
                "from_agent": from_agent.value,
                "questions_count": len(questions),
            },
        )

    except Exception as e:
        execution_time = (datetime.now(UTC) - start_time).total_seconds()
        error_msg = f"Question generation failed: {e!s}"
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
    """Tool for answering questions from other agents (batch-aware)."""
    start_time = datetime.now(UTC)

    try:
        # Extract parameters
        questions = task.parameters.get("questions", [])
        my_insights = task.parameters.get("my_insights", [])
        discussion_context = task.parameters.get("discussion_context", {})
        assigned_to = task.assigned_to

        if not questions:
            # Fallback for old single-answer tasks
            question = task.parameters.get("question")
            if question:
                questions = [question]
            else:
                raise ValueError("Missing required parameter: questions")

        if not assigned_to:
            # Extract personality from first question's to_agent if not assigned
            personality_str = questions[0].to_agent
            personality = AgentPersonality(personality_str) if isinstance(personality_str, str) else personality_str
        else:
            personality = AgentPersonality(assigned_to) if isinstance(assigned_to, str) else assigned_to

        # Create agent
        agent = get_cached_agent(personality, task.context.get("model_name", "deepseek-chat"))

        # Answer questions in batch
        responses = await agent.answer_questions_batch(questions, my_insights, discussion_context)

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        logger.debug(f"Generated {len(responses)} responses from {personality.value}")

        return TaskResult(
            task_id=task.task_id,
            success=True,
            execution_time=execution_time,
            responses=responses,
            confidence=(sum(r.confidence for r in responses) / len(responses) if responses else 0.0),
            metadata={
                "personality": personality.value,
                "responses_count": len(responses),
            },
        )

    except Exception as e:
        execution_time = (datetime.now(UTC) - start_time).total_seconds()
        error_msg = f"Answer generation failed: {e!s}"
        logger.error(error_msg)

        return TaskResult(
            task_id=task.task_id,
            success=False,
            execution_time=execution_time,
            error_message=error_msg,
            confidence=0.0,
            metadata={"personality": str(personality) if "personality" in locals() else "unknown"},
        )


async def evaluate_convergence_tool(task: Task) -> TaskResult:
    """Tool for evaluating discussion convergence."""
    start_time = datetime.now(UTC)

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
        agent = get_cached_agent(personality, task.context.get("model_name", "deepseek-chat"))

        # Evaluate convergence
        evaluation = await agent.evaluate_convergence(all_insights, all_questions, all_responses, discussion_context)

        execution_time = (datetime.now(UTC) - start_time).total_seconds()

        logger.debug(f"{personality.value} convergence evaluation: {evaluation.get('convergence_score', 0.0)}")

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
        execution_time = (datetime.now(UTC) - start_time).total_seconds()
        error_msg = f"Convergence evaluation failed: {e!s}"
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
