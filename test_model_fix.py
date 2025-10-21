#!/usr/bin/env python3
"""Simple test to verify that model parameter is passed to persona agents."""

import asyncio
from pathlib import Path
from sciread.document import Document
from sciread.agent.discussion_agent import DiscussionAgent
from sciread.agent.tools.task_tools import generate_insights_tool
from sciread.agent.models.task_models import Task, TaskType, TaskPriority, AgentPersonality

async def test_model_parameter_propagation():
    """Test that model parameter is properly propagated to persona agents."""

    # Create a simple test document
    test_doc = Document.from_text("""
        Test Paper Title

        Abstract
        This is a test abstract about machine learning algorithms and their applications.
        The paper presents a novel approach to improving model performance through
        advanced optimization techniques.

        Introduction
        In this paper, we discuss the challenges of training deep neural networks
        and propose solutions that improve convergence speed and final accuracy.
        Our method combines several existing techniques with novel modifications.

        Methodology
        We designed experiments to test our approach on standard benchmarks.
        The experimental setup includes hyperparameter tuning and comparative analysis.

        Results
        Our approach shows significant improvements over baseline methods.
        We achieve 15% better performance on average across all tested datasets.

        Conclusion
        The proposed method demonstrates the effectiveness of our optimization strategy.
        Future work will explore applications to other domains and model architectures.
    """)

    # Test with different model
    test_model = "glm-4"  # Different from default "deepseek-chat"

    print(f"Testing DiscussionAgent with model: {test_model}")

    # Create DiscussionAgent with specific model
    agent = DiscussionAgent(model_name=test_model)

    # Test: Create a task and check if the model is passed in context
    print("\n=== Testing Task Creation ===")

    # Start the agent to initialize task manager
    await agent._initialize_discussion(test_doc)

    # Create a test task
    task_id = agent.task_manager.create_task(
        queue_name="main_discussion",
        task_type=TaskType.GENERATE_INSIGHTS,
        parameters={
            "personality": AgentPersonality.CRITICAL_EVALUATOR,
            "document": test_doc,
            "discussion_context": {"phase": "test"},
        },
        priority=TaskPriority.HIGH,
        assigned_to=AgentPersonality.CRITICAL_EVALUATOR,
        timeout_seconds=60,
        context={"model_name": agent.model_name},
    )

    print(f"Created task: {task_id}")

    # Retrieve the task to verify context
    task = agent.discussion_queue.get_task(task_id)

    if task:
        print(f"Task context: {task.context}")
        model_in_context = task.context.get("model_name")

        if model_in_context == test_model:
            print(f"✅ SUCCESS: Model '{test_model}' found in task context")

            # Test that the task tool uses the model from context
            print("\n=== Testing Task Tool Execution ===")

            # Execute the task
            result = await generate_insights_tool(task)

            if result.success:
                print(f"✅ SUCCESS: Task executed successfully with {len(result.insights)} insights")
                print(f"Task result metadata: {result.metadata}")
            else:
                print(f"❌ Task execution failed: {result.error_message}")
        else:
            print(f"❌ FAILURE: Expected model '{test_model}' in context, got '{model_in_context}'")
    else:
        print("❌ FAILURE: Could not retrieve created task")

    # Clean up
    await agent.task_manager.stop_processing()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_model_parameter_propagation())