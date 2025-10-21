#!/usr/bin/env python3
"""Simple test for imports and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    from sciread.agent.models.discussion_models import AgentPersonality, DiscussionState
    print("✓ Successfully imported discussion models")
except Exception as e:
    print(f"✗ Failed to import discussion models: {e}")

try:
    from sciread.agent.models.task_models import Task, TaskType
    print("✓ Successfully imported task models")
except Exception as e:
    print(f"✗ Failed to import task models: {e}")

try:
    from sciread.agent.personality_agents import create_personality_agent
    print("✓ Successfully imported personality agents")
except Exception as e:
    print(f"✗ Failed to import personality agents: {e}")

try:
    from sciread.agent.consensus_builder import ConsensusBuilder
    print("✓ Successfully imported consensus builder")
except Exception as e:
    print(f"✗ Failed to import consensus builder: {e}")

try:
    from sciread.document import Document
    print("✓ Successfully imported Document")
except Exception as e:
    print(f"✗ Failed to import Document: {e}")

print("\nTesting basic functionality...")

try:
    # Test creating a simple document
    test_content = "# Test Document\n\nThis is a test document for testing the system."
    doc = Document.from_text(test_content)
    print(f"✓ Created document with {len(doc.get_chunks())} chunks")
except Exception as e:
    print(f"✗ Failed to create document: {e}")

try:
    # Test creating a personality agent
    agent = create_personality_agent(AgentPersonality.CRITICAL_EVALUATOR)
    print(f"✓ Created personality agent: {agent.personality.value}")
except Exception as e:
    print(f"✗ Failed to create personality agent: {e}")

print("\nBasic test completed!")