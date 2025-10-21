#!/usr/bin/env python3
"""Test script for the new multi-agent discussion system."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sciread.document import Document
from sciread.agent import DiscussionAgent, AgentPersonality
from sciread.logging_config import setup_logging, info


async def test_discussion_system():
    """Test the discussion system with a sample document."""
    # Setup logging
    setup_logging(level="INFO")

    info("Testing multi-agent discussion system...")

    try:
        # For testing, create a simple text document
        test_content = """
        # Sample Academic Paper

        ## Abstract

        This paper presents a novel approach to machine learning optimization that achieves
        state-of-the-art performance on several benchmark datasets. Our method combines
        deep learning techniques with traditional optimization algorithms to create a hybrid
        approach that outperforms existing methods by 15% on average.

        ## Introduction

        Machine learning optimization has been a challenging problem for decades.
        Traditional methods often struggle with high-dimensional data, while modern
        deep learning approaches require significant computational resources. Our approach
        addresses both limitations by introducing a new optimization paradigm that
        leverages the strengths of both approaches.

        ## Methodology

        We propose the Hybrid Optimization Algorithm (HOA) that combines:
        1. Deep neural networks for feature extraction
        2. Traditional gradient descent for fine-tuning
        3. Novel regularization techniques to prevent overfitting

        Our algorithm achieves better convergence properties and requires 40% less
        computational resources compared to state-of-the-art methods.

        ## Experiments

        We evaluated our method on 5 benchmark datasets:
        - Dataset A: 92% accuracy (vs 85% baseline)
        - Dataset B: 88% accuracy (vs 80% baseline)
        - Dataset C: 95% accuracy (vs 89% baseline)

        Statistical significance was confirmed with p < 0.01.

        ## Conclusion

        Our hybrid approach represents a significant advancement in machine learning
        optimization. The method is both more efficient and more accurate than
        existing approaches, opening new possibilities for real-world applications.

        ## Future Work

        Future research will focus on:
        - Extending the approach to other domains
        - Reducing computational requirements further
        - Theoretical analysis of convergence properties
        """

        # Create document from text
        doc = Document.from_text(test_content)
        info(f"Created document with title: {doc.metadata.title or 'Untitled'}")

        # Test individual personality agents
        info("Testing personality agents...")
        from sciread.agent import create_personality_agent

        for personality in AgentPersonality:
            agent = create_personality_agent(personality)
            info(f"Created {personality.value} agent successfully")

        # Test full discussion system
        info("Testing full discussion system...")
        discussion_agent = DiscussionAgent(
            model_name="deepseek-chat",
            max_iterations=2,  # Keep short for testing
            convergence_threshold=0.6,
            max_discussion_time_minutes=5  # 5 minutes for testing
        )

        # Run the analysis
        info("Starting discussion analysis...")
        result = await discussion_agent.analyze_document(doc)

        # Print results
        info("Discussion analysis completed!")
        print(f"\nDocument Title: {result.document_title}")
        print(f"Summary: {result.summary[:200]}...")
        print(f"Key Contributions: {len(result.key_contributions)}")
        print(f"Consensus Points: {len(result.consensus_points)}")
        print(f"Divergent Views: {len(result.divergent_views)}")
        print(f"Confidence Score: {result.confidence_score:.2f}")

        print(f"\nKey Contributions:")
        for i, contribution in enumerate(result.key_contributions[:3], 1):
            print(f"{i}. {contribution}")

        print(f"\nConsensus Points:")
        for i, point in enumerate(result.consensus_points[:3], 1):
            print(f"{i}. {point.topic}: {point.content[:100]}...")

        if result.divergent_views:
            print(f"\nDivergent Views:")
            for i, view in enumerate(result.divergent_views[:2], 1):
                print(f"{i}. {view.topic}: {view.content[:100]}...")

        print(f"\nFull Summary:")
        print(result.summary)

        info("Test completed successfully!")
        return True

    except Exception as e:
        info(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_discussion_system())
    sys.exit(0 if success else 1)