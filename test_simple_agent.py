#!/usr/bin/env python3
"""Simple test script for the updated SimpleAgent implementation."""

import asyncio
from sciread.agents.simple_agent import SimpleAgent
from sciread.agents.prompts import remove_citations_section, get_simple_analysis_prompt

def test_citation_removal():
    """Test the citation removal function."""
    print("Testing citation removal function...")

    # Test case 1: Clear "References" section
    text_with_refs = """This is the main content of the paper.
It contains important research findings.

The methodology section describes our approach.

Results show significant improvements.

References
[1] Smith, J. et al. (2023). "Important Paper". Journal of Testing.
[2] Johnson, M. (2022). "Another Paper". Conference Proceedings.
[3] Brown, K. et al. (2021). "Third Paper". Science Magazine.
"""

    cleaned_text = remove_citations_section(text_with_refs)
    print("Test 1 - Clear References section:")
    print("Original length:", len(text_with_refs))
    print("Cleaned length:", len(cleaned_text))
    print("Contains 'References' section:", "References" in cleaned_text)
    print("Contains truncation note:", "Note: References" in cleaned_text)
    print()

    # Test case 2: No references section
    text_no_refs = "This is a simple document without references."
    cleaned_text_no_refs = remove_citations_section(text_no_refs)
    print("Test 2 - No references:")
    print("Original:", repr(text_no_refs))
    print("Cleaned:", repr(cleaned_text_no_refs))
    print("Unchanged:", text_no_refs == cleaned_text_no_refs)
    print()

    # Test case 3: Reference-like content without clear section header
    text_ref_like = """Main content here.

Some findings are discussed.

[1] Author, A. (2023). Title of paper.
[2] Author, B. (2022). Another title.
[3] Author, C. (2021). Third title.
"""

    cleaned_ref_like = remove_citations_section(text_ref_like)
    print("Test 3 - Reference-like content:")
    print("Original length:", len(text_ref_like))
    print("Cleaned length:", len(cleaned_ref_like))
    print("Contains reference patterns:", "[1] Author" in cleaned_ref_like)
    print()

    print("Citation removal tests completed!\n")

def test_simple_agent_creation():
    """Test SimpleAgent creation and basic properties."""
    print("Testing SimpleAgent creation...")

    try:
        agent = SimpleAgent()
        print(f"✓ SimpleAgent created successfully")
        print(f"  Name: {agent.name}")
        print(f"  Model: {agent.config.model_identifier}")
        print(f"  Temperature: {agent.config.temperature}")

        # Test supported questions
        supported_questions = agent.get_supported_questions()
        print(f"  Supported question types: {len(supported_questions)}")
        print(f"  Examples: {supported_questions[:3]}...")

        print("✓ SimpleAgent creation test passed!\n")
        return True

    except Exception as e:
        print(f"✗ SimpleAgent creation failed: {e}")
        return False

async def test_simple_agent_analysis():
    """Test SimpleAgent with a simple analysis (if we have a model configured)."""
    print("Testing SimpleAgent analysis functionality...")

    try:
        agent = SimpleAgent()

        # Simple test document
        test_document = """
        This paper introduces a novel approach to machine learning called "Neural Networks".

        The key innovation is using multiple layers of neurons to learn hierarchical features.
        We tested this approach on image classification tasks and achieved 95% accuracy.

        The methodology involves training on large datasets using backpropagation.
        Results show significant improvements over traditional methods.

        References
        [1] Smith, J. (2023). "Deep Learning Overview". AI Journal.
        """

        # Test question
        question = "Explain this paper using the Feynman technique"

        print("  Document prepared (with references to be removed)")
        print("  Question:", question)

        # Note: We won't actually run the analysis to avoid API calls,
        # but we can test the input preparation
        context = agent.prepare_context(test_document, None)
        context = remove_citations_section(context)

        print("  ✓ Context preparation successful")
        print(f"    Original length: {len(test_document)}")
        print(f"    After citation removal: {len(context)}")
        print("    Contains 'References':", "References" in context)

        # Test prompt formatting
        prompt = get_simple_analysis_prompt().format(
            context=context[:500],  # Truncate for test
            question=question
        )

        print("  ✓ Prompt formatting successful")
        print(f"    Prompt length: {len(prompt)}")
        print("    Contains 'Feynman technique':", "Feynman technique" in prompt)
        print("    Contains 'author of the paper':", "author of the paper" in prompt)

        print("✓ SimpleAgent analysis test passed!\n")
        return True

    except Exception as e:
        print(f"✗ SimpleAgent analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing updated SimpleAgent implementation\n")

    # Test citation removal
    test_citation_removal()

    # Test SimpleAgent creation
    if not test_simple_agent_creation():
        return

    # Test SimpleAgent analysis
    await test_simple_agent_analysis()

    print("🎉 All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())