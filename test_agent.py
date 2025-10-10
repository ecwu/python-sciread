#!/usr/bin/env python3
"""Simple test script for the agent module."""

import asyncio
from pathlib import Path

from src.sciread.agent import create_agent, remove_references_section
from src.sciread.document import Document


async def test_agent():
    """Test the document agent functionality."""
    print("Creating agent...")

    # Create an agent
    agent = create_agent("deepseek/deepseek-chat")
    print(f"Agent created: {agent}")

    # Create a test document from sample text
    sample_text = Path("1706.03762v5.txt").read_text(encoding="latin-1")
    print("Creating document from text...")
    doc = Document.from_text(sample_text)
    print(f"Document created: {len(doc.text)} characters")

    # Test the reference removal function
    cleaned_text = remove_references_section(sample_text)
    print(f"Text after reference removal: {len(cleaned_text)} characters")

    # Test the agent analysis
    task_prompt = """Your task is to write a detailed report using the Feynman technique to explain a given paper. When creating this report, you should approach it as if you were the author of the paper.
Here are some important constraints:
- Content Format: When writing the content of the report, use multiple levels of titles and vivid markdown formats such as bold & italic text, code block, table, and lists. This will make the report easy to read.
- Serious Work Requirement: This report is a serious piece of work, so avoid using emojis when writing the report.
- Content Don'ts: Do not start the content with the paper title, it already displayed in the page title; No need to mention/introduce the faynman technique in the content.
    """

    print("Starting document analysis...")
    try:
        result = await agent.analyze_document(
            document=doc,
            task_prompt=task_prompt,
            remove_references=True,
            clean_text=True,
        )

        print("Analysis completed successfully!")
        print("\n" + "=" * 50)
        print("ANALYSIS RESULT:")
        print("=" * 50)
        print(result)
        print("=" * 50)

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()

    print("Test completed.")


if __name__ == "__main__":
    asyncio.run(test_agent())
