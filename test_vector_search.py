#!/usr/bin/env python3
"""Simple test script for document vectorization and semantic search.

This script demonstrates:
1. Loading a PDF as text (without Mineru)
2. Splitting into chunks using CumulativeFlowSplitter
3. Building a vector index
4. Interactive semantic search
"""

import sys
from pathlib import Path

from sciread.document import Document
from sciread.document.loaders.pdf_loader import PdfLoader
from sciread.document.splitters.cumulative_flow import CumulativeFlowSplitter


def main():
    """Main function to test document vectorization and semantic search."""
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_vector_search.py <path_to_pdf>")
        print("Example: python test_vector_search.py paper.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    # Validate PDF file
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Warning: File does not have .pdf extension: {pdf_path}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    print(f"\n{'='*60}")
    print(f"Loading document: {pdf_path.name}")
    print(f"{'='*60}\n")

    # Step 1: Load PDF as text (to_markdown=False means no Mineru)
    print("Step 1: Loading PDF as text (no splitting yet)...")
    loader = PdfLoader(to_markdown=False)
    load_result = loader.load(pdf_path)

    if not load_result.success:
        print(f"Error loading PDF: {load_result.errors}")
        sys.exit(1)

    # Create document without any splitting
    doc = Document(
        source_path=pdf_path,
        text=load_result.text,
        metadata=load_result.metadata,
    )
    print(f"✓ Loaded {len(doc.text)} characters")

    # Step 2: Split using CumulativeFlowSplitter
    print("\nStep 2: Splitting document using CumulativeFlowSplitter...")
    print("  - Extracting sentences from text...")
    splitter = CumulativeFlowSplitter(
        similarity_threshold=0.45,
        min_segment_sentences=2,
        min_segment_chars=100,
        max_segment_chars=500,
    )

    # Test Ollama connection first
    print("  - Testing connection to Ollama server...")
    if not splitter.test_ollama_connection():
        print("  ✗ Error: Cannot connect to Ollama server!")
        print("    Make sure Ollama is running: 'ollama serve'")
        sys.exit(1)
    print("  ✓ Connected to Ollama server")

    # Get embedding model info
    print(f"  - Using embedding model: {splitter.ollama_client.model}")
    print("  - Computing embeddings for sentences...")
    print("    (This may take a moment for longer documents)")

    chunks = splitter.split(doc.text)
    # Set chunks using the internal method (this is a test script)
    doc._set_chunks(chunks)
    print(f"✓ Created {len(doc.chunks)} chunks using CumulativeFlowSplitter")

    # Show chunk statistics
    print("\nChunk Statistics:")
    print(f"  Total chunks: {len(doc.chunks)}")
    chunk_lengths = [len(chunk.content) for chunk in doc.chunks]
    print(
        f"  Average chunk size: {sum(chunk_lengths) // len(chunk_lengths)} characters"
    )
    print(f"  Min chunk size: {min(chunk_lengths)} characters")
    print(f"  Max chunk size: {max(chunk_lengths)} characters")

    # Show first few chunks
    print("\nFirst 3 chunks preview:")
    for i, chunk in enumerate(doc.chunks[:3], 1):
        preview = chunk.content[:100].replace("\n", " ")
        if len(chunk.content) > 100:
            preview += "..."
        print(f"  [{i}] {preview}")

    # Step 3: Build vector index
    print("\nStep 3: Building vector index...")
    print("(This may take a moment depending on the number of chunks)")
    doc.build_vector_index(persist=False)
    print("✓ Vector index built successfully")

    # Step 4: Interactive search loop
    print(f"\n{'='*60}")
    print("Semantic Search Interface")
    print(f"{'='*60}")
    print("\nYou can now search the document using natural language queries.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            # Get query from user
            query = input("Enter your query: ").strip()

            # Check for exit commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            # Skip empty queries
            if not query:
                continue

            # Perform semantic search with improved cosine similarity
            print(f"\nSearching for: '{query}'")

            # Use the improved semantic_search method with scores
            results_with_scores = doc.semantic_search(
                query, top_k=3, return_scores=True
            )

            if not results_with_scores:
                print("No results found.")
                continue

            # Display results with similarity scores
            print(f"\nFound {len(results_with_scores)} relevant chunks:\n")
            for i, (chunk, similarity) in enumerate(results_with_scores, 1):
                print(f"{'─'*60}")
                print(f"Result {i}:")
                print(f"Position: {chunk.position}")
                print(f"Length: {len(chunk.content)} characters")
                print(f"Chunk Confidence: {chunk.confidence:.2f}")
                print(f"Match Similarity (Cosine): {similarity:.4f}")
                print("\nContent:")
                # Show full content
                print(chunk.content)
                print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()
