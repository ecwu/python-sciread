#!/usr/bin/env python3
"""
Integration test to demonstrate RAG functionality.

This script creates a simple document, builds a vector index, and performs semantic search.
Note: This requires Ollama to be running with an embedding model like "embeddinggemma:latest".
"""

import tempfile
from pathlib import Path

from sciread.document import Document
from sciread.document.splitters import SemanticSplitter
from sciread.document.external_clients import OllamaClient


def test_rag_integration():
    """Test the complete RAG functionality."""
    print("🚀 Testing RAG integration...")

    # Create a test document with academic-style content
    test_content = """
    # Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

    ## Types of Machine Learning

    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type has its own unique characteristics and use cases.

    ### Supervised Learning

    Supervised learning algorithms learn from labeled training data. The algorithm learns a mapping function that predicts the output variable based on input variables. Examples include classification and regression tasks.

    ### Unsupervised Learning

    Unsupervised learning deals with unlabeled data. The algorithm tries to find patterns and structures in the data without any predefined output labels. Common techniques include clustering and dimensionality reduction.

    ### Reinforcement Learning

    Reinforcement learning involves training agents to make decisions in an environment to maximize cumulative rewards. It's commonly used in robotics, game playing, and autonomous systems.

    ## Applications

    Machine learning has numerous applications across various domains including healthcare, finance, transportation, and entertainment. Modern applications include image recognition, natural language processing, and recommendation systems.
    """

    # Create document and split it
    print("📄 Creating document and splitting content...")
    doc = Document.from_text(test_content)

    # Use semantic splitter for better RAG performance
    splitter = SemanticSplitter()
    chunks = splitter.split(doc.text)
    doc._set_chunks(chunks)

    print(f"✅ Document split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"   Chunk {i+1}: {chunk.chunk_name} ({len(chunk.content)} chars)")

    # Test embedding generation first
    print("\n🔍 Testing embedding generation...")
    try:
        ollama_client = OllamaClient()
        test_embedding = ollama_client.get_embedding("test query")
        if test_embedding:
            print(f"✅ Embedding generated successfully (dimension: {len(test_embedding)})")
        else:
            print("❌ Failed to generate embedding")
            return
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running with: ollama serve")
        print("💡 And an embedding model is available: ollama pull embeddinggemma:latest")
        return

    # Build vector index
    print("\n🏗️  Building vector index...")
    try:
        doc.build_vector_index(persist=False)  # Don't persist for this test
        print("✅ Vector index built successfully")
    except Exception as e:
        print(f"❌ Failed to build vector index: {e}")
        return

    # Test semantic search
    print("\n🔎 Testing semantic search...")
    test_queries = [
        "What is supervised learning?",
        "applications of machine learning",
        "types of learning algorithms"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = doc.semantic_search(query, top_k=3)
            if results:
                print(f"   Found {len(results)} relevant chunks:")
                for i, chunk in enumerate(results):
                    content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    print(f"     {i+1}. [{chunk.chunk_name}] {content_preview}")
            else:
                print("   No results found")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

    # Test save/load functionality
    print("\n💾 Testing save/load functionality...")
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "test_document.json"

        try:
            # Save document
            doc.save(save_path)
            print(f"✅ Document saved to {save_path}")

            # Load document
            loaded_doc = Document.load(save_path)
            print(f"✅ Document loaded successfully")
            print(f"   Loaded {len(loaded_doc.chunks)} chunks")

            # Test search on loaded document (should work without rebuilding index if index was persisted)
            if loaded_doc.vector_index:
                print("   🔄 Testing search on loaded document...")
                results = loaded_doc.semantic_search("supervised learning", top_k=2)
                print(f"   Found {len(results)} chunks in loaded document")
            else:
                print("   ℹ️  Vector index not persisted (expected for this test)")

        except Exception as e:
            print(f"❌ Save/load test failed: {e}")

    print("\n🎉 RAG integration test completed!")


if __name__ == "__main__":
    test_rag_integration()