"""RAG ReAct agent for intelligent document analysis.

This module implements a RAG (Retrieval-Augmented Generation) + ReAct agent
for intelligent iterative document analysis using semantic search and pydantic-ai framework.
"""

import traceback
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List

from pydantic_ai import Agent
from pydantic_ai import ModelRetry
from pydantic_ai import RunContext

from ..document import Document
from ..document.loaders.pdf_loader import PdfLoader
from ..document.splitters.cumulative_flow import CumulativeFlowSplitter
from ..embedding_provider import get_embedding_client
from ..llm_provider import get_model
from ..logging_config import get_logger
from .models.rag_react_models import RAGReActAgentOutput
from .prompts.rag_react import format_agent_prompt

logger = get_logger(__name__)


@dataclass
class RAGReActDeps:
    """Dependencies for RAG ReAct agent iterative analysis."""

    document: Document
    task: str
    max_loops: int = 8
    show_progress: bool = True
    previous_queries: List[str] = field(default_factory=list)
    current_report: str = ""
    loop_count: int = 0
    accessed_chunk_ids: set[str] = field(default_factory=set)
    current_search_query: str = ""
    current_retrieved_content: str = ""
    current_search_results_summary: str = ""


@dataclass
class RAGReActState:
    """State management for RAG ReAct analysis using message history."""

    previous_queries: List[str] = field(default_factory=list)
    current_report: str = ""
    loop_count: int = 0
    current_search_query: str = ""
    current_retrieved_content: str = ""
    current_search_results_summary: str = ""


def load_and_process_document_for_rag(
    file_path: str | Path, to_markdown: bool = False
) -> Document:
    """Load and process document specifically for RAG analysis.

    This function loads PDFs without using Mineru API and uses CumulativeFlowSplitter
    for intelligent chunking based on semantic similarity.

    Args:
        file_path: Path to the document file (PDF or TXT)
        to_markdown: Whether to convert PDF to markdown (always False for RAG)

    Returns:
        Processed Document with chunks and vector index ready for RAG analysis

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the loading or processing fails
    """
    logger.info(f"Loading document for RAG analysis: {file_path}")

    # Determine file type and load accordingly
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        # Load PDF without Mineru API
        logger.info("Loading PDF without Mineru API...")
        loader = PdfLoader(to_markdown=False)
        load_result = loader.load(file_path)

        if not load_result.success:
            raise RuntimeError(f"Failed to load PDF: {load_result.errors}")

        # Create document with loaded text
        doc = Document(
            source_path=file_path,
            text=load_result.text,
            metadata=load_result.metadata,
        )
        logger.info(f"Loaded {len(doc.text)} characters from PDF")

    elif file_path.suffix.lower() == ".txt":
        # Load text file
        logger.info("Loading text file...")
        doc = Document.from_file(file_path, to_markdown=False)
        logger.info(f"Loaded {len(doc.text)} characters from text file")

    else:
        # Try to load as general document
        logger.warning(
            f"Unsupported file type: {file_path.suffix}, attempting general load..."
        )
        doc = Document.from_file(file_path, to_markdown=False)

    # Use CumulativeFlowSplitter for intelligent chunking with Ollama embeddings
    logger.info("Splitting document using CumulativeFlowSplitter...")
    splitter = CumulativeFlowSplitter(
        similarity_threshold=0.45,
        min_segment_sentences=2,
        min_segment_chars=100,
        max_segment_chars=500,
    )

    # Test Ollama connection
    logger.info("Testing connection to Ollama server...")
    if not splitter.test_ollama_connection():
        raise RuntimeError(
            "Cannot connect to Ollama server! Make sure Ollama is running: 'ollama serve'"
        )

    logger.info(f"Using embedding model: {splitter.ollama_client.model}")
    logger.info("Computing embeddings for sentences...")

    # Split the document
    chunks = splitter.split(doc.text)
    doc._set_chunks(chunks)
    logger.info(f"Created {len(doc.chunks)} chunks using CumulativeFlowSplitter")

    # Log chunk statistics
    if doc.chunks:
        chunk_lengths = [len(chunk.content) for chunk in doc.chunks]
        logger.info(
            f"Average chunk size: {sum(chunk_lengths) // len(chunk_lengths)} characters"
        )
        logger.info(f"Min chunk size: {min(chunk_lengths)} characters")
        logger.info(f"Max chunk size: {max(chunk_lengths)} characters")

    return doc


def get_initial_search_query(task: str) -> str:
    """Get the initial search query based on the analysis task.

    Args:
        task: The analysis task or question

    Returns:
        Initial search query string
    """
    # Extract key concepts from the task for initial search
    task_lower = task.lower()

    # Check for common analysis patterns
    if "research question" in task_lower or "objective" in task_lower:
        return "research questions objectives problem statement"
    elif "methodology" in task_lower or "method" in task_lower:
        return "research methodology experimental setup approach"
    elif "result" in task_lower or "finding" in task_lower:
        return "main results findings experimental outcomes"
    elif "contribution" in task_lower or "significance" in task_lower:
        return "key contributions novelty significance impact"
    else:
        # Default to comprehensive initial search
        return "abstract introduction research overview main objectives"


def format_search_results_summary(search_results, query: str) -> str:
    """Format a summary of search results for the agent.

    Args:
        search_results: List of search results (chunks or chunk-score tuples)
        query: The search query that generated these results

    Returns:
        Formatted summary string
    """
    if not search_results:
        return f"No results found for query: '{query}'"

    summary_parts = [
        f"Found {len(search_results)} relevant chunks for query: '{query}'"
    ]

    # Add information about the sections found
    sections = set()

    for result in search_results:
        chunk = result[0] if isinstance(result, tuple) else result
        if chunk.chunk_name != "unknown":
            sections.add(chunk.chunk_name)

    if sections:
        summary_parts.append(f"Sections: {', '.join(sections)}")

    return " | ".join(summary_parts)


def retrieve_content_for_query(
    document: Document, query: str, accessed_chunk_ids: set[str], top_k: int = 5
) -> tuple[str, list]:
    """Retrieve content using semantic search for a given query, filtering out already accessed chunks.

    Args:
        document: Processed document with vector index
        query: Search query string
        accessed_chunk_ids: Set of chunk IDs that have already been accessed
        top_k: Number of results to retrieve

    Returns:
        Tuple of (formatted_content, search_results)
    """
    logger.info(
        f"Searching for content with query: '{query}' (excluding {len(accessed_chunk_ids)} already accessed chunks)"
    )

    # Build vector index if not already built using SiliconFlow embeddings
    if not document.vector_index:
        logger.info(
            "Building vector index for semantic search with SiliconFlow embeddings..."
        )
        # Use embedding provider system to create SiliconFlow client for RAG search
        embedding_client = get_embedding_client(
            "siliconflow/Qwen/Qwen3-Embedding-8B",
            cache_embeddings=True,
        )
        document.build_vector_index(embedding_client=embedding_client)

    # Perform semantic search with more results to filter from
    # We search for more than top_k to account for filtering out already accessed chunks
    expanded_top_k = (
        top_k * 3
    )  # Get 3x more results to ensure we have enough fresh content
    all_search_results = document.semantic_search(
        query, top_k=expanded_top_k, return_scores=True
    )

    if not all_search_results:
        logger.warning(f"No search results found for query: '{query}'")
        return "", []

    # Filter out already accessed chunks
    filtered_results = []
    for chunk, score in all_search_results:
        if chunk.id not in accessed_chunk_ids:
            filtered_results.append((chunk, score))

        # Stop once we have enough results
        if len(filtered_results) >= top_k:
            break

    if not filtered_results:
        logger.warning(
            f"No fresh content found for query: '{query}' (all {len(all_search_results)} results already accessed)"
        )
        return "", []

    logger.info(
        f"Retrieved {len(filtered_results)} fresh chunks for query: '{query}' (from {len(all_search_results)} total results)"
    )

    # Update accessed chunk IDs
    new_chunk_ids = {chunk.id for chunk, _ in filtered_results}
    accessed_chunk_ids.update(new_chunk_ids)

    # Format the retrieved content
    content_parts = []
    for chunk, score in filtered_results:
        section_name = (
            chunk.chunk_name if chunk.chunk_name != "unknown" else "unknown section"
        )
        content_parts.append(
            f"=== {section_name.upper()} (Similarity: {score:.3f}, Chunk ID: {chunk.id}) ===\n{chunk.content}"
        )

    formatted_content = "\n\n".join(content_parts)
    logger.debug(f"Formatted retrieved content: {len(formatted_content)} characters")

    return formatted_content, filtered_results


def analyze_document_with_rag_react(
    document_file: str,
    task: str,
    model: str = "deepseek-chat",
    max_loops: int = 8,
    to_markdown: bool = False,  # Always False for RAG
    show_progress: bool = True,
) -> str:
    """Analyze a document using the RAG ReAct agent.

    Args:
        document_file: Path to the document file (PDF or TXT)
        task: Analysis task or question about the document
        model: Model identifier for the LLM provider
        max_loops: Maximum number of analysis iterations
        to_markdown: Whether to convert PDF to markdown (always False for RAG)
        show_progress: Whether to show progress during analysis

    Returns:
        Comprehensive analysis report generated by the RAG ReAct agent

    Raises:
        FileNotFoundError: If the document file is not found
        Exception: If the analysis fails
    """
    logger.info(f"Starting RAG ReAct analysis for file: {document_file}")
    logger.info(f"Task: {task[:100]}...")
    logger.info(
        f"Configuration: model={model}, max_loops={max_loops}, to_markdown={to_markdown}, show_progress={show_progress}"
    )

    # Check if file exists
    if not Path(document_file).exists():
        raise FileNotFoundError(f"Document file not found: {document_file}")

    # Load and process the document using custom RAG processing
    document = load_and_process_document_for_rag(document_file, to_markdown=to_markdown)

    # Create and run the RAG ReAct agent
    agent = RAGReActAgent(model=model, max_loops=max_loops)
    result = agent.analyze_document(document, task, show_progress=show_progress)

    logger.debug("RAG ReAct analysis completed successfully!")
    return result


class RAGReActAgent:
    """RAG ReAct agent for intelligent document analysis with semantic search.

    This agent implements the Retrieval-Augmented Generation + ReAct pattern
    to analyze documents by iteratively searching for relevant content, making
    decisions about what to search for next, and building a comprehensive report
    using native message history.
    """

    def __init__(self, model: str = "deepseek-chat", max_loops: int = 8):
        """Initialize the RAG ReAct agent.

        Args:
            model: Model identifier for the LLM provider
            max_loops: Maximum number of analysis iterations
        """
        self.logger = get_logger(__name__)
        self.max_loops = max_loops
        self.model = get_model(model)
        self.model_identifier = model

        # Create the pydantic-ai agent with dependencies and structured output
        self.agent = Agent(
            model=self.model,
            deps_type=RAGReActDeps,
            output_type=RAGReActAgentOutput,
        )

        # Add context-aware system prompt with better error handling
        @self.agent.system_prompt
        async def rag_react_system_prompt(ctx: RunContext[RAGReActDeps]) -> str:
            """Generate system prompt with current analysis state."""
            deps = ctx.deps

            # Format status summary
            status = f"Searching and analyzing (loop {deps.loop_count + 1} of {deps.max_loops})"

            # Get content for current search
            if not deps.current_report and not deps.previous_queries:
                # First iteration - get initial search content
                search_query = get_initial_search_query(deps.task)
                retrieved_content, search_results = retrieve_content_for_query(
                    deps.document, search_query, deps.accessed_chunk_ids
                )

                if not retrieved_content.strip():
                    raise ModelRetry(
                        f"No content found for initial search query: '{search_query}'. "
                        "The document might not have relevant content or the search failed."
                    )

                search_results_summary = format_search_results_summary(
                    search_results, search_query
                )

                # Store the current search information for this iteration
                deps.current_search_query = search_query
                deps.current_retrieved_content = retrieved_content
                deps.current_search_results_summary = search_results_summary
            else:
                # Use the search query that was prepared in the previous iteration
                search_query = deps.current_search_query
                retrieved_content = deps.current_retrieved_content
                search_results_summary = deps.current_search_results_summary

            # Format the agent prompt with all necessary information
            return format_agent_prompt(
                task=deps.task,
                status=status,
                search_query=search_query,
                retrieved_content=retrieved_content,
                search_results_summary=search_results_summary,
                current_report=deps.current_report,
                previous_queries=deps.previous_queries.copy(),
            )

        self.logger.info(
            f"Initialized RAGReActAgent with model: {model} (max_loops={max_loops})"
        )

    def analyze_document(
        self, document: Document, task: str, show_progress: bool = True
    ) -> str:
        """Main analysis method that orchestrates the RAG ReAct loop using semantic search.

        Args:
            document: Processed document with vector index capabilities
            task: Analysis task or question about the document
            show_progress: Whether to print reasoning at each step

        Returns:
            Comprehensive analysis report
        """
        self.logger.info(f"Starting RAG ReAct analysis for task: {task[:100]}...")

        # Initialize analysis state
        state = RAGReActState()
        accessed_chunk_ids = set()
        message_history = []

        # Main RAG ReAct loop with message history
        while state.loop_count < self.max_loops:
            state.loop_count += 1

            try:
                # Create dependencies for this iteration
                deps = RAGReActDeps(
                    document=document,
                    task=task,
                    max_loops=self.max_loops,
                    show_progress=show_progress,
                    previous_queries=state.previous_queries,
                    current_report=state.current_report,
                    loop_count=state.loop_count,
                    accessed_chunk_ids=accessed_chunk_ids,
                    current_search_query=state.current_search_query,
                    current_retrieved_content=state.current_retrieved_content,
                    current_search_results_summary=state.current_search_results_summary,
                )

                # Run the agent with message history for context persistence
                self.logger.debug("Running agent with message history")
                result = self.agent.run_sync(
                    "Execute analysis iteration",
                    deps=deps,
                    message_history=message_history,
                )
                agent_output = result.output

                self.logger.debug(
                    f"Agent response: should_stop={agent_output.should_stop}, next_query={agent_output.next_search_query}"
                )

                # Print reasoning for this iteration if show_progress is enabled
                if show_progress:
                    print(f"\n--- Loop {state.loop_count}/{self.max_loops} ---")
                    print(f"Search query: '{deps.current_search_query}'")
                    print(f"Chunks accessed so far: {len(accessed_chunk_ids)}")
                    print(f"Reasoning: {agent_output.reasoning}")
                    if agent_output.should_stop:
                        print("Decision: STOP - Analysis complete")
                    else:
                        print(f"Next search query: '{agent_output.next_search_query}'")
                        print(f"Search strategy: {agent_output.search_strategy}")
                    print("-" * 50)

                # Update state
                if agent_output.report_section.strip():
                    if state.current_report:
                        state.current_report += "\n\n"
                    state.current_report += agent_output.report_section

                # Add current search query to previous queries
                if (
                    deps.current_search_query
                    and deps.current_search_query not in state.previous_queries
                ):
                    state.previous_queries.append(deps.current_search_query)

                # Update message history with this iteration
                message_history.extend(result.new_messages())

                # Check if agent wants to stop
                if agent_output.should_stop:
                    self.logger.info(
                        f"Agent chose to stop after loop {state.loop_count}: {agent_output.reasoning}"
                    )
                    break

                # Prepare for next iteration by retrieving content for the next search query
                if agent_output.next_search_query.strip():
                    next_content, next_results = retrieve_content_for_query(
                        document, agent_output.next_search_query, accessed_chunk_ids
                    )

                    if not next_content.strip():
                        self.logger.warning(
                            f"No fresh content found for next search query: '{agent_output.next_search_query}'"
                        )
                        break

                    # Store the next search results in the state for the next iteration
                    state.current_search_query = agent_output.next_search_query
                    state.current_retrieved_content = next_content
                    state.current_search_results_summary = (
                        format_search_results_summary(
                            next_results, agent_output.next_search_query
                        )
                    )
                else:
                    self.logger.info("No next search query provided")
                    break

            except Exception as e:
                self.logger.error(
                    f"Agent execution failed in loop {state.loop_count}: {e}"
                )
                self.logger.error(f"Exception type: {type(e)}")
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                break

        self.logger.info(f"RAG ReAct analysis completed after {state.loop_count} loops")

        # Log and print the final report
        if state.current_report:
            self.logger.info(f"Report length: {len(state.current_report)} characters")

            # Print the final report to console
            print("\n" + "=" * 80)
            print("FINAL ANALYSIS REPORT")
            print("=" * 80)
            print(state.current_report)
            print("=" * 80)
        else:
            self.logger.warning("No final report generated")
            print("\nWarning: No analysis report was generated")

        return state.current_report

    def __repr__(self) -> str:
        """String representation of the RAGReActAgent."""
        return (
            f"RAGReActAgent(model={self.model_identifier}, max_loops={self.max_loops})"
        )
