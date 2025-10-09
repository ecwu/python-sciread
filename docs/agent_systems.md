# Agent Systems

The sciread agent module provides three different approaches for analyzing academic papers with LLM-driven agents. Each agent is designed for different use cases and document characteristics.

## Overview

The agent systems integrate seamlessly with the existing document processing and LLM provider infrastructure to provide intelligent analysis of academic papers.

### Available Agents

1. **SimpleAgent** - Direct full document processing with single LLM calls
2. **ToolCallingAgent** - Controller-based system with specialized sub-agents for different sections
3. **MultiAgentSystem** - Collaborative agents for high-level research question analysis

## Quick Start

```python
import asyncio
from sciread.agents import analyze_document, get_agent_recommendations
from sciread.document import Document

# Load a document
doc = Document.from_file("research_paper.pdf")
doc.load()
doc.split()

# Analyze with automatic agent selection
result = await analyze_document(doc, "What is the main contribution of this paper?")

print(f"Agent: {result.agent_name}")
print(f"Analysis: {result.content}")
```

## Agent Types

### SimpleAgent

The SimpleAgent processes the entire document with a single LLM call. It's straightforward and effective for documents that fit within the model's context window.

**Best for:**
- Short to medium-sized papers
- Simple analysis questions
- Quick summaries
- Documents with clear structure

**Usage:**
```python
from sciread.agents import SimpleAgent, AgentConfig

config = AgentConfig(model_identifier="deepseek-chat", temperature=0.3)
agent = SimpleAgent(config)

result = await agent.analyze(document, "Summarize this paper")
```

**Features:**
- Fast processing with single LLM call
- Automatic token estimation
- Context length management
- Retry logic for reliability

### ToolCallingAgent

The ToolCallingAgent uses a controller to analyze the abstract and determine which sections need detailed analysis, then coordinates multiple specialized sub-agents.

**Best for:**
- Long papers with clear sections
- Comprehensive analysis tasks
- When specific sections are important
- Documents with structured content

**Usage:**
```python
from sciread.agents import ToolCallingAgent

agent = ToolCallingAgent()
result = await agent.analyze(document, "Analyze methodology and results")
```

**How it works:**
1. Controller analyzes abstract and available sections
2. Determines which sections to analyze based on the question
3. Dispatches specialized section agents in parallel
4. Synthesizes results into comprehensive analysis

**Supported sections:**
- Abstract
- Introduction
- Methods/Methodology
- Experiments
- Results
- Conclusion
- Related Work
- Discussion

### MultiAgentSystem

The MultiAgentSystem coordinates multiple specialized agents, each focusing on specific aspects of research analysis (questions, motivation, methods, findings, contributions).

**Best for:**
- High-level research questions
- Complex analytical tasks
- Understanding research context and implications
- When multiple perspectives are valuable

**Usage:**
```python
from sciread.agents import MultiAgentSystem

system = MultiAgentSystem()
result = await system.analyze(document, "What is the research question?")
```

**Specialized agents:**
- **Research Question Agent**: Identifies core research questions and problems
- **Motivation Agent**: Analyzes why the research was conducted
- **Methodology Agent**: Examines research methods and approaches
- **Findings Agent**: Investigates results and discoveries
- **Contribution Agent**: Evaluates contributions and impact

## Configuration

All agents can be configured using `AgentConfig`:

```python
from sciread.agents import AgentConfig

config = AgentConfig(
    model_identifier="deepseek-chat",  # LLM model to use
    temperature=0.3,                   # Creativity level (0.0-1.0)
    max_tokens=4000,                   # Maximum response length
    timeout=300,                       # Request timeout in seconds
    max_retries=3,                     # Number of retry attempts
    retry_delay=1.0,                   # Delay between retries
    include_metadata=True,             # Include metadata in results
    track_processing=True,             # Track processing statistics
)

agent = SimpleAgent(config)
```

## Convenience Functions

### analyze_document()

The easiest way to analyze documents with automatic agent selection:

```python
from sciread.agents import analyze_document

# Automatic agent selection
result = await analyze_document(document, question)

# Force specific agent type
result = await analyze_document(document, question, agent_type="simple")
result = await analyze_document(document, question, agent_type="tool_calling")
result = await analyze_document(document, question, agent_type="multi_agent")
```

### get_agent_recommendations()

Get recommendations for which agents are most suitable:

```python
from sciread.agents import get_agent_recommendations

recommendations = get_agent_recommendations(document, question)

for rec in recommendations:
    print(f"{rec['agent_name']}: {'✅' if rec['is_suitable'] else '❌'}")
    if rec.get('reason'):
        print(f"  Reason: {rec['reason']}")
```

### create_agent()

Create specific agent instances:

```python
from sciread.agents import create_agent

# Create specific agent types
simple_agent = create_agent("simple")
tool_agent = create_agent("tool_calling")
multi_agent = create_agent("multi_agent")

# Create auto-selecting function
auto_analyzer = create_agent("auto")
result = await auto_analyzer(document, question)
```

## AgentOrchestrator

For comprehensive analysis using multiple agents:

```python
from sciread.agents import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Analyze with multiple agents
result = await orchestrator.comprehensive_analysis(
    document,
    "Analyze this research paper comprehensively",
    agent_types=["simple", "tool_calling", "multi_agent"]
)

print(f"Best agent: {result['recommendations']['best_agent']}")
print(f"Most detailed: {result['recommendations']['most_detailed']}")

# Get individual agent results
for agent_name, analysis in result['results'].items():
    print(f"{agent_name}: {analysis['content'][:200]}...")
```

## Research Question Templates

The MultiAgentSystem provides specialized prompts for common research questions:

```python
from sciread.agents import get_research_question_prompts

prompts = get_research_question_prompts()
print(list(prompts.keys()))
# ['research_question', 'motivation', 'methodology', 'findings', 'contribution']
```

**Common questions:**
- "What is the Research Question?"
- "Why is the author doing this topic?"
- "How did the author do the research?"
- "What did the author get from the result?"
- "What is the main contribution?"

## Error Handling

All agents provide comprehensive error handling and logging:

```python
result = await agent.analyze(document, question)

if result.success:
    print(f"Analysis: {result.content}")
    print(f"Execution time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.error_message}")
```

**Common error scenarios:**
- Invalid input parameters
- Model execution failures
- Timeout errors
- Token limit exceeded
- Network connectivity issues

## Performance Considerations

### Token Estimation

Agents can estimate token requirements before processing:

```python
tokens = agent.estimate_tokens(document, question=question)
print(f"Estimated tokens needed: {tokens:,}")
```

### Document Suitability

Check if an agent is suitable for your document:

```python
is_suitable, reason = agent.is_suitable_for_document(document)
if is_suitable:
    print(f"✅ {reason}")
else:
    print(f"❌ {reason}")
    # Consider using a different agent
```

### Processing Time

- **SimpleAgent**: Fastest, single LLM call
- **ToolCallingAgent**: Medium, multiple parallel calls
- **MultiAgentSystem**: Slowest, multiple coordinated calls

## Integration with Document Processing

The agent systems integrate seamlessly with the document module:

```python
from sciread.document import Document
from sciread.agents import analyze_document

# Process document
doc = Document.from_file("paper.pdf")
doc.load()
doc.split()

# Analyze specific sections
abstract_chunks = doc.get_chunks(chunk_type="abstract")
result = await analyze_document(abstract_chunks, "Summarize the research problem")

# Track processing
doc.mark_chunks_processed([chunk for chunk in abstract_chunks])
coverage = doc.get_coverage()
print(f"Coverage: {coverage.chunk_coverage:.1f}%")
```

## Logging and Monitoring

All agents use the project's logging system:

```python
from sciread import setup_logging, get_logger

# Configure logging
setup_logging(level="INFO", log_file="agent_analysis.log")
logger = get_logger(__name__)

# Agent execution is automatically logged
result = await agent.analyze(document, question)
# Logs include execution time, success status, token usage, etc.
```

## Best Practices

### Choosing the Right Agent

1. **Use SimpleAgent for:**
   - Short papers (< 10 pages)
   - Simple questions
   - Quick analysis
   - When speed is important

2. **Use ToolCallingAgent for:**
   - Medium to long papers
   - Section-specific analysis
   - Comprehensive understanding
   - When paper structure is important

3. **Use MultiAgentSystem for:**
   - High-level research questions
   - Complex analytical tasks
   - Understanding research context
   - When multiple perspectives are needed

### Optimizing Performance

- Use token estimation to avoid exceeding context limits
- Pre-filter documents to remove irrelevant sections
- Use appropriate temperature settings for your use case
- Monitor execution time and token usage
- Consider caching results for repeated analyses

### Error Recovery

- Always check `result.success` before using results
- Implement retry logic for failed analyses
- Use different agent types as fallbacks
- Log errors for debugging and monitoring

## Examples

See `examples/agent_usage.py` for comprehensive usage examples, including:

- Simple document analysis
- Section-based analysis
- Collaborative research question analysis
- Convenience function usage
- Comprehensive multi-agent analysis

## Testing

Run the integration test to verify the system works correctly:

```bash
python test_agent_integration.py
```

The test suite covers:
- Agent creation and configuration
- Input validation
- Factory functions
- Agent recommendations
- Error handling
- Token estimation
- Suitability checking