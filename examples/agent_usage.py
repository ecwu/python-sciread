"""
Example usage of the sciread agent systems.

This file demonstrates how to use the three different agent approaches
for analyzing academic papers with LLM-driven agents.
"""

import asyncio
from pathlib import Path

from sciread.agents import (
    AgentConfig,
    AgentOrchestrator,
    MultiAgentSystem,
    SimpleAgent,
    ToolCallingAgent,
    analyze_document,
    create_agent,
    get_agent_recommendations,
)
from sciread.document import Document


async def simple_agent_example():
    """Example using SimpleAgent for basic document analysis."""
    print("=== SimpleAgent Example ===")

    # Create a simple agent
    config = AgentConfig(model_identifier="deepseek-chat", temperature=0.3)
    agent = SimpleAgent(config)

    # Example document text
    document_text = """
    # Title: A Novel Approach to Machine Learning

    ## Abstract
    We propose a new machine learning algorithm that achieves state-of-the-art performance
    on benchmark datasets. Our method combines deep learning with traditional statistical
    approaches to improve accuracy and interpretability.

    ## Introduction
    Machine learning has become increasingly important in recent years. However, existing
    methods often struggle with complex datasets. In this paper, we address this limitation
    by introducing a hybrid approach that leverages the strengths of both deep learning
    and traditional methods.

    ## Methodology
    Our approach consists of three main components: data preprocessing, feature extraction,
    and model training. We use a combination of convolutional neural networks and
    ensemble methods to achieve optimal performance.

    ## Results
    We evaluate our method on three benchmark datasets. Our approach achieves 95% accuracy,
    outperforming existing methods by 10% on average.

    ## Conclusion
    We have demonstrated that our hybrid approach significantly improves performance
    on machine learning tasks. Future work will explore applications to other domains.
    """

    # Analyze the document
    question = "What is the main contribution of this paper?"
    result = await agent.analyze(document_text, question)

    print(f"Question: {question}")
    print(f"Agent: {result.agent_name}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    if result.success:
        print(f"Answer:\n{result.content}")
    else:
        print(f"Error: {result.error_message}")
    print()


async def tool_calling_agent_example():
    """Example using ToolCallingAgent for section-based analysis."""
    print("=== ToolCallingAgent Example ===")

    # Create a document with sections
    doc = Document.from_text("""
    # Research Paper on Natural Language Processing

    ## Abstract
    This paper presents a new transformer-based architecture for natural language processing
    tasks that achieves state-of-the-art performance while requiring fewer computational resources.

    ## Introduction
    Natural language processing has evolved significantly with the introduction of transformer
    models. However, these models often require substantial computational resources.
    This paper addresses this challenge by proposing an efficient transformer variant.

    ## Related Work
    Previous work on efficient transformers includes methods like sparse attention and
    knowledge distillation. Our approach builds upon these ideas but introduces novel
    architectural improvements.

    ## Methodology
    We introduce a new attention mechanism that reduces computational complexity from O(n²) to O(n log n).
    Our model also incorporates dynamic routing to process tokens more efficiently.

    ## Experiments
    We evaluate our model on GLUE benchmark and several custom datasets.
    The results show that our approach achieves comparable performance to BERT-large
    while using 50% less computational resources.

    ## Results
    On average, our model achieves 89% accuracy on GLUE tasks, compared to 90% for BERT-large.
    However, training time is reduced by 60% and memory usage by 55%.

    ## Conclusion
    We have demonstrated that it's possible to maintain high performance while significantly
    reducing computational requirements. Our approach opens up new possibilities for
    deploying large language models in resource-constrained environments.
    """)

    # Split the document into chunks (this would normally be done by the document loader)
    chunks = [
        doc.text,  # For this example, we'll use the full text
    ]

    # Create tool-calling agent
    config = AgentConfig(model_identifier="deepseek-chat", temperature=0.3)
    agent = ToolCallingAgent(config)

    # Analyze with a comprehensive question
    question = "Provide a comprehensive analysis of this research paper, including methodology, results, and contributions."
    result = await agent.analyze(doc, question)

    print(f"Question: {question}")
    print(f"Agent: {result.agent_name}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Chunks Processed: {result.chunks_processed}")
    if result.success:
        print(f"Analysis:\n{result.content}")
    else:
        print(f"Error: {result.error_message}")
    print()


async def multi_agent_system_example():
    """Example using MultiAgentSystem for collaborative analysis."""
    print("=== MultiAgentSystem Example ===")

    # Create a document
    doc = Document.from_text("""
    # Understanding Climate Change Impacts on Global Agriculture

    ## Abstract
    Climate change poses significant challenges to global agriculture. This study analyzes
    the impacts of climate change on crop yields across different regions and proposes
    adaptive strategies to ensure food security.

    ## Introduction
    Agriculture is fundamentally dependent on climate conditions. Rising temperatures,
    changing precipitation patterns, and increased frequency of extreme weather events
    threaten agricultural productivity worldwide. Understanding these impacts is crucial
    for developing effective adaptation strategies.

    ## Research Questions
    1. How does climate change affect crop yields in different agricultural regions?
    2. What are the most significant climate-related threats to food security?
    3. Which adaptation strategies are most effective for maintaining agricultural productivity?

    ## Methodology
    We conducted a comprehensive analysis using climate data from 1980-2020 and crop yield
    data from major agricultural regions. Our methodology combines statistical analysis
    with machine learning models to identify relationships between climate variables and crop productivity.

    ## Data and Analysis
    We analyzed data from 50 countries covering 80% of global agricultural production.
    Climate variables included temperature, precipitation, humidity, and extreme weather events.
    Crop data included yields for wheat, rice, maize, and soybeans.

    ## Results
    Our analysis reveals that temperature increases of 1°C above historical averages
    lead to average yield reductions of 6% for wheat and 4% for rice. However, some
    regions show adaptive capacity with minimal yield changes.

    ## Discussion
    The impacts of climate change on agriculture are highly variable by region and crop type.
    Sub-Saharan Africa and South Asia are particularly vulnerable due to limited adaptive capacity.
    Developed countries show greater resilience through advanced agricultural technologies.

    ## Conclusion
    Climate change significantly impacts global agriculture, but targeted adaptation strategies
    can mitigate many negative effects. International cooperation and technology transfer
    are essential for ensuring global food security.

    ## Future Work
    Future research should focus on developing climate-resilient crop varieties and
    improving agricultural practices in vulnerable regions.
    """)

    # Create multi-agent system
    config = AgentConfig(model_identifier="deepseek-chat", temperature=0.3)
    system = MultiAgentSystem(config)

    # Ask high-level research questions
    questions = [
        "What is the Research Question?",
        "Why is the author doing this topic?",
        "How did the author do the research?",
        "What did the author get from the result?",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await system.analyze(doc, question)

        if result.success:
            print(f"Answer:\n{result.content[:500]}...")
            print(f"[Full answer: {len(result.content)} characters]")
        else:
            print(f"Error: {result.error_message}")
        print("-" * 50)

    print()


async def convenience_functions_example():
    """Example using convenience functions for easy analysis."""
    print("=== Convenience Functions Example ===")

    document_text = """
    # The Future of Artificial Intelligence in Healthcare

    ## Abstract
    Artificial intelligence (AI) is revolutionizing healthcare by enabling more accurate
    diagnostics, personalized treatment plans, and efficient healthcare delivery.

    ## Introduction
    Healthcare systems worldwide face challenges including rising costs, workforce shortages,
    and increasing patient demands. AI technologies offer promising solutions to address
    these challenges while improving patient outcomes.

    ## Key Applications
    1. Medical imaging and diagnostics
    2. Drug discovery and development
    3. Personalized medicine
    4. Administrative workflow automation
    5. Remote patient monitoring

    ## Benefits
    - Improved diagnostic accuracy
    - Reduced healthcare costs
    - Enhanced patient experience
    - Better resource allocation
    - Faster drug development

    ## Challenges
    - Data privacy and security
    - Regulatory compliance
    - Algorithm bias and fairness
    - Integration with existing systems
    - Healthcare professional training

    ## Future Outlook
    AI will continue to transform healthcare, but successful implementation requires
    addressing technical, ethical, and regulatory challenges while ensuring equitable access.
    """

    # Use the convenience function for automatic agent selection
    question = "What are the main benefits and challenges of AI in healthcare?"
    result = await analyze_document(document_text, question, agent_type="auto")

    print(f"Auto-selected analysis for: {question}")
    print(f"Success: {result.success}")
    print(f"Agent: {result.agent_name}")
    if result.success:
        print(f"Answer:\n{result.content}")
    print()

    # Get agent recommendations
    recommendations = get_agent_recommendations(document_text, question)
    print("Agent Recommendations:")
    for rec in recommendations:
        status = "✅ Suitable" if rec["is_suitable"] else "❌ Not suitable"
        print(f"  {rec['agent_name']}: {status}")
        if rec.get("reason"):
            print(f"    Reason: {rec['reason']}")
    print()


async def orchestrator_example():
    """Example using AgentOrchestrator for comprehensive analysis."""
    print("=== AgentOrchestrator Example ===")

    doc = Document.from_text("""
    # Blockchain Technology in Supply Chain Management

    ## Abstract
    This paper explores the application of blockchain technology for improving transparency
    and efficiency in supply chain management. We present a case study of a multinational
    corporation that implemented blockchain solutions.

    ## Introduction
    Supply chains face challenges including lack of transparency, inefficient tracking,
    and fraud. Blockchain technology offers potential solutions through its immutable
    ledger and smart contract capabilities.

    ## Problem Statement
    Traditional supply chain management suffers from information asymmetry, lack of real-time
    tracking, and vulnerability to fraud. These issues result in estimated annual losses
    of billions of dollars globally.

    ## Proposed Solution
    We propose a blockchain-based supply chain management system that provides:
    - Real-time tracking of goods
    - Immutable transaction records
    - Automated smart contract execution
    - Enhanced transparency for all stakeholders

    ## Implementation
    Our solution was implemented across three product lines in 15 countries.
    The system handles over 1 million transactions monthly and involves 500+ suppliers.

    ## Results
    After 12 months of implementation, we observed:
    - 40% reduction in tracking errors
    - 25% improvement in delivery times
    - 60% reduction in fraud incidents
    - 30% cost savings in documentation processing

    ## Conclusion
    Blockchain technology significantly improves supply chain efficiency and transparency.
    However, successful implementation requires careful planning and stakeholder buy-in.
    """)

    # Create orchestrator
    config = AgentConfig(model_identifier="deepseek-chat", temperature=0.3)
    orchestrator = AgentOrchestrator(config)

    # Perform comprehensive analysis
    question = "Analyze the effectiveness of blockchain technology in supply chain management based on this case study."
    result = await orchestrator.comprehensive_analysis(doc, question, agent_types=["simple", "tool_calling"])

    print(f"Comprehensive Analysis for: {question}")
    print(f"Total Agents Used: {len(result['agents_used'])}")
    print(f"Successful Analyses: {len(result['successful_analyses'])}")
    print(f"Failed Analyses: {len(result['failed_analyses'])}")

    print("\nAgent Results:")
    for agent_name, analysis in result["results"].items():
        print(f"\n{agent_name}:")
        if analysis["success"]:
            print(f"  ✓ Success ({analysis['execution_time']:.2f}s)")
            print(f"  Content preview: {analysis['content'][:200]}...")
        else:
            print(f"  ✗ Failed: {analysis['error_message']}")

    print(f"\nRecommendations:")
    print(f"  Fastest Agent: {result['recommendations']['best_agent']}")
    print(f"  Most Detailed: {result['recommendations']['most_detailed']}")
    print()


async def main():
    """Run all examples."""
    print("Sciread Agent System Examples\n" + "="*50 + "\n")

    # Run examples (commented out to avoid actual LLM calls during testing)
    # Uncomment the examples you want to run:

    await simple_agent_example()
    await tool_calling_agent_example()
    await multi_agent_system_example()
    await convenience_functions_example()
    await orchestrator_example()


if __name__ == "__main__":
    # Note: These examples require actual LLM API calls.
    # Make sure you have configured your API keys in config/sciread.toml
    # before running these examples.

    print("Sciread Agent System Examples")
    print("="*50)
    print("\nTo run these examples:")
    print("1. Configure your API keys in config/sciread.toml")
    print("2. Install the package: pip install -e .")
    print("3. Run: python examples/agent_usage.py")
    print("\nEach example demonstrates different agent capabilities:")
    print("- SimpleAgent: Direct document processing")
    print("- ToolCallingAgent: Section-based analysis")
    print("- MultiAgentSystem: Collaborative research analysis")
    print("- Convenience functions: Easy-to-use interfaces")
    print("- Orchestrator: Comprehensive multi-agent analysis")