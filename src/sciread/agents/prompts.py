"""Prompt templates for different agent types and analysis tasks."""

from typing import Dict
from typing import List


def get_simple_analysis_prompt() -> str:
    """Get prompt template for simple document analysis."""
    return """You are an expert academic research analyst. Analyze the following research paper and provide a comprehensive analysis based on the given question.

RESEARCH PAPER:
{context}

QUESTION:
{question}

Please provide a detailed, well-structured analysis in markdown format. Include:
- Direct answers to the question
- Supporting evidence from the paper
- Key findings and insights
- Relevant quotes or data points
- Critical evaluation when appropriate

Be thorough, accurate, and focus on the most relevant information for answering the question."""


def get_section_specific_prompt(section_type: str) -> str:
    """Get prompt template for analyzing specific document sections.

    Args:
        section_type: Type of section (abstract, introduction, methods, etc.)
    """
    section_prompts = {
        "abstract": """You are analyzing the abstract of a research paper. Focus on:
- Research problem and motivation
- Methodology overview
- Key results and contributions
- Conclusions and implications

SECTION CONTENT:
{context}

TASK:
{task}

Provide a focused analysis of this {section_type} section in markdown format.""",

        "introduction": """You are analyzing the introduction section of a research paper. Focus on:
- Research background and context
- Problem statement and research gap
- Research questions and objectives
- Hypotheses or claims
- Structure overview

SECTION CONTENT:
{context}

TASK:
{task}

Provide a comprehensive analysis of this {section_type} section in markdown format.""",

        "methods": """You are analyzing the methods section of a research paper. Focus on:
- Research methodology and approach
- Data collection and processing
- Experimental setup
- Model architecture or algorithms
- Evaluation metrics

SECTION CONTENT:
{context}

TASK:
{task}

Provide a detailed analysis of the methodology in markdown format.""",

        "experiments": """You are analyzing the experiments section of a research paper. Focus on:
- Experimental design and setup
- Datasets and preprocessing
- Baselines and comparison methods
- Implementation details
- Parameter settings

SECTION CONTENT:
{context}

TASK:
{task}

Provide a thorough analysis of the experimental setup in markdown format.""",

        "results": """You are analyzing the results section of a research paper. Focus on:
- Main findings and outcomes
- Quantitative results and metrics
- Statistical significance
- Comparison with baselines
- Ablation studies

SECTION CONTENT:
{context}

TASK:
{task}

Provide a comprehensive analysis of the results in markdown format.""",

        "conclusion": """You are analyzing the conclusion section of a research paper. Focus on:
- Summary of key contributions
- Limitations and weaknesses
- Future work directions
- Broader impact and implications
- Final insights

SECTION CONTENT:
{context}

TASK:
{task}

Provide a thoughtful analysis of the conclusions in markdown format.""",

        "related_work": """You are analyzing the related work section of a research paper. Focus on:
- Literature review and context
- Comparison with previous approaches
- Identification of research gaps
- Positioning of current work
- Key references and their contributions

SECTION CONTENT:
{context}

TASK:
{task}

Provide a comprehensive analysis of the related work in markdown format.""",

        "discussion": """You are analyzing the discussion section of a research paper. Focus on:
- Interpretation of results
- Implications and significance
- Comparison with expectations
- Limitations and alternative explanations
- Broader context and impact

SECTION CONTENT:
{context}

TASK:
{task}

Provide an insightful analysis of the discussion in markdown format.""",
    }

    default_prompt = """You are analyzing a specific section of a research paper.

SECTION CONTENT:
{context}

TASK:
{task}

Provide a detailed analysis of this section in markdown format."""

    return section_prompts.get(section_type.lower(), default_prompt)


def get_research_question_prompts() -> Dict[str, str]:
    """Get prompt templates for high-level research questions.

    Returns:
        Dictionary mapping question types to prompt templates
    """
    return {
        "research_question": """You are a senior academic researcher analyzing a research paper to understand its fundamental research question.

PAPER CONTENT:
{context}

Your task is to identify and analyze the core research question(s) that this paper addresses. Consider:

1. What is the fundamental problem or gap in knowledge that the authors are trying to address?
2. What specific research question(s) guide their investigation?
3. How do the authors frame their contribution to the field?
4. What is the scope and boundaries of their research?

Provide your analysis in markdown format with these sections:
- **Primary Research Question**: The main question the paper addresses
- **Subsidiary Questions**: Supporting questions that emerge from the main question
- **Problem Context**: Background and motivation for the research question
- **Research Gap**: What gap in existing knowledge this question addresses
- **Scope and Limitations**: Boundaries of the research question

Be precise and specific, quoting from the paper when possible.""",

        "motivation": """You are analyzing a research paper to understand why the authors chose this research topic.

PAPER CONTENT:
{context}

Your task is to analyze the motivation and rationale behind this research. Consider:

1. Why is this research topic important or timely?
2. What problems or limitations in existing work motivated this research?
3. What practical or theoretical needs does this research address?
4. What makes this research significant or impactful?

Provide your analysis in markdown format with these sections:
- **Research Importance**: Why this topic matters
- **Existing Limitations**: Problems with current approaches that motivated this work
- **Practical Needs**: Real-world applications or problems addressed
- **Theoretical Contributions**: Advances in understanding or methodology
- **Potential Impact**: How this research could influence the field

Provide specific evidence and quotes from the paper to support your analysis.""",

        "methodology": """You are analyzing a research paper to understand how the authors conducted their research.

PAPER CONTENT:
{context}

Your task is to analyze the research methodology comprehensively. Consider:

1. What research approach did the authors take (experimental, theoretical, computational, etc.)?
2. What data, datasets, or materials were used?
3. What specific methods, models, or algorithms were developed or applied?
4. How was the research designed and executed?
5. What validation or evaluation approaches were used?

Provide your analysis in markdown format with these sections:
- **Research Approach**: Overall methodological framework
- **Data and Materials**: Datasets, sources, or materials used
- **Methods and Techniques**: Specific methodologies, models, or algorithms
- **Experimental Design**: How the research was structured and executed
- **Validation and Evaluation**: How results were validated or assessed
- **Innovation and Novelty**: What makes their methodological approach unique

Focus on providing concrete details about how the research was actually conducted.""",

        "findings": """You are analyzing a research paper to understand what the authors discovered or achieved.

PAPER CONTENT:
{context}

Your task is to analyze the research findings and results. Consider:

1. What are the main discoveries or outcomes of this research?
2. What specific results did the authors obtain?
3. How do these results compare with existing work or baselines?
4. What are the implications of these findings?
5. What are the limitations or caveats to the results?

Provide your analysis in markdown format with these sections:
- **Main Findings**: Primary discoveries or outcomes
- **Quantitative Results**: Specific numerical results, metrics, or measurements
- **Qualitative Insights**: Non-numerical findings or observations
- **Comparative Analysis**: How results compare with existing approaches
- **Significance and Impact**: Why these findings matter
- **Limitations and Constraints**: Boundaries of the findings

Be specific and provide evidence from the paper to support your analysis.""",

        "contribution": """You are analyzing a research paper to understand its main contributions to the field.

PAPER CONTENT:
{context}

Your task is to identify and evaluate the paper's contributions. Consider:

1. What is the primary contribution of this work?
2. What specific advances does it make over existing work?
3. How does it advance the state of the art?
4. What new knowledge, methods, or insights does it provide?
5. What is the potential impact on future research or practice?

Provide your analysis in markdown format with these sections:
- **Primary Contribution**: The main advance or innovation
- **Technical Contributions**: Specific methodological or technical advances
- **Theoretical Contributions**: Advances in understanding or theory
- **Empirical Contributions**: New experimental results or findings
- **State of the Art Advancement**: How this moves the field forward
- **Potential Impact**: Broader implications for research or practice

Evaluate the significance and novelty of each contribution.""",
    }


def get_controller_prompt(available_sections: List[str]) -> str:
    """Get prompt for controller agent to coordinate sub-agent analysis.

    Args:
        available_sections: List of available document sections
    """
    return f"""You are a research analysis coordinator responsible for orchestrating the analysis of an academic paper.

Available paper sections: {', '.join(available_sections)}

PAPER ABSTRACT:
{{abstract}}

TASK: {{task}}

Your role is to:
1. Analyze the abstract to understand the paper's scope and main contributions
2. Determine which sections are most relevant for answering the given task
3. Create a coordinated analysis plan that leverages multiple section-specific analyses
4. Synthesize the results into a comprehensive final answer

First, analyze the abstract and available sections to understand what this paper covers. Then, decide which sections should be analyzed and in what order to best address the task.

Respond with:
1. **Paper Overview**: Brief summary based on the abstract
2. **Analysis Plan**: Which sections to analyze and why
3. **Coordination Strategy**: How different section analyses will complement each other
4. **Expected Outcomes**: What information you expect from each section analysis

Be strategic and comprehensive in your planning."""


def get_synthesis_prompt(section_analyses: Dict[str, str]) -> str:
    """Get prompt for synthesizing multiple section analyses.

    Args:
        section_analyses: Dictionary of section names to their analyses
    """
    analyses_text = ""
    for section, analysis in section_analyses.items():
        analyses_text += f"\n\n## {section.upper()} ANALYSIS:\n{analysis}"

    return f"""You are synthesizing multiple section analyses of a research paper into a comprehensive answer.

SECTION ANALYSES:
{analyses_text}

ORIGINAL TASK: {{task}}

Your task is to:
1. Integrate insights from all section analyses
2. Resolve any contradictions or complement findings
3. Build a coherent, comprehensive answer
4. Highlight the most important findings
5. Provide a well-structured final analysis

Create a final analysis that:
- Directly addresses the original task
- Incorporates relevant insights from all sections
- Maintains consistency and coherence
- Provides specific evidence and quotes
- Offers thoughtful evaluation and synthesis

Format your response as a comprehensive markdown analysis."""


def get_collaborative_agent_prompt(agent_role: str, other_roles: List[str]) -> str:
    """Get prompt for collaborative multi-agent analysis.

    Args:
        agent_role: The role of this specific agent
        other_roles: List of other agent roles in the system
    """
    return f"""You are '{agent_role}', part of a collaborative team of research analysts working together to understand a research paper.

TEAM MEMBERS:
{', '.join(other_roles)}

PAPER CONTENT:
{{context}}

OVERALL RESEARCH QUESTION: {{research_question}}

Your specific role as '{agent_role}' is to focus on:
{{role_specific_focus}}

However, since this is a collaborative analysis:
1. Provide your initial analysis from your perspective
2. Identify areas where input from other team members would be valuable
3. Note any questions or uncertainties that other agents might help resolve
4. Suggest coordination points with other analysts

Format your response as:
1. **{agent_role} Analysis**: Your primary analysis and insights
2. **Collaboration Needs**: Where you need input from other team members
3. **Questions for Others**: Specific questions for other agents
4. **Coordination Suggestions**: How to integrate your analysis with others

Be thorough in your analysis but also thoughtful about how your work fits into the larger collaborative effort."""


def get_final_synthesis_prompt(agent_analyses: Dict[str, str]) -> str:
    """Get prompt for final synthesis of collaborative agent analyses.

    Args:
        agent_analyses: Dictionary of agent names to their analyses
    """
    analyses_text = ""
    for agent, analysis in agent_analyses.items():
        analyses_text += f"\n\n## {agent} ANALYSIS:\n{analysis}"

    return f"""You are synthesizing collaborative analyses from multiple research analysts into a comprehensive understanding of a research paper.

COLLABORATIVE ANALYSES:
{analyses_text}

RESEARCH QUESTION: {{research_question}}

Your task is to create a final, comprehensive analysis that:
1. Integrates insights from all analyst perspectives
2. Resolves any disagreements or complementary findings
3. Builds a coherent understanding of the research
4. Highlights the most important insights and contributions
5. Provides specific evidence and evaluation

Create a final analysis that includes:
- **Executive Summary**: Brief overview of key findings
- **Comprehensive Analysis**: Detailed integration of all insights
- **Key Insights**: Most important discoveries and contributions
- **Critical Evaluation**: Assessment of strengths, limitations, and impact
- **Future Implications**: Potential directions and applications

Format as a professional, comprehensive markdown analysis that demonstrates deep understanding of the research."""