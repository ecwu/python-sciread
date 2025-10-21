"""Consensus building prompts for multi-agent discussion system."""

from typing import Dict, List, Any


CONSENSUS_BUILDER_SYSTEM_PROMPT = """You are an expert Consensus Builder for academic research analysis. Your role is to synthesize insights from multiple expert agents into a balanced, comprehensive assessment of a research paper.

Your core responsibilities:
- Synthesize diverse perspectives into coherent analysis
- Identify points of agreement and disagreement
- Balance competing viewpoints fairly
- Provide nuanced assessment of paper's contributions and limitations
- Maintain academic rigor and objectivity

The four expert perspectives you're synthesizing:
1. **Critical Evaluation**: Methodological rigor, limitations, potential flaws
2. **Innovative Insights**: Novel contributions, breakthrough potential, future directions
3. **Practical Applications**: Real-world applicability, implementation feasibility
4. **Theoretical Integration**: Conceptual contributions, theoretical significance

Your synthesis should:
- Give appropriate weight to each perspective
- Acknowledge where consensus exists and where disagreements remain
- Provide balanced assessment that honors both strengths and weaknesses
- Be specific and evidence-based rather than vague
- Help readers understand the paper's true significance and limitations

You are not trying to force artificial agreement but to honestly represent the collective understanding that emerged from the discussion."""


SUMMARY_SYNTHESIS_PROMPT = """Create a comprehensive summary based on multi-agent discussion of an academic paper.

**Paper Information:**
Title: {document_title}
Abstract: {document_abstract}

**Key Insights from Expert Analysis:**
{insights_summary}

**Points of Strong Agreement:**
{consensus_points}

**Areas of Ongoing Discussion:**
{divergent_views}

**Discussion Context:**
- Total iterations: {iterations}
- Final convergence score: {convergence_score:.2f}
- Insights generated: {total_insights}
- Questions and answers exchanged: {total_qa}

**Instructions:**
1. Create a balanced, comprehensive summary (400-600 words)
2. Highlight the paper's main contributions as identified by the analysis
3. Acknowledge both strengths and limitations identified in discussion
4. Note where experts agreed and where significant differences remain
5. Provide context for understanding the paper's place in its field
6. Maintain academic tone while being accessible

**Structure your summary with these sections:**
- **Overview**: Brief introduction to the paper and its main focus
- **Key Contributions**: Main contributions identified by expert analysis
- **Methodological Assessment**: Critical evaluation of approaches and methods
- **Significance and Impact**: Assessment of the paper's importance and implications
- **Limitations and Concerns**: Issues raised during expert discussion
- **Future Directions**: Potential next steps and research opportunities

Focus on creating a synthesis that honors all expert perspectives while providing clear, actionable understanding of the paper's true value."""


SIGNIFICANCE_ASSESSMENT_PROMPT = """Assess the overall significance of an academic paper based on comprehensive multi-agent analysis.

**Paper Context:**
Title: {document_title}
Field: [Extract from paper content]

**Analysis Summary:**
{summary_highlights}

**Contribution Assessment:**
{contribution_analysis}

**Consensus Strength:**
- Convergence Score: {convergence_score:.2f}
- Strong Consensus Points: {strong_consensus_count}
- Key Disagreements: {divergent_view_count}

**Expert Perspectives:**
{expert_perspectives}

**Assessment Framework:**
Evaluate significance across multiple dimensions:

1. **Theoretical Significance** (0.0-1.0):
   - Advances understanding of fundamental concepts
   - Challenges or extends existing theories
   - Provides new conceptual frameworks

2. **Methodological Innovation** (0.0-1.0):
   - Introduces novel research methods or approaches
   - Improves upon existing methodologies
   - Enables new types of research

3. **Practical Impact** (0.0-1.0):
   - Solves real-world problems
   - Has commercial or industrial applications
   - Influences practice or policy

4. **Field Advancement** (0.0-1.0):
   - Opens new research directions
   - Influences future work in the field
   - Addresses important gaps in knowledge

5. **Scholarly Value** (0.0-1.0):
   - Rigorous and well-executed research
   - Clear contribution to academic literature
   - Likely to be cited and built upon

**Provide your assessment:**
```
Theoretical Significance: [0.0-1.0]
Methodological Innovation: [0.0-1.0]
Practical Impact: [0.0-1.0]
Field Advancement: [0.0-1.0]
Scholarly Value: [0.0-1.0]
Overall Significance: [0.0-1.0]

Significance Assessment:
[Your detailed assessment explaining the ratings, highlighting both strengths and limitations]
```

Be honest about limitations while giving credit where due. Consider both the paper's intrinsic quality and its potential impact."""


CONTRIBUTION_EXTRACTION_PROMPT = """Extract the key contributions of an academic paper from multi-agent analysis results.

**Paper Title:** {document_title}

**Top Expert Insights:**
{top_insights}

**Consensus on Contributions:**
{consensus_contributions}

**Expert Perspectives on Value:**
{expert_value_assessments}

**Analysis Criteria:**
Identify contributions that are:
- Clearly supported by the analysis
- Recognized across multiple expert perspectives
- Represent genuine advancement over previous work
- Specific and actionable rather than vague claims

**Types of Contributions to Identify:**
1. **Theoretical Contributions**: New concepts, frameworks, models
2. **Methodological Contributions**: New methods, approaches, techniques
3. **Empirical Contributions**: New findings, data, experimental results
4. **Practical Contributions**: Applications, tools, implementations
5. **Knowledge Integration**: Synthesis of existing work in new ways

**Format your response as:**
```
Key Contributions:
1. [Contribution title/description]
   - Type: [theoretical/methodological/empirical/practical/integration]
   - Supporting Evidence: [Brief evidence from analysis]
   - Expert Support: [Which agents recognize this contribution]
   - Significance: [Why this matters]

2. [Next contribution...]

Overall Assessment:
[Brief summary of the paper's main contribution to the field]
```

Focus on quality over quantity - identify the most important 3-5 contributions rather than listing everything mentioned."""


DIVERGENT_VIEW_ANALYSIS_PROMPT = """Analyze divergent views that emerged from multi-agent discussion of an academic paper.

**Discussion Context:**
Paper Title: {document_title}
Convergence Score: {convergence_score:.2f}
Total Agents: 4

**Divergent Views Identified:**
{divergent_views}

**Related Questions and Challenges:**
{related_challenges}

**Analysis Framework:**
For each divergent view, analyze:
1. **Nature of Divergence**: What specifically is disagreed upon?
2. **Agent Positions**: Which agents hold which views and why?
3. **Evidence Base**: What evidence supports each position?
4. **Resolution Potential**: Could this disagreement be resolved with more information?
5. **Impact Significance**: How does this divergence affect overall assessment?

**Categorize Divergence Types:**
- **Methodological Disagreements**: Different views on research approaches
- **Interpretive Differences**: Different readings of results or implications
- **Value Judgments**: Different assessments of importance or impact
- **Scope Differences**: Different views on boundaries or generalizability
- **Future-Oriented Disagreements**: Different expectations about potential

**Format your analysis:**
```
Divergent View 1: [Brief title]
- Agents in Conflict: [Agent names and positions]
- Core Disagreement: [What specifically is disagreed upon]
- Evidence Considerations: [Key evidence for each position]
- Resolution Prospects: [Likelihood of resolving this disagreement]
- Impact on Assessment: [How this affects overall evaluation]

Divergent View 2: [...]

Overall Conflict Assessment:
[Brief analysis of the nature and significance of disagreements]
```

Be balanced in presenting different positions and honest about where consensus was not achieved."""


def build_summary_synthesis_prompt(
    document_title: str,
    document_abstract: str,
    insights_summary: str,
    consensus_points: str,
    divergent_views: str,
    iterations: int,
    convergence_score: float,
    total_insights: int,
    total_qa: int
) -> str:
    """Build prompt for creating comprehensive summary."""
    return SUMMARY_SYNTHESIS_PROMPT.format(
        document_title=document_title,
        document_abstract=document_abstract,
        insights_summary=insights_summary,
        consensus_points=consensus_points,
        divergent_views=divergent_views,
        iterations=iterations,
        convergence_score=convergence_score,
        total_insights=total_insights,
        total_qa=total_qa
    )


def build_significance_assessment_prompt(
    document_title: str,
    summary_highlights: str,
    contribution_analysis: str,
    convergence_score: float,
    strong_consensus_count: int,
    divergent_view_count: int,
    expert_perspectives: str
) -> str:
    """Build prompt for assessing paper significance."""
    return SIGNIFICANCE_ASSESSMENT_PROMPT.format(
        document_title=document_title,
        summary_highlights=summary_highlights,
        contribution_analysis=contribution_analysis,
        convergence_score=convergence_score,
        strong_consensus_count=strong_consensus_count,
        divergent_view_count=divergent_view_count,
        expert_perspectives=expert_perspectives
    )


def build_contribution_extraction_prompt(
    document_title: str,
    top_insights: str,
    consensus_contributions: str,
    expert_value_assessments: str
) -> str:
    """Build prompt for extracting key contributions."""
    return CONTRIBUTION_EXTRACTION_PROMPT.format(
        document_title=document_title,
        top_insights=top_insights,
        consensus_contributions=consensus_contributions,
        expert_value_assessments=expert_value_assessments
    )


def build_divergent_view_analysis_prompt(
    document_title: str,
    convergence_score: float,
    divergent_views: str,
    related_challenges: str
) -> str:
    """Build prompt for analyzing divergent views."""
    return DIVERGENT_VIEW_ANALYSIS_PROMPT.format(
        document_title=document_title,
        convergence_score=convergence_score,
        divergent_views=divergent_views,
        related_challenges=related_challenges
    )