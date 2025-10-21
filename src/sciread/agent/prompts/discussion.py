"""Discussion coordinator prompts for multi-agent system."""

from typing import Dict, List, Any


DISCUSSION_COORDINATOR_SYSTEM_PROMPT = """You are a Discussion Coordinator for multi-agent academic paper analysis. Your role is to manage and coordinate discussions between four expert agents to achieve a comprehensive understanding of research papers.

Your core responsibilities:
- Orchestrate multi-phase discussion process
- Ensure all personality types contribute their unique perspectives
- Monitor discussion progress and convergence
- Facilitate constructive dialogue between agents
- Balance thoroughness with efficiency

The four expert agents are:
1. **Critical Evaluator**: Identifies limitations, methodological flaws, and potential weaknesses
2. **Innovative Insighter**: Recognizes novel contributions and breakthrough potential
3. **Practical Applicator**: Assesses real-world applications and implementation feasibility
4. **Theoretical Integrator**: Analyzes theoretical frameworks and conceptual contributions

Your coordination strategy should ensure:
- All agents have opportunities to contribute
- Discussions remain focused on understanding the paper's significance
- Questions are constructive and lead to deeper analysis
- Convergence is achieved without forcing artificial consensus
- Both strengths and limitations are thoroughly examined

You will manage discussion phases:
1. **Initial Analysis**: Each agent generates initial insights
2. **Questioning**: Agents ask targeted questions about each other's insights
3. **Responding**: Agents address questions and potentially revise their insights
4. **Convergence**: Evaluate if sufficient agreement has been reached
5. **Consensus Building**: Synthesize final comprehensive assessment

Always maintain academic rigor while ensuring productive dialogue."""


PHASE_TRANSITION_PROMPT = """Evaluate whether to advance to the next discussion phase based on current progress.

**Current Phase:** {current_phase}
**Iteration:** {iteration}/{max_iterations}
**Time Elapsed:** {time_elapsed}

**Progress Indicators:**
- Insights Generated: {total_insights} across {agents_with_insights} agents
- Questions Asked: {total_questions}
- Questions Answered: {total_answers}
- Average Insight Quality: {avg_insight_quality:.2f}
- Discussion Activity: {activity_level}

**Phase-Specific Criteria:**
{phase_criteria}

**Decision Framework:**
1. **Advance if**: Minimum requirements met and activity is productive
2. **Continue if**: Progress is steady but more refinement needed
3. **Timeout if**: Maximum iterations reached or time limit exceeded

Provide your recommendation in this format:
```
Decision: [advance/continue/timeout]
Reasoning: [Your reasoning for this decision]
Next Phase: [Next phase name or "current"]
```
"""


INITIAL_ANALYSIS_CRITERIA = """For Initial Analysis Phase:
- Minimum 2 insights per agent (8 total)
- Average importance score > 0.5
- All agents have participated
- No obvious missing perspectives
- Quality of insights justifies moving to questioning"""

QUESTIONING_CRITERIA = """For Questioning Phase:
- Minimum 4 questions asked (at least 1 per insight)
- Questions are targeted and constructive
- Questions span multiple personality perspectives
- No agent completely ignored
- Questions show evidence of careful insight review"""

RESPONDING_CRITERIA = """For Responding Phase:
- At least 80% of questions answered
- Responses are thoughtful and specific
- Some insights revised based on questions
- Responded questions show understanding of concerns
- All target agents had opportunity to respond"""

CONVERGENCE_CRITERIA = """For Convergence Evaluation:
- Multiple iterations of questioning/responding completed
- New insights/questions are diminishing
- Agents report higher convergence scores
- Key points of agreement identified
- Remaining disagreements are clearly articulated"""


def build_phase_evaluation_prompt(
    current_phase: str,
    iteration: int,
    max_iterations: int,
    time_elapsed: str,
    progress_metrics: Dict[str, Any]
) -> str:
    """Build a prompt for evaluating phase progress."""
    phase_criteria_map = {
        "initial_analysis": INITIAL_ANALYSIS_CRITERIA,
        "questioning": QUESTIONING_CRITERIA,
        "responding": RESPONDING_CRITERIA,
        "convergence": CONVERGENCE_CRITERIA,
    }

    phase_criteria = phase_criteria_map.get(current_phase, "Evaluate if current phase objectives have been met.")

    return PHASE_TRANSITION_PROMPT.format(
        current_phase=current_phase,
        iteration=iteration,
        max_iterations=max_iterations,
        time_elapsed=time_elapsed,
        phase_criteria=phase_criteria,
        **progress_metrics
    )


CONVERGENCE_EVALUATION_PROMPT = """Evaluate the convergence of the multi-agent discussion and determine if consensus has been reached.

**Discussion Statistics:**
- Iterations Completed: {iterations}
- Total Insights: {total_insights}
- Total Questions: {total_questions}
- Total Responses: {total_responses}

**Agent Participation:**
{agent_participation}

**Insight Quality Trends:**
{quality_trends}

**Key Patterns:**
{key_patterns}

**Convergence Indicators:**
- Are insights becoming more consistent across agents?
- Are questions and answers leading to refinement rather than disagreement?
- Have major issues been resolved or clearly identified?
- Are remaining disagreements fundamental or minor?

**Evaluation Framework:**
Rate each aspect on a scale of 0.0-1.0:

1. **Consistency**: How aligned are the insights across different agents?
2. **Completeness**: Have all important aspects been thoroughly examined?
3. **Resolution**: Have major conflicts been addressed?
4. **Stability**: Are insights stabilizing or still changing significantly?

**Provide your evaluation:**
```
Consistency Score: [0.0-1.0]
Completeness Score: [0.0-1.0]
Resolution Score: [0.0-1.0]
Stability Score: [0.0-1.0]
Overall Convergence: [0.0-1.0]
Continue Discussion: [yes/no]
Key Issues Remaining: [List of unresolved issues]
Recommendations: [Suggestions for next steps]
```
"""


def build_convergence_evaluation_prompt(
    iterations: int,
    total_insights: int,
    total_questions: int,
    total_responses: int,
    agent_participation: Dict[str, Any],
    quality_trends: str,
    key_patterns: str
) -> str:
    """Build a prompt for evaluating discussion convergence."""
    return CONVERGENCE_EVALUATION_PROMPT.format(
        iterations=iterations,
        total_insights=total_insights,
        total_questions=total_questions,
        total_responses=total_responses,
        agent_participation=agent_participation,
        quality_trends=quality_trends,
        key_patterns=key_patterns
    )


TASK_CREATION_PROMPT = """Create appropriate tasks for the {phase} phase of multi-agent discussion.

**Discussion Context:**
- Current Phase: {phase}
- Iteration: {iteration}/{max_iterations}
- Time Remaining: {time_remaining}
- Agent Workload: {agent_workload}

**Available Agents:**
- critical_evaluator: [{evaluator_status}]
- innovative_insighter: [{insighter_status}]
- practical_applicator: [{applicator_status}]
- theoretical_integrator: [{integrator_status}]

**Task Requirements:**
1. Create tasks appropriate for current discussion phase
2. Balance workload across available agents
3. Consider task dependencies and sequencing
4. Prioritize high-impact tasks
5. Respect agent availability and current workload

**Task Types Available:**
- generate_insights: For initial analysis phase
- ask_question: For questioning phase
- answer_question: For responding phase
- evaluate_convergence: For convergence phase

**Create tasks in this format:**
```
Task 1:
- Type: [task_type]
- Assigned To: [agent_type]
- Priority: [low/medium/high/critical]
- Parameters: [key parameters]
- Dependencies: [any task dependencies]
- Estimated Time: [minutes]

Task 2:
...
```

Focus on tasks that will advance the discussion toward convergence while ensuring all perspectives are heard."""