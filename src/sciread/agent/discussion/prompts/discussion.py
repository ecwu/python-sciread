"""Discussion coordinator prompts for multi-agent system."""

from typing import Any

DISCUSSION_COORDINATOR_SYSTEM_PROMPT = """你是一名多智能体学术论文分析的讨论协调者。你的职责是协调四位专家型智能体展开讨论，以形成对论文的全面理解。

你的核心职责：
- 组织多阶段讨论流程
- 确保每种角色都能贡献独特视角
- 监控讨论进展与收敛情况
- 促进建设性的多方对话
- 在分析深度与执行效率之间保持平衡

四位专家型智能体分别是：
1. **批判性评估者**：识别局限、方法缺陷和潜在薄弱点
2. **创新洞察者**：识别新颖贡献与突破潜力
3. **实践应用者**：评估真实应用价值与落地可行性
4. **理论整合者**：分析理论框架与概念贡献

你的协调策略应确保：
- 所有智能体都有充分发言机会
- 讨论始终围绕论文意义展开
- 提问具有建设性并推动更深层分析
- 在不强行制造共识的前提下实现收敛
- 论文的优势与局限都得到充分讨论

你需要管理以下阶段：
1. **初始分析**：各智能体生成初步洞见
2. **提问阶段**：智能体围绕彼此洞见提出针对性问题
3. **回应阶段**：智能体回答问题，并在必要时修订洞见
4. **收敛评估**：判断是否已达到足够共识
5. **共识构建**：综合形成最终评估

无论在任何阶段，都要保持学术严谨并确保对话高效推进。"""


PHASE_TRANSITION_PROMPT = """请根据当前进展判断是否应进入下一讨论阶段。

**当前阶段：** {current_phase}
**轮次：** {iteration}/{max_iterations}
**已耗时：** {time_elapsed}

**进度指标：**
- 已生成洞见：{total_insights} 条，覆盖 {agents_with_insights} 个智能体
- 已提出问题：{total_questions}
- 已回答问题：{total_answers}
- 平均洞见质量：{avg_insight_quality:.2f}
- 讨论活跃度：{activity_level}

**阶段判定标准：**
{phase_criteria}

**决策框架：**
1. **Advance if**：满足最低要求且讨论仍富有成效
2. **Continue if**：进展稳定，但还需要进一步打磨
3. **Timeout if**：达到最大轮次或超出时间限制

请按以下格式输出建议：
```
Decision: [advance/continue/timeout]
Reasoning: [请用中文说明你的判断理由]
Next Phase: [下一阶段名称，或填写 "current"]
```
"""


INITIAL_ANALYSIS_CRITERIA = """初始分析阶段标准：
- 每个智能体至少生成 2 条洞见（总计至少 8 条）
- 平均重要性评分大于 0.5
- 所有智能体都已参与
- 没有明显缺失的视角
- 洞见质量足以进入提问阶段"""

QUESTIONING_CRITERIA = """提问阶段标准：
- 至少提出 4 个问题（平均每条关键洞见至少对应 1 个问题）
- 问题应具有针对性和建设性
- 问题覆盖多种角色视角
- 没有任何智能体被完全忽略
- 提问能够体现对洞见的认真审阅"""

RESPONDING_CRITERIA = """回应阶段标准：
- 至少 80% 的问题已得到回答
- 回答内容具体且经过思考
- 部分洞见已根据提问进行修订
- 回答能体现对关切点的理解
- 所有被提问的智能体都已获得回应机会"""

CONVERGENCE_CRITERIA = """收敛评估阶段标准：
- 已完成多轮提问与回应
- 新洞见和新问题数量开始下降
- 智能体报告的收敛评分更高
- 已识别出关键共识点
- 剩余分歧被清晰表达"""


def build_phase_evaluation_prompt(
    current_phase: str, iteration: int, max_iterations: int, time_elapsed: str, progress_metrics: dict[str, Any]
) -> str:
    """Build a prompt for evaluating phase progress."""
    phase_criteria_map = {
        "initial_analysis": INITIAL_ANALYSIS_CRITERIA,
        "questioning": QUESTIONING_CRITERIA,
        "responding": RESPONDING_CRITERIA,
        "convergence": CONVERGENCE_CRITERIA,
    }

    phase_criteria = phase_criteria_map.get(current_phase, "请判断当前阶段目标是否已经达成。")

    return PHASE_TRANSITION_PROMPT.format(
        current_phase=current_phase,
        iteration=iteration,
        max_iterations=max_iterations,
        time_elapsed=time_elapsed,
        phase_criteria=phase_criteria,
        **progress_metrics,
    )


CONVERGENCE_EVALUATION_PROMPT = """请评估多智能体讨论的收敛程度，并判断是否已经形成足够共识。

**讨论统计：**
- 已完成轮次：{iterations}
- 总洞见数：{total_insights}
- 总问题数：{total_questions}
- 总回答数：{total_responses}

**智能体参与情况：**
{agent_participation}

**洞见质量趋势：**
{quality_trends}

**关键模式：**
{key_patterns}

**收敛判断指标：**
- 各智能体的洞见是否越来越一致？
- 问答过程是否更多带来 refinement，而不是加剧分歧？
- 主要问题是否已解决，或至少已被清晰识别？
- 当前剩余分歧是根本性的，还是次要的？

**评估框架：**
请对以下维度按 0.0-1.0 打分：

1. **Consistency**：不同智能体的洞见有多一致？
2. **Completeness**：重要方面是否都已得到充分审视？
3. **Resolution**：主要冲突是否已经被处理？
4. **Stability**：洞见是否趋于稳定，还是仍在显著变化？

**输出要求：**
- 解释性内容请使用中文。
- 以下字段标签保留英文以兼容解析流程。

请按以下格式输出：
```
Consistency Score: [0.0-1.0]
Completeness Score: [0.0-1.0]
Resolution Score: [0.0-1.0]
Stability Score: [0.0-1.0]
Overall Convergence: [0.0-1.0]
Continue Discussion: [yes/no]
Key Issues Remaining: [请用中文列出未解决的问题]
Recommendations: [请用中文给出下一步建议]
```
"""


def build_convergence_evaluation_prompt(
    iterations: int,
    total_insights: int,
    total_questions: int,
    total_responses: int,
    agent_participation: dict[str, Any],
    quality_trends: str,
    key_patterns: str,
) -> str:
    """Build a prompt for evaluating discussion convergence."""
    return CONVERGENCE_EVALUATION_PROMPT.format(
        iterations=iterations,
        total_insights=total_insights,
        total_questions=total_questions,
        total_responses=total_responses,
        agent_participation=agent_participation,
        quality_trends=quality_trends,
        key_patterns=key_patterns,
    )


TASK_CREATION_PROMPT = """请为多智能体讨论的 {phase} 阶段创建合适的任务。

**讨论上下文：**
- 当前阶段：{phase}
- 当前轮次：{iteration}/{max_iterations}
- 剩余时间：{time_remaining}
- 智能体负载：{agent_workload}

**可用智能体：**
- critical_evaluator: [{evaluator_status}]
- innovative_insighter: [{insighter_status}]
- practical_applicator: [{applicator_status}]
- theoretical_integrator: [{integrator_status}]

**任务要求：**
1. 为当前阶段创建合适的任务
2. 在可用智能体之间平衡工作量
3. 考虑任务依赖关系与执行顺序
4. 优先安排高影响力任务
5. 尊重智能体可用性与当前负载

**可选任务类型：**
- generate_insights：用于初始分析阶段
- ask_question：用于提问阶段
- answer_question：用于回应阶段
- evaluate_convergence：用于收敛评估阶段

**请按以下格式创建任务：**
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

请聚焦那些既能推进讨论走向收敛、又能确保所有视角都被听见的任务。"""
