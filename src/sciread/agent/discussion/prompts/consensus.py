"""Consensus building prompts for multi-agent discussion system."""

CONSENSUS_BUILDER_SYSTEM_PROMPT = """你是一名学术研究分析中的共识整合专家。你的职责是综合多个专家智能体的洞见，形成对论文平衡而全面的评估。

你的核心职责：
- 将多元视角整合成连贯分析
- 识别共识点与分歧点
- 公平权衡不同观点
- 对论文贡献与局限给出细致判断
- 保持学术严谨与客观性

你要综合的四类专家视角包括：
1. **批判性评估**：方法严谨性、局限与潜在缺陷
2. **创新洞见**：新颖贡献、突破潜力、未来方向
3. **实践应用**：真实世界可用性与落地可行性
4. **理论整合**：概念贡献与理论意义

你的综合分析应当：
- 为每种视角赋予恰当权重
- 明确指出已形成共识和仍存在分歧之处
- 同时尊重论文的优点与不足
- 具体、基于证据，避免空泛描述
- 帮助读者理解论文真正的价值与局限

你不需要强行制造一致意见，而是要诚实呈现讨论中形成的集体理解。"""


SUMMARY_SYNTHESIS_PROMPT = """请基于多智能体对一篇学术论文的讨论，撰写一份综合摘要。

**论文信息：**
标题：{document_title}
摘要：{document_abstract}

**专家分析提炼出的关键洞见：**
{insights_summary}

**高度一致的观点：**
{consensus_points}

**仍在讨论中的问题：**
{divergent_views}

**讨论上下文：**
- 总轮次：{iterations}
- 最终收敛评分：{convergence_score:.2f}
- 生成洞见数：{total_insights}
- 问答交互总数：{total_qa}

**要求：**
1. 写一段 400-600 字的平衡、完整的中文摘要
2. 突出分析中识别出的论文主要贡献
3. 同时承认讨论中发现的优点与局限
4. 明确哪些方面专家形成了共识，哪些分歧仍然存在
5. 帮助读者理解论文在所属领域中的位置
6. 保持学术语气，但表达清晰易懂

**建议结构：**
- **Overview**：简要介绍论文及其主要关注点
- **Key Contributions**：专家分析识别出的主要贡献
- **Methodological Assessment**：对方法路径的批判性评估
- **Significance and Impact**：论文的重要性与影响
- **Limitations and Concerns**：讨论中提出的局限与担忧
- **Future Directions**：潜在后续方向与研究机会

请在尊重各专家视角的前提下，生成一份能够清晰传达论文真实价值的中文综合总结。"""


SIGNIFICANCE_ASSESSMENT_PROMPT = """请基于多智能体综合分析，评估一篇学术论文的整体重要性。

**论文背景：**
标题：{document_title}
领域：[请从论文内容中提炼]

**分析摘要：**
{summary_highlights}

**贡献评估：**
{contribution_analysis}

**共识强度：**
- 收敛评分：{convergence_score:.2f}
- 强共识点数量：{strong_consensus_count}
- 关键分歧数量：{divergent_view_count}

**专家视角：**
{expert_perspectives}

**评估框架：**
请从以下多个维度评估论文的重要性：

1. **Theoretical Significance** (0.0-1.0)：
   - 是否推进了对基础概念的理解
   - 是否挑战或扩展了既有理论
   - 是否提出了新的概念框架

2. **Methodological Innovation** (0.0-1.0)：
   - 是否引入了新方法或新路径
   - 是否改进了既有方法论
   - 是否使新的研究成为可能

3. **Practical Impact** (0.0-1.0)：
   - 是否解决了真实世界问题
   - 是否具有商业或产业应用前景
   - 是否可能影响实践或政策

4. **Field Advancement** (0.0-1.0)：
   - 是否打开了新的研究方向
   - 是否可能影响领域后续工作
   - 是否填补了重要知识空白

5. **Scholarly Value** (0.0-1.0)：
   - 研究是否严谨且执行扎实
   - 是否对学术文献有清晰贡献
   - 是否具有被引用和延展的潜力

**输出要求：**
- 解释性正文请使用中文。
- 评分字段标签保留英文以兼容解析。

请按以下格式输出：
```
Theoretical Significance: [0.0-1.0]
Methodological Innovation: [0.0-1.0]
Practical Impact: [0.0-1.0]
Field Advancement: [0.0-1.0]
Scholarly Value: [0.0-1.0]
Overall Significance: [0.0-1.0]

Significance Assessment:
[请用中文详细解释评分，并同时指出优势与局限]
```

请在给予应有肯定的同时，诚实呈现局限，并同时考虑论文的内在质量与潜在影响。"""


CONTRIBUTION_EXTRACTION_PROMPT = """请从多智能体分析结果中提炼学术论文的关键贡献。

**论文标题：** {document_title}

**专家高价值洞见：**
{top_insights}

**关于贡献的共识：**
{consensus_contributions}

**专家对价值的判断：**
{expert_value_assessments}

**分析标准：**
请优先识别那些满足以下条件的贡献：
- 有明确分析依据支撑
- 获得多个专家视角认可
- 相较既有工作体现真实推进
- 具体明确，而非空泛表述

**可识别的贡献类型：**
1. **理论贡献**：新概念、框架、模型
2. **方法贡献**：新方法、路径、技术
3. **实证贡献**：新发现、新数据、实验结果
4. **实践贡献**：应用、工具、实现方案
5. **知识整合**：以新方式整合已有工作

**输出格式要求：**
- 具体内容请使用中文。
- 结构标签保留如下：
```
Key Contributions:
1. [Contribution title/description]
   - Type: [theoretical/methodological/empirical/practical/integration]
   - Supporting Evidence: [请用中文简述依据]
   - Expert Support: [请用中文说明哪些角色认可该贡献]
   - Significance: [请用中文说明其重要性]

2. [Next contribution...]

Overall Assessment:
[请用中文概括论文对该领域的主要贡献]
```

请重质量而非数量，优先识别最重要的 3-5 项贡献，而不是把所有提到的内容全部罗列出来。"""


DIVERGENT_VIEW_ANALYSIS_PROMPT = """请分析多智能体讨论过程中出现的分歧观点。

**讨论上下文：**
论文标题：{document_title}
收敛评分：{convergence_score:.2f}
智能体总数：4

**已识别出的分歧观点：**
{divergent_views}

**相关问题与挑战：**
{related_challenges}

**分析框架：**
请针对每项分歧分析以下内容：
1. **Nature of Divergence**：具体分歧点是什么？
2. **Agent Positions**：各角色分别持什么观点，原因是什么？
3. **Evidence Base**：各方立场分别有哪些证据支撑？
4. **Resolution Potential**：若获取更多信息，这项分歧是否可能被解决？
5. **Impact Significance**：这项分歧会如何影响整体评估？

**可选分歧类型：**
- **Methodological Disagreements**：对研究方法路径存在不同看法
- **Interpretive Differences**：对结果或含义存在不同解读
- **Value Judgments**：对重要性或影响的评价不同
- **Scope Differences**：对边界或可推广性的判断不同
- **Future-Oriented Disagreements**：对未来潜力的预期不同

**输出格式要求：**
- 具体说明请使用中文。
- 结构标签保留如下：
```
Divergent View 1: [Brief title]
- Agents in Conflict: [请用中文说明冲突各方及其立场]
- Core Disagreement: [请用中文说明核心分歧]
- Evidence Considerations: [请用中文说明各方关键证据]
- Resolution Prospects: [请用中文判断能否解决]
- Impact on Assessment: [请用中文说明对整体评估的影响]

Divergent View 2: [...]

Overall Conflict Assessment:
[请用中文概括这些分歧的性质与重要性]
```

请平衡呈现不同立场，并诚实说明哪些地方尚未形成共识。"""


def build_summary_synthesis_prompt(
    document_title: str,
    document_abstract: str,
    insights_summary: str,
    consensus_points: str,
    divergent_views: str,
    iterations: int,
    convergence_score: float,
    total_insights: int,
    total_qa: int,
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
        total_qa=total_qa,
    )


def build_significance_assessment_prompt(
    document_title: str,
    summary_highlights: str,
    contribution_analysis: str,
    convergence_score: float,
    strong_consensus_count: int,
    divergent_view_count: int,
    expert_perspectives: str,
) -> str:
    """Build prompt for assessing paper significance."""
    return SIGNIFICANCE_ASSESSMENT_PROMPT.format(
        document_title=document_title,
        summary_highlights=summary_highlights,
        contribution_analysis=contribution_analysis,
        convergence_score=convergence_score,
        strong_consensus_count=strong_consensus_count,
        divergent_view_count=divergent_view_count,
        expert_perspectives=expert_perspectives,
    )


def build_contribution_extraction_prompt(
    document_title: str, top_insights: str, consensus_contributions: str, expert_value_assessments: str
) -> str:
    """Build prompt for extracting key contributions."""
    return CONTRIBUTION_EXTRACTION_PROMPT.format(
        document_title=document_title,
        top_insights=top_insights,
        consensus_contributions=consensus_contributions,
        expert_value_assessments=expert_value_assessments,
    )


def build_divergent_view_analysis_prompt(
    document_title: str, convergence_score: float, divergent_views: str, related_challenges: str
) -> str:
    """Build prompt for analyzing divergent views."""
    return DIVERGENT_VIEW_ANALYSIS_PROMPT.format(
        document_title=document_title,
        convergence_score=convergence_score,
        divergent_views=divergent_views,
        related_challenges=related_challenges,
    )
