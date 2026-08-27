<div align="center">

# 🧬 Hy-Agent · MitoEvidence

### 面向 β 细胞线粒体研究的可追溯证据综述与评估方案

**Claim–Evidence–Condition Evaluation for Auditable Scientific Synthesis**

![Status](https://img.shields.io/badge/status-integration%20plan-4C78A8)
![Priority](https://img.shields.io/badge/priority-evaluation--first-E45756)
![Model](https://img.shields.io/badge/model-Hy3-7A5195)
![Platform](https://img.shields.io/badge/platform-OpenCompass%20RC10-1677FF)
![Scope](https://img.shields.io/badge/scope-research%20only-2A9D8F)
![License](https://img.shields.io/badge/license-Apache--2.0-3C8D40)

[![Mito Agent](https://img.shields.io/badge/Mito--Agent-online%20%7C%20pending%20integration-00A67E)](https://agent.blueskun.com:8444/)

<sub>腾讯犀牛鸟 2026 · 开放式场景：AI 应用与评判标准设计 · 方案版本 V1.1 · 2026-08-27</sub>

</div>

---

> [!IMPORTANT]
> 本文是一份实施与预注册方案，不是实验结果报告。文中所有百分比、阈值和性能指标均为建议验收线，必须在开发集校准后冻结；未达到目标时应如实报告，不得把计划或预期包装成已验证结论。

> [!NOTE]
> 本方案以作者手稿为事实与需求主线；模型生成的两份策略仅作为备选设计输入。现有 OpenCompass 离线评测平台与产品手册已经过只读核验，Mito-Agent 的“已落地、待接入”状态来自项目方说明；凡无法由仓库、运行日志或可核验材料确认的效果结论，仍统一标记为“待核验”。

## 快速导航

- [1. 项目决策摘要](#1-项目决策摘要)
- [2. 场景、用户与产品边界](#2-场景用户与产品边界)
- [3. 设计思路与技术取舍](#3-设计思路与技术取舍)
- [4. 总体架构](#4-总体架构)
- [5. 重点技术](#5-重点技术)
- [6. MitoEvidence-Eval 评估方法](#6-mitoevidence-eval-评估方法)
- [7. 实验设计与统计方案](#7-实验设计与统计方案)
- [8. 已构建评测平台与接入方案](#8-已构建评测平台与接入方案)
- [9. 预期效果与验收线](#9-预期效果与验收线)
- [10. 时间规划](#10-时间规划)
- [11. 风险、降级与边界](#11-风险降级与边界)
- [12. 交付物与仓库结构](#12-交付物与仓库结构)
- [13. 两分钟 Demo 脚本](#13-两分钟-demo-脚本)
- [14. 启动清单](#14-启动清单)
- [附录 A：策略取舍记录](#附录-a策略取舍记录)
- [附录 B：术语表](#附录-b术语表)
- [附录 C：参考依据](#附录-c参考依据)

---

## 1. 项目决策摘要

### 1.1 一句话定位

MitoEvidence 不是“再做一个医学聊天机器人”，而是以现有 `hy-agent`、已部署 Mito-Agent 与离线评测平台为工程底座，将证据绑定能力扩展为：

> 一个面向 β 细胞线粒体科研人员的可追溯快速证据综述 Agent，以及一套先经过独立验证、再用于比较 Agent 版本的证据中心评估方法 MitoEvidence-Eval。

### 1.2 核心研究命题

> 将医学综述拆成可回到论文原文、并保留完整实验条件的 Claim–Evidence–Condition（CEC）单元后，一个由确定性规则、证据约束 Judge、Hard Gate 与专家仲裁组成的评估器，能否可靠识别引用伪造、证据错配、条件偷换、冲突遗漏和过度推断；在这把“可信尺子”监督下，混合检索、逐主张核验和拒答能否相较直接生成与普通 RAG 可复现地降低严重证据错误。

### 1.3 最终取舍

| 层级 | 本期内容 | 决策 |
|---|---|---|
| **T1 · 必做** | EvidenceSpan、CEC、混合检索、逐主张核验、拒答、评估器、挑战集、复现实验 | 本项目主线，缺一不可 |
| **T2 · 门槛增强** | 条件化证据索引、证据图导航、冲突证据扩展 | 仅在 T1 的失败归因证明“条件匹配”是主要瓶颈时进入 |
| **T3 · 研究展望** | KGE、SFT/GRPO 路由、真正图像语义理解、跨领域扩展 | 不进入本期主结论与关键路径 |

### 1.4 四个主要产出

| 产出 | 内容 | 成功标准 |
|---|---|---|
| 📊 **能力基线** | 通用知识、数学理工、多语言、代码、推理、上下文六类回归 | 只在同协议、同快照下形成可比较证据，不与医学分数混算 |
| 🧪 **评估方法** | MitoEvidence-Eval、十维 Rubric、SEER、Hard Gate、挑战集 | 能复现专家判断并识别受控错误 |
| 🔬 **科研应用** | 带原文锚、条件矩阵、冲突分区和拒答的快速证据综述 | 相对基线降低严重证据错误 |
| 📦 **复现工程** | 配置化评测、不可变快照、完整 Episode、结果与 Demo | 任一正式分数均可回到版本、输入、轨迹与原始证据 |

---

## 2. 场景、用户与产品边界

### 2.1 目标用户

1. β 细胞、胰岛和线粒体代谢方向的基础医学研究者；
2. 显微成像与线粒体形态定量分析人员；
3. 需要快速完成证据摸底、条件比较和引用核查的研究生及科研助理。

首要用户为第 3 类。他们最频繁地执行“先判断某个机制是否有证据”的任务，也最容易被语言流畅但证据错误的综述误导。

### 2.2 典型任务

**示例问题**

> 高糖条件下，β 细胞线粒体碎片化与胰岛素分泌障碍之间有哪些已发表证据？不同实验模型的结论是否一致？

**系统输入**

- 自然语言研究问题；
- 可选的物种、细胞类型、扰动、剂量、时长、方法与结局约束；
- 可选的指定论文或本地公开材料；
- 检索时间范围和文献类型限制。

**系统输出**

- 可复现检索式、数据库、日期和语料快照；
- 证据矩阵及支持、反驳、证据不足三分区；
- 逐原子主张引用的综述正文；
- 可点击回到 DOI/PMID/PMCID、章节、页码、图表号和原文片段的证据链；
- 每条结论成立的实验条件；
- 不确定性、拒答原因与人工复核标记；
- 完整运行轨迹与自动评估结果。

### 2.3 明确边界

> [!WARNING]
> 本产品仅用于科研证据导航，不用于患者诊断、治疗决策或个体化临床建议。

- 产品名使用“可追溯快速证据综述”，不宣称“自动系统综述”；
- 只有在完整实现检索协议、纳排流程、偏倚风险评价和 PRISMA 报告后，才可升级系统综述表述；
- 只能宣称定位图表、读取图题/图注/OCR 文本，不宣称理解显微图像的生物学语义；
- 结论只对冻结语料快照成立，“库内未找到”不等于“领域中不存在”；
- KGE 或模型推断产生的边只能标记为待核验假设，不能冒充已发表证据。

### 2.4 当前基础与待核验项

| 状态 | 能力/资产 | 处理方式 |
|---|---|---|
| ✅ 源码与手册可核验 | OpenCompass + Eval-Dominator 离线评测平台：模型/数据管理、六类目录、任务编排、题级分析、日志产物、报告与独立 Code Runner | 作为评测与实验编排底座复用；RC10 安全与发布门禁另行收口 |
| ✅ 已落地、待接入 | [Mito-Agent 在线实例](https://agent.blueskun.com:8444/) | 作为真实医学实验与文献综述 Agent；补齐机器可调用接口、Trace 导出与评测适配器 |
| ✅ 现有能力、待迁移核验 | Hy3 API 接入、证据绑定的多文档问答、子目标 DAG、引用白名单、单轮修复、拒绝未知引用、自动化测试与 Demo | 迁入 `hy-agent` 后逐项以测试和运行记录确认 |
| 🟡 手稿报告、待入库核验 | MinerU、OvisOCR2、自有推理端点、约 3000 篇候选文献 | W1 提供版本、健康检查、样例输出和 SHA-256 后才能标记已完成 |
| 🟠 实现中/待开发 | CEC 抽取、EvidenceSpan 物理锚、十维 Rubric、评估执行器 | T1 主线 |
| ⚪ 未开始 | MedicalQuestionSet、EvaluatorChallengeSet、专家标注与元评估 | T1 主线 |
| 🔵 条件增强 | 证据图、KGE、GRPO、图像级检索 | 达门槛后再立项 |

### 2.5 当前入口与凭据边界

| 入口 | 当前状态 | 本期动作 |
|---|---|---|
| `hy-agent` GitHub 仓库 | 已建立 | 作为公开方案、接口契约、评测配置与复现材料的唯一主仓库 |
| [Mito-Agent](https://agent.blueskun.com:8444/) | 已部署、待评测接入 | 接入 AppAdapter，导出结构化 Episode 与工具轨迹 |
| OpenCompass 离线评测平台 | RC10 功能底座已构建 | 新增医学数据集、Agent 与 Evaluator 适配器，保留六类能力回归 |

> [!SECURITY]
> 在线实例需要授权访问。账号、密码、API Key、令牌和内网地址只通过受控私密渠道交接，禁止写入公开 README、Issue、截图、日志或 Git 历史。

---

## 3. 设计思路与技术取舍

### 3.1 先验证“尺子”，再比较“系统”

评估器不是项目末尾附加的 Judge，而是主要研究对象。正确顺序是：

~~~text
定义错误与边界
  → 构建金标与挑战集
  → 验证评估器的判别力、一致性、稳定性和抗投机性
  → 冻结评估器
  → 用冻结评估器比较 Agent 版本
~~~

如果顺序反过来，系统提升将建立在一把未经验证的尺子上，无法得出可信结论。

### 3.2 从文档块升级为 CEC

普通 RAG 的最小单元是 chunk；MitoEvidence 的最小可信单元是：

~~~text
原子主张 Claim
  ↕ 支持 / 反驳 / 部分支持 / 不确定
原文证据 EvidenceSpan
  ↕ 成立条件
物种 · 样本 · 细胞 · 扰动 · 剂量 · 时长 · 方法 · 结局 · 效应方向
~~~

这使“引用存在但不支持结论”“论文结论正确但实验条件被偷换”成为可计算、可审计的错误，而不是模糊的语言质量问题。

### 3.3 大模型与确定性组件的分工

| 交给 Hy3/语义模型 | 交给确定性程序 | 交给专家 |
|---|---|---|
| 问题拆解、查询改写、证据综合、主张拆分、语义蕴含候选 | 标识符解析、Schema、原文位置、单位、格式、快照、预算、哈希 | 因果边界、冲突裁决、Hard Gate、低置信样本、金标 |

任何可确定性验证的内容不交给 LLM 主观判断；任何高风险边界不由 LLM 单独否决。

### 3.4 防止循环评测

- Runtime 与 Evaluator 使用不同 Prompt、不同上下文和独立配置；
- Judge 只能读取冻结 EvidenceSpan，不得联网自行补知识；
- 系统名称匿名，避免品牌偏见；
- L1 规则与 L4 专家不依赖生成模型；
- 保留双人独立标注和争议裁决；
- 若 Judge 与生成均使用 Hy3，必须明确“逻辑隔离但非模型独立”，并以专家一致性作为最终锚点。

### 3.5 模型策略的主要修正

| 原策略建议 | 本方案处理 | 原因 |
|---|---|---|
| 全量运行通用 Benchmark | 降为附录回归 | 与真实用户场景不是同一构念 |
| 3000 篇顶刊建图 | 改为可复现高召回检索 + 证据质量分层 | 避免期刊与正向结果选择偏差 |
| 超图、KGE、GRPO 同期上线 | 分为 T2/T3 | 避免范围失控和消融混杂 |
| 图文混合检索 | MVP 只做正文、图题、图注、表格与 OCR | Hy3 非原生视觉模型，不夸大图像理解 |
| OpenCompass 作为医学总分平台 | 只复用配置、适配器、推理/评估分离和结果汇总思想 | 场景 Rubric 与通用榜单不得混合 |

---

## 4. 总体架构

### 4.1 四平面架构

~~~mermaid
flowchart TB
    subgraph U["① 用户与审阅平面"]
        U1["科研问题 + 条件约束"]
        U2["证据矩阵 / 逐主张综述"]
        U3["点击回原文 / 专家复核"]
    end

    subgraph R["② 在线 Agent 运行平面"]
        R0["Mito-Agent Web/API<br/>已落地 · 待接入"]
        R1["查询理解与条件抽取"]
        R2["受预算控制的检索器"]
        R3["BM25 + Dense + Rerank + 条件过滤"]
        R4["多文献证据综合"]
        R5["逐主张核验"]
        R6{"决策"}
        R7["ANSWER"]
        R8["ABSTAIN"]
        R9["HUMAN_REVIEW"]

        R0 --> R1 --> R2 --> R3 --> R4 --> R5 --> R6
        R6 --> R7
        R6 --> R8
        R6 --> R9
        R5 -.补检一次.-> R2
    end

    subgraph G["③ 只读证据与快照平面"]
        G1["Corpus Snapshot"]
        G2["EvidenceSpan Index"]
        G3["CEC Store"]
        G4["Model / Prompt / Tool Manifest"]
        G5["SHA-256 Artifact Manifest"]
    end

    subgraph E["④ 离线评测与控制平面"]
        E0["OpenCompass + Eval-Dominator<br/>已构建评测平台"]
        E1["冻结 EvidenceReviewEpisode"]
        E2["L1 确定性规则"]
        E3["L2 证据约束 Judge"]
        E4["L3 Hard Gate"]
        E5["L4 专家仲裁"]
        E6["SEER + D1-D10 + 归因 + 成本"]
        E7["Adapter / Task / Attempt / Cache / Report"]

        E0 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7
    end

    U1 --> R0
    R7 --> U2
    R8 --> U2
    R9 --> U3
    G -.只读接地.-> R
    G -.版本核验.-> E
    R7 -.冻结.-> E1
    R8 -.冻结.-> E1
    R9 -.冻结.-> E1
    U3 --> E5
~~~

### 4.2 三条架构硬规则

1. **ANSWER、ABSTAIN、HUMAN_REVIEW 是同级动作。**“不确定时不行动”是一种正确决策，必须单独评分。
2. **密封测试只进 Evaluator，不回流 Runtime。**测试答案、证据组和评分反馈不得进入 Prompt、检索调参或训练。
3. **任何正式结果必须绑定不可变身份。**模型、语料、索引、Prompt、工具、Runner、随机种子和评估器版本缺一不可。

### 4.3 数据流

~~~mermaid
sequenceDiagram
    participant P as 论文/开放全文
    participant I as Ingestion
    participant C as CEC Store
    participant A as Mito-Agent / Hy3 Runtime
    participant V as Verifier
    participant E as OpenCompass / MitoEvidence-Eval
    participant H as Expert

    P->>I: PDF/XML/元数据
    I->>I: 解析、OCR、去重、撤稿/勘误检查
    I->>C: EvidenceSpan + Condition + Provenance
    A->>C: 受约束检索
    C-->>A: 候选证据与物理锚
    A->>V: 原子主张 + 引用 + 条件
    V-->>A: 通过 / 补检 / 降级 / 删除 / 拒答
    A->>E: 冻结 Episode
    E->>H: 低置信、冲突与门禁样本
    H-->>E: 盲评与裁决
    E-->>A: 仅开发集可返回诊断；密封集禁止回流
~~~

### 4.4 核心数据契约

| 对象 | 必需字段 | 作用 |
|---|---|---|
| TaskSpec | question_id、题型、难度、answerability、约束 | 定义评测任务 |
| EvidenceSpan | paper_id、section、page、paragraph、figure/table、bbox、text、parser_version | 把结论定位回原文 |
| CECRecord | claim、stance、evidence_span、conditions、uncertainty、review_status | 表达结论及成立条件 |
| EvidenceReviewEpisode | TaskSpec、环境清单、检索轨迹、输出、主张、证据、成本 | 冻结一次完整运行 |
| AcceptableEnvelope | 必答子问题、允许结论、替代证据组、必须保留条件、禁止外推 | 避免把开放综述简化为唯一标准答案 |
| EvaluationResult | SEER、D1-D10、门禁、错误标签、各层理由、人工状态 | 形成可复查评测结果 |

---

## 5. 重点技术

### T1-1 · 结构化解析与证据锚

**目标**：让每个关键引用不仅回到论文，还能回到具体原文位置。

**流程**

1. DOI/PMID/PMCID 元数据解析与去重；
2. PDF/XML 结构化解析；
3. 图题、图注、表格和 OCR 文本对齐；
4. 生成页码、章节、段落、图表号和 bbox；
5. 保存解析器版本、置信度和文档哈希；
6. 解析失败进入隔离清单，不静默跳过。

**建议验收**

- 金标集中的关键 EvidenceSpan 100% 可回原文；
- 自动摄取样本的定位正确率建议不低于 95%，低于阈值自动转人工；
- 文档内容变化一字节后必须产生新的语料/索引指纹。

### T1-2 · CEC 抽取与条件规范化

**建议 Schema**

~~~json
{
  "claim_id": "CLAIM-007",
  "claim_type": "PUBLISHED_ASSERTION",
  "claim_text": "...",
  "evidence_span_ids": ["SPAN-0142"],
  "stance": "support",
  "conditions": {
    "species_or_sample": "...",
    "cell_type": "...",
    "perturbation": "...",
    "dose": "...",
    "duration": "...",
    "method": "...",
    "outcome": "...",
    "effect_direction": "..."
  },
  "inference_level": "association",
  "review_status": "human_checked"
}
~~~

**规则**

- 已发表主张必须绑定至少一个 EvidenceSpan；
- 支持、反驳、部分支持和不确定均可表达；
- 物种、细胞模型和效应方向为高权重槽位；
- 单位转换保存原始值和标准化值；
- 研究设计、比较组、样本量、效应量、统计不确定性和撤稿/勘误状态作为增强字段；
- 推断候选统一标记为 HYPOTHESIS_PENDING_REVIEW，不进入正式答案。

### T1-3 · 可复现语料与池化金标

开放式文献检索不存在天然完备的召回率分母。本方案采用：

1. PubMed / Europe PMC / PMC OA 等可复现检索源；
2. 记录检索式、数据库、日期、版本、纳排原因和全文许可；
3. 用 BM25、Dense、人工检索等多路结果建立 pooled evidence；
4. 由专家判断池中哪些 span 属于可接受证据组；
5. Recall@K 的分母只解释为“冻结语料与池化金标范围内的证据组”。

期刊、年份和引用数只作为分层变量，不作为证据真值。

### T1-4 · 混合检索与条件过滤

~~~text
BM25 专业词召回
  + Dense 语义召回
  → RRF / Cross-encoder 重排
  → 物种、细胞、剂量、时长、方法过滤
  → 图题/图注/表格通道补充
  → 证据组去重与上下文预算控制
~~~

文本向量与知识图谱向量不构成“Word2Vec → Transformer → RotatE”的替代链。MVP 使用文本检索与结构化条件；KGE 仅在 T3 研究。

### T1-5 · 逐主张核验、补检与拒答

对每个原子主张执行：

1. 标识符与元数据验证；
2. 原文位置存在性验证；
3. 主张—证据蕴含判断；
4. 条件槽逐项比对；
5. 支持 / 部分支持 / 反驳 / 不确定分类；
6. 不通过时执行一次受预算补检，或降级措辞、删除主张；
7. 核心证据仍不足时 ABSTAIN，并根据风险进入 HUMAN_REVIEW。

API 暂时不可解析的引用标记 UNRESOLVED，不能直接判定伪造。

### T1-6 · 四层评测执行引擎

| 层 | 内容 | 为什么需要 |
|---|---|---|
| L1 确定性规则 | 标识符、位置、Schema、单位、格式、预算、哈希 | 可复现、低成本、无主观波动 |
| L2 证据约束 Judge | 主张拆分、蕴含、条件、冲突、推断层级 | 处理规则难以覆盖的语义关系 |
| L3 Hard Gate | 严重证据错误的一票否决 | 防止流畅度和篇幅抵消医学错误 |
| L4 专家仲裁 | 金标、边界、门禁、低置信和冲突 | 为自动评估提供人工锚点 |

### T1-7 · 确定性变形算子与 Challenge Set

每次变形都保存 mutation manifest：

~~~json
{
  "mutation_id": "MUT-0431",
  "parent_episode": "EP-0087",
  "operator": "species_swap",
  "target": "CLAIM-007.conditions.species_or_sample",
  "before": "rat cell line",
  "after": "human primary islet",
  "expected_dimension": "D5",
  "expected_gate": "FAIL",
  "human_verified": true
}
~~~

**四类变形**

- 破坏性：伪造 DOI、引用错配、物种/细胞偷换、剂量漂移、方向反转、因果越级、删反证；
- 等价性：同义改写、引用重编号、段落重排、主张拆分；
- 投机性：篇幅膨胀、术语堆砌、重复引用、全篇模糊限定；
- 证据干预：移除关键论文、注入冲突研究，要求下游局部改变而非整篇重写。

### T2 · 条件化证据索引 / 证据图

第一阶段先用关系表或 JSON 实现 Paper–Experiment–Claim–EvidenceSpan–Condition，不强制部署图数据库，也不先包装成“超图创新”。

**进入 T2 的建议门槛**

- 在开发集上将 K 加倍后，Evidence Recall 增量小于 2 个百分点，说明普通召回接近平台期；
- 剩余严重错误中，CONDITION_MISMATCH 或 CONFLICT_OMISSION 占比不低于 30%；
- CEC 关键槽位通过人工审计；
- 条件图必须在固定 Top-K、Token 和延迟预算下与 T1 公平比较。

### T3 · KGE 与策略学习

KARL、ToolOmni 和 AgentGL 可作为图导航与工具策略学习的研究参考，但不能证明其在本场景必然有效。

进入 T3 前必须满足：

- T1/T2 评估器、奖励、数据切分和工具协议均冻结；
- 规则 Router 已有稳定基线；
- 训练轨迹与密封测试完全隔离；
- 至少三次独立种子实验无严重安全回退；
- 任何预测边都必须回到已发表 EvidenceSpan 后才能进入答案。

---

## 6. MitoEvidence-Eval 评估方法

### 6.1 双轨评测：底座能力与医学 Agent 能力分开

评测采用“两类能力轨道 + 一条元评估门禁”，避免用公共考试分数替代真实医学任务：

| 轨道 | 回答的问题 | 评测对象 | 结果呈现 |
|---|---|---|---|
| **A · 六类基础能力回归** | 模型升级后，通用能力是否退化？ | 固定模型服务与公共 Benchmark | 六类能力向量、逐基准原生指标、协议与覆盖率 |
| **B · 医学 Agent 场景评测** | Agent 能否给出条件保真、证据可追溯的医学实验与文献综述？ | Mito-Agent 全链路 Episode | SEER、D1–D10、Hard Gate、成本、失败归因 |
| **C · 评估器元评估** | 这把“尺子”本身是否可靠？ | EV0–EV4 与 Challenge Set | 专家一致性、判别力、稳定性、攻击检出与不变性 |

轨道 A 先建立底座回归画像；轨道 C 先于轨道 B 的正式系统比较完成并冻结。三条轨道不得压成一个可相互抵消的总分。

### 6.2 六类基础能力与医学扩展

现有离线评测平台已登记六类目录与 21 个 benchmark family。本项目复用目录与执行链，并为医学中文、线粒体图像分割方法和长文献上下文新增领域数据集；新增题库与公共 Benchmark 分开报告。

| 能力 | 现有公共基线 | 本项目领域化任务 | 主要判定 |
|---|---|---|---|
| **通用知识** | MMLU、MMLU-Pro、C-Eval、CMMLU、AGIEval | 科研常识、实验设计术语、证据等级与错误边界；不把期刊档次当真值 | Accuracy / Macro，分学科覆盖与错误类型 |
| **数学与理工科任务能力** | GSM8K、GPQA、SuperGPQA | 生物能量学、显微分辨率与采样、剂量/时间/单位换算、统计与效应方向判断 | Exact Match / Accuracy；过程约束与单位一致性 |
| **多语言任务能力（医学和中文）** | MGSM、MMMLU；C-Eval/CMMLU 作为中文交叉参照 | 中文医学问题、英文学术证据检索、中英术语对齐、双语主张—证据一致性 | 任务正确率、双语一致率、术语规范、CEC 支持正确性 |
| **代码任务能力** | EvalPlus、CRUXEval、BigCodeBench；MultiPL-E 保持门禁状态 | 线粒体图像分割原理判断；语义/实例与 2D/3D 方案选择；拓扑、边界、类别不平衡及指标设计；“底核方法设计”按项目题库定义生成算法、伪代码与测试 | 原理 Rubric + pass@1/单测 + 可复现性；执行代码仅进入隔离 Runner |
| **推理能力** | BIG-Bench Hard、DROP、CLUEWSC2020、GPQA-Diamond | 跨论文证据链、相关/机制/因果层级、冲突证据与异质性、应答/拒答决策 | Exact/F1 + CEC 逻辑一致性 + Hard Gate |
| **上下文能力** | LongBench、LongBench v2 | 单篇长论文、多论文证据池、图题/图注/表格定位、跨段条件保持与中间位置证据召回 | Evidence Recall@K、答案正确性、引用支持率、位置分桶 |

> [!NOTE]
> `smoke10` 只用于验证端到端链路，不进入正式报告；HKCanto-Eval、MultiPL-E 等受许可或 Runner 门禁约束的项目保持不可提交，不因本地存在文件而宣称“已支持”。

### 6.3 两个样本集严格分离

| 数据集 | 评什么 | 建议规模 | 标签来源 |
|---|---|---:|---|
| MedicalQuestionSet | Agent 系统版本 | Pilot 100 题：40 开发 + 60 密封 | 真实科研问题、池化证据、双人盲评与裁决 |
| EvaluatorChallengeSet | 评估器版本 | ≥ 60 个变形家族，每组 1 个原始 + N 个变形 | mutation manifest + 人工复核 |

Challenge Set 的原始输出至少来自三种来源：专家撰写、Hy3 运行输出、独立生成风格或历史错误样本，避免评估器只识别某种模板。

### 6.4 题型覆盖

| 题型 | 建议比例 | 主要考察 |
|---|---:|---|
| 单篇事实与原文定位 | 15% | 物理证据锚 |
| 实验条件与方法比较 | 20% | 条件保真 |
| 跨论文机制综合 | 20% | 核心结论覆盖 |
| 相互冲突的研究 | 15% | 反证与异质性 |
| 证据不足、应拒答 | 15% | 拒答校准 |
| 图题、图注或表格证据 | 15% | 非正文证据通道 |

问题按论文、机制主题和时间分组切分；同一论文或近似模板不得跨开发集与密封集。

### 6.5 十维 100 分 Rubric

原九维方案缺少对“是否真正回答问题、核心内容是否完整”的足够约束。本方案升级为十维，并将检索过程指标从最终答案 Rubric 中分离。

| 维度 | 权重 | 可操作判定 |
|---|---:|---|
| D1 任务正确性与核心覆盖 | 12 | 必答子问题、关键结论和 AcceptableEnvelope 覆盖；非答所问不得靠正确引用得高分 |
| D2 来源真实性与可追溯性 | 12 | 标识符有效且关键引用可回原文位置 |
| D3 主张—证据支持正确性 | 18 | 完全支持 1、缺非核心限定 0.5、无关/反驳/缺核心条件 0 |
| D4 引用完整性 | 8 | 有有效支持证据的可验证主张权重 / 全部可验证主张权重 |
| D5 实验条件与效应保真 | 15 | 物种、细胞、方向权重 2；扰动、剂量、时长、方法、结局权重 1 |
| D6 证据覆盖与冲突平衡 | 10 | 按黄金证据组计；反对/不一致证据组权重 2 |
| D7 推断校准与拒答 | 10 | 不越过观察→相关→机制→因果→临床层级；答/拒决策正确 |
| D8 术语与实体规范 | 5 | 标准名称、同义词解析、基因/蛋白及缩写消歧 |
| D9 可复现性与格式 | 6 | 快照、检索记录、逐主张引用、局限与审核状态可复查 |
| D10 科研人员可用性 | 4 | 直接回答、条件可见、证据分区、待复核醒目、无无关堆砌 |
| **合计** | **100** | 诊断分数，不替代主指标或 Hard Gate |

题型间不在运行时随意重归一。每类题使用预注册的适用维度配置，同时保留完整维度向量。

### 6.6 主指标：SEER

~~~text
SEER = 含严重证据错误的关键输出主张数 / 可评估关键输出主张数

严重错误包括：
伪造标识符、引用错绑、条件偷换、效应方向反转、
关键反证遗漏、无证据因果/临床外推
~~~

**零分母规则**

- 可回答问题却没有关键主张：SEER 记 NA，同时 D1=0 并触发 NON_RESPONSIVE；
- 不可回答问题且正确拒答：SEER=0，主要由 D7 评价；
- 报告 SEER 时必须同时报告 D1 和 Hard Gate，防止靠“少说或不说”人为降低错误率。

### 6.7 Hard Gate

以下任一情况直接 FAIL，与总分并行：

1. 经双源或人工确认的伪造 DOI、PMID 或论文；
2. 关键结论没有任何可定位支持证据却使用确定性表述；
3. 核心主张被所引原文直接反驳；
4. 核心结论发生物种、细胞模型或效应方向偷换；
5. 把相关性写成确定因果并改变核心结论；
6. 证据不足问题未拒答；
7. 从基础实验直接生成无支持的临床建议。

建议状态：

| 状态 | 条件 |
|---|---|
| PASS | 总分与关键维度达到开发集校准阈值，且无门禁 |
| REVIEW | 总分处于边界、关键维度偏低或 Judge 低置信，但无硬失败 |
| FAIL | 触发 Hard Gate，或总分低于冻结阈值 |

80/60 等分界只能作为初始候选，必须按专家接受/拒绝结果校准后冻结。

---

## 7. 实验设计与统计方案

### 7.1 系统单因素消融

为避免一次加入多个模块导致无法归因，采用单因素递增：

| 版本 | 设置 | 研究问题 | 层级 |
|---|---|---|---|
| S0 | Hy3 直接生成 | 无检索基线 | T1 |
| S1 | S0 + 固定 Top-K Dense RAG | 普通向量 RAG 的贡献 | T1 |
| S2 | S1 + BM25 + 融合/重排 | 混合召回与精排贡献 | T1 |
| S3 | S2 + 条件过滤 | 实验条件结构化的贡献 | T1 |
| S4 | S3 + 逐主张核验 + 一次补检 + 拒答 | 核验与安全决策的贡献 | T1 |
| S5 | S4 + 条件化证据图 | 图导航与冲突扩展贡献 | T2，过门槛才做 |
| S6 | S5 + 学习型 Router | 策略学习贡献 | T3，不进本期主结论 |

主结论只依赖 S0–S4。S5 未完成不会破坏交付。

### 7.2 评估器消融

| 版本 | 组成 | 用途 |
|---|---|---|
| EV0 | 整体 LLM Judge 总分 | 常见基线 |
| EV1 | 纯确定性规则 | 客观项基线 |
| EV2 | 规则 + 逐主张证据 Judge | 检验 CEC 价值 |
| EV3 | EV2 + 条件核验 + Hard Gate | 完整自动评估器 |
| EV4 | EV3 + 低置信专家仲裁 | 半自动上限 |

比较专家一致性、严重错误召回、干净样本误报、重复波动、成本与延迟。

### 7.3 评估方法有效性验证

| 验证 | 设计 | 主要指标 |
|---|---|---|
| 判别力 | 每题高/中/低三档，长度相近且匿名打乱 | 完整排序、Pairwise Accuracy、Kendall τ |
| 人工一致性 | 至少 2 名领域研究者独立盲评；关键子集建议第 3 人 | 加权 κ、ICC(2,1)、Spearman ρ、MAE |
| Judge 稳定性 | 同一输出同配置独立评估 ≥ 3 次 | 分数标准差、门禁翻转率 |
| 系统生成稳定性 | 同一问题多种随机种子重复生成 | SEER 与维度方差 |
| 对抗性 | 破坏性与提示注入攻击 | 检出率、严重错误假阴性率、干净误报 |
| 等价不变性 | 改写、重排、编号变化 | 分数差、门禁翻转 |
| 局部干预 | 移除关键论文或加入冲突论文 | 相关主张改变、无关主张稳定 |
| 构念效度 | 单维度受控扰动 | 目标维度下降集中度 |

### 7.4 公平性控制

- S0–S5 固定 Hy3 权重、服务版本、推理模式、温度和输出上限；
- S1–S5 使用同一语料快照，S5 使用冻结图快照；
- 每个版本预注册 Top-K、Token、步骤、延迟和工具调用预算；
- 提示词、检索参数、Judge Prompt、Rubric 与阈值在密封测试前冻结；
- 所有失败、空输出、超时和拒答都进入结果；
- 人工不得在查看金标后修改自动系统输出；
- 人机协同版本单列，并报告人工分钟数、修改类型和总成本。

### 7.5 统计方案

- 主要分析单元为 question / episode，不把同一题的多个变形当作独立样本；
- 配对比较按问题聚类 bootstrap 95% CI；
- 序数维度使用加权 Cohen κ；
- 连续总分使用绝对一致性 ICC(2,1)；
- 自动—人工排序使用 Spearman ρ 与 Kendall τ；
- 多系统比较预先指定主对比 S4 vs S2，其余使用 Holm 校正或标为探索性；
- Pilot 100 题用于可行性与方差估计；正式推广性结论需基于 Pilot 后的功效分析扩样；
- 置信区间跨 0 时只报告趋势，不宣称显著提升。

---

## 8. 已构建评测平台与接入方案

本项目不再从零建设评测后台。现有“天枢星·司南”离线平台已采用 OpenCompass 作为推理与评分引擎，并由 Eval-Dominator 提供模型、数据、任务、产物与报告工作流。本期重点是把 Mito-Agent 作为可回放的 Agent 被测对象接入，并在同一平台中并列承载“六类基础能力回归”和“MitoEvidence 医学场景评测”。

### 8.1 当前平台证据

<p align="center">
  <img src="assets/opencompass-evaluation-platform.png" alt="OpenCompass 离线评测平台任务详情页" width="920" />
</p>

<p align="center"><sub>现有评测平台任务详情示例：可见准备、构建、推理、评测阶段，以及概览、指标、题级分析、产物和日志入口。</sub></p>

> [!IMPORTANT]
> 截图中的 200 题、7 个子集和 92.0% 是既有平台某次任务的界面示例，只证明工作流和结果展示已经落地，不是 MitoEvidence 的实验结果，也不得用于宣称本方案已达到预期效果。

### 8.2 已具备的产品能力

| 能力 | 已有实现 | 本项目如何复用 |
|---|---|---|
| 模型接入 | OpenAI-compatible `/v1/models` 与最小 Chat 协议探测；配置文件/预设管理 | 登记 Hy3 服务身份、上下文与生成预算，保留服务 revision/digest |
| 六类目录 | 6 类、21 个 benchmark family；绑定规模、指标、协议、许可与关联 | 作为底座回归画像；医学自定义集保持独立分组 |
| 数据治理 | CSV/TSV 字段识别、结构校验、revision、manifest 与 SHA-256 | 导入 MedicalQuestionSet、ChallengeSet 与 mutation manifest |
| 任务编排 | master/subtask、full/smoke10、preflight、缓存、断点续跑与仅评分入口 | 编排 S0–S4、EV0–EV4 与不同 seed/预算 |
| 过程可见 | 准备、构建、推理、评测/解析阶段；概览、指标、题级分析、产物、日志 | 展示 Episode、CEC、门禁原因、失败归因与专家队列 |
| 能力报告 | 六类能力、覆盖基准数、原生主指标、样本数、区间、任务与协议指纹 | 新增 SEER、D1–D10、Hard Gate、成本与配对统计视图 |
| 代码执行 | 独立 Code Runner 与运行时 profile | 承载分割算法代码题的单测；生成代码不在平台主进程执行 |
| 离线交付 | 源码/数据与 platform/runner 镜像分卷、校验、恢复和验收流程 | 形成可离线复现实验包与 Artifact Manifest |

### 8.3 Mito-Agent 接入架构

~~~mermaid
flowchart LR
    Q["MedicalQuestionSet / ChallengeSet"] --> OA["OpenCompass DatasetAdapter"]
    OA --> MA["MitoAgentAdapter"]
    MA --> AG["Mito-Agent<br/>已部署 · 待接入"]
    AG --> TR["Trace Normalizer"]
    TR --> EP["EvidenceReviewEpisode"]
    EP --> EV["MitoEvidence-Eval<br/>L1–L4 / EV0–EV4"]
    EV --> RP["SEER · D1–D10 · Gate · Cost"]
    RP --> UI["Eval-Dominator 任务详情与报告"]
~~~

[打开 Mito-Agent 在线实例](https://agent.blueskun.com:8444/)（授权凭据由项目方通过私密渠道提供）。

接入不通过浏览器抓取界面，而是实现稳定的机器接口或后端包装层。目标契约至少包含：

1. 输入：`question`、实验条件约束、`dataset_item_id`、seed、预算与 trace 等级；
2. 输出：最终答案、原子主张、引用、EvidenceSpan、CEC 条件、ANSWER/ABSTAIN/HUMAN_REVIEW 决策；
3. 轨迹：检索查询、召回与重排、工具调用、补检、失败、Token、延迟和产物引用；
4. 身份：Agent/模型/Prompt/语料/索引/工具版本和不可变哈希；
5. 安全：凭据通过环境变量或秘密管理系统注入，日志与 Episode 只保存脱敏引用。

### 8.4 模块映射

| 模块 | 项目实现 |
|---|---|
| ModelAdapter | 在现有平台登记 Hy3 OpenAI-compatible API 与服务身份 |
| AppAdapter | `MitoAgentAdapter` 调用现有 Agent 后端；S0–S5 各版本显式标识 |
| DatasetAdapter | MedicalQuestionSet / EvaluatorChallengeSet |
| Inferencer | 运行 Agent 并冻结 Episode |
| Evaluator | L1–L4 与 EV0–EV4 |
| Summarizer | SEER、D1–D10、门禁、统计与成本 |
| TraceStore | 配置、快照、尝试、日志和制品 |

### 8.5 正式分数身份

任一正式分数必须绑定：

~~~text
benchmark_id
+ dataset_revision / hash / split
+ prompt_version
+ evaluator_version
+ aggregation_policy
+ model_config / served_revision
+ corpus / index / graph snapshot
+ runner / tool identity
+ attempt_id / seed
~~~

### 8.6 工程不变量

1. full 与 smoke 运行使用不同缓存身份，smoke 不进入正式报告；
2. 结果按 attempt 追加，不覆盖历史成功结果；
3. 结果与终态原子提交，基础设施失败不计作模型错误；
4. 缓存键包含数据内容哈希和所有影响结果的选项；
5. 取消、超时、重启恢复具有明确状态，不把“仍在运行”写成“已取消”；
6. API Key 不进入日志、配置回显、结果文件或前端；
7. 生成代码若需执行，必须进入隔离 Runner，并记录镜像和工具链身份；
8. 离线发布包包含源码、配置、数据清单、镜像/依赖锁、SHA-256 与健康检查。

### 8.7 通用 Benchmark 的位置

MMLU-Pro、C-Eval、GSM8K、GPQA、EvalPlus、LongBench 等只用于：

- 底座能力画像；
- 升级或训练后的回归检查；
- 附录说明。

它们不进入医学场景总分，也不能代替 MitoEvidence-Eval 的有效性验证。

> [!CAUTION]
> 当前 RC10 可作为功能与集成底座，但不能仅凭“页面可运行”视为公开生产就绪。正式上线前必须收口默认凭据与密钥、CORS/访问控制、Code Runner 隔离、任务恢复与 attempt 追加、镜像/源码身份及离线恢复演练等门禁；这些工程修复不改变“先验证评估器”的研究顺序。

---

## 9. 预期效果与验收线

### 9.1 评估方法

| 目标 | 建议预注册线 | 判定 |
|---|---:|---|
| 高/中/低完整排序正确率 | ≥ 85% | 判别力 |
| Pairwise Accuracy | ≥ 0.90 | 判别力 |
| 自动—人工 Spearman ρ | ≥ 0.75 | 外部一致性 |
| 人工序数加权 κ | ≥ 0.60 | 标注可靠性 |
| 总分 ICC(2,1) | ≥ 0.75 | 连续评分一致性 |
| Judge 重复评估中位标准差 | ≤ 3 / 100 | 稳定性 |
| 门禁状态翻转率 | ≤ 5% | 稳定性 |
| 破坏性攻击检出率 | ≥ 85% | 鲁棒性 |
| 严重引用错误假阴性率 | ≤ 10% | 安全性 |
| 等价变形中位分数变化 | ≤ 2 分 | 不变性 |
| 等价变形门禁翻转 | 0 次 | 不变性 |
| 篇幅/术语投机 | 不产生显著增益 | 抗投机 |

### 9.2 应用系统

| 目标 | 主比较 | 验收方式 |
|---|---|---|
| 降低 SEER | S4 vs S2 | 配对差值及问题聚类 95% CI |
| 提高任务正确性与核心覆盖 | S4 vs S2 | D1 与必答子问题覆盖 |
| 提高主张—证据支持 | S4 vs S2 | D3 |
| 降低条件与方向错误 | S3/S4 vs S2 | D5 与 CONDITION_MISMATCH |
| 提高冲突证据覆盖 | 冲突题上 S4 vs S2 | D6 |
| 提高正确拒答 | S4 vs 无拒答版本 | Precision / Recall / F1 与 risk–coverage |
| 控制成本 | 所有版本 | 质量—Token—延迟—调用次数 Pareto |

### 9.3 工程与用户体验

| 目标 | 建议验收 |
|---|---|
| 六类底座回归 | 固定协议下形成完整能力向量；升级前后差异可定位到具体基准与样本，不强求单一总分 |
| 关键引用回原文 | 金标 100%；自动样本 ≥ 95%，其余转人工 |
| 正式结果可复现 | 每行结果可链接 Episode、配置和哈希 |
| 错误可归因 | 检索、重排、解析、条件、生成、核验、拒答、预算错误可区分 |
| 科研核对效率 | 以用户实验实测“定位一条支持证据的时间”，不预设虚假提升值 |
| 负结果透明 | 未达标、空输出、超时、置信区间跨 0 均保留 |

> [!CAUTION]
> “比基线好”是待检验假设，不是交付承诺。真正的成功包括发现某个复杂模块无增益，并能用可复现数据说明原因。

---

## 10. 时间规划

### 10.1 资源假设

**W0 已有基础：** OpenCompass + Eval-Dominator 离线平台已构建，Mito-Agent 已在线运行，`hy-agent` 作为统一公开仓库。主计划不重复开发通用后台，优先完成 Adapter、Trace、领域数据集和评估器。

12 周计划按以下最低配置估算：

- 1 名技术负责人 / 后端；
- 1 名检索与证据工程师；
- 1 名评测与数据工程师；
- 0.5 名界面 / DevOps 支持；
- 2 名兼职领域研究者负责标注和裁决。

若只有 1–2 名开发者，建议将主计划扩展至 16–20 周。

### 10.2 十二周主计划

| 周期 | 阶段 | 核心工作 | 交付物 | 进入门槛 |
|---|---|---|---|---|
| W1 | P0 接入冻结 | 冻结用户/题型/主终点；确认 Mito-Agent 调用方式、Trace 与凭据边界 | Scope v1、接口契约、风险清单 | Agent 与 Hy3 服务可联通 |
| W2 | P1 平台接入 | 实现 MitoAgentAdapter、Episode 转换、单题 smoke、失败状态 | 首个端到端 Episode | 不通过 UI 抓取，凭据不落盘 |
| W2–3 | P2 六类基线 | 选择代表性公共协议跑 smoke；对可报告项目运行 full 基线 | 六类能力画像 v1 | 协议、样本、模型身份可追溯 |
| W3–4 | P3 语料与证据锚 | 150–300 篇 Pilot 摄取、元数据、EvidenceSpan、标注工具 | Corpus v1、定位审计、Schema | 关键 span 可回原文 |
| W4–5 | P4 系统基线 | 完成 S0–S3、CEC 初版、快照、预算与失败归因 | 医学基线与首批 Case | 各版本可独立复现 |
| W5–6 | P5 评估器 | D1–D10、L1/L2、Hard Gate、标注指南与校准题 | EV0–EV3、20 题校准集 | 规则与人工协议可执行 |
| W6–7 | P6 挑战集 | 变形算子、manifest、不同来源原始输出 | ChallengeSet v1 | 每个变形可离线回放 |
| W7–8 | **P7 评估器有效性** | 判别力、一致性、稳定性、对抗、不变性、构念效度 | 元评估报告、冻结 Evaluator | 达门槛或按预案降级 |
| W9 | P8 完整核验系统 | S4、一次补检、拒答、人工复核队列 | T1 完整系统 | SEER 不劣于 S2 |
| W10 | P9 密封测试 | 冻结测试、配对比较、统计与失败归因 | 正式结果表与 Case | 全部结果有 Episode |
| W11 | P10 工程验收 | 安全、缓存、重启、复现、少量研究者试用 | QA 与效率记录 | 关键发布门禁关闭 |
| W12 | P11 封装发布 | README、数据卡、Demo、离线复现与交付检查 | 完整提交包 | 空环境可复现核心结果 |

### 10.3 甘特图

~~~mermaid
gantt
    title MitoEvidence 12 周实施计划
    dateFormat  YYYY-MM-DD
    axisFormat  %m/%d

    section 范围与数据
    接口冻结与 Agent 接入     :a1, 2026-08-31, 14d
    语料摄取与 EvidenceSpan  :a2, 2026-09-14, 14d

    section 系统
    六类能力基线             :b0, 2026-09-07, 14d
    S0-S3 基线与 Episode     :b1, 2026-09-21, 14d
    CEC 与逐主张评估         :b2, 2026-09-28, 14d
    S4 核验与拒答            :b3, 2026-10-26, 7d

    section 评估器
    Challenge Set            :c1, 2026-10-05, 14d
    元评估与冻结             :crit, c2, 2026-10-19, 14d

    section 发布
    密封测试与失败归因       :d1, 2026-11-02, 7d
    安全、复现与用户试用     :d2, after d1, 7d
    文档、Demo 与发布        :d3, after d2, 7d
~~~

### 10.4 六周最小路径

若时间只有 6 周，明确砍掉 T2/T3、真正图像通道、跨模型验证和全量语料：

~~~text
W1  MitoAgentAdapter + Episode/Trace 契约 + 单题端到端 smoke
W2  代表性六类基线 + 150 篇开放全文 + EvidenceSpan
W3  S0–S3 + CEC + 40 题 Pilot + 双人抽检
W4  Challenge Set + EV0–EV3
W5  判别力 / 一致性 / 稳定性 / 对抗 + S4
W6  密封小测 + 结果表 + 典型 Case + README + Demo
~~~

六周版本不保留图表题，或将其单列为未支持子集，不能一边删除图表通道、一边把图表题计入主比较。

---

## 11. 风险、降级与边界

| 风险 | 影响 | 预防 / 降级 |
|---|---|---|
| 文献选择偏差 | 正向结果与高影响期刊过表达 | 高召回检索、分层质量卡、禁止顶刊白名单充当真值 |
| CEC 抽取不准 | 错误证据污染检索和评分 | 缩小语料、提高人工抽检、关键槽位人工补齐 |
| 循环评测 | 生成与 Judge 自洽但错误 | L1/L4 锚点、盲评、逻辑隔离、专家一致性 |
| 测试泄漏 | 结果虚高 | 问题/模板/标签/轨迹隔离，密封测试审计 |
| 金标不完整 | Recall 与覆盖失真 | pooled retrieval、替代证据组、封闭世界声明 |
| 评估器被投机 | 篇幅/术语刷分 | 投机变形、Hard Gate、单维度干预 |
| 评估器过严 | 同义改写被扣分 | 等价不变性作为独立门槛 |
| 多模态名不副实 | 宣称超过实际能力 | MVP 只做图题/图注/表格/OCR，视觉语义延期 |
| Hy3 API 不稳定 | 无法完成正式运行 | 固定服务版本、重试/超时、失败保留、预留离线回放 |
| 专家人力不足 | 金标规模不足 | 分层抽样、争议样本优先、保留双人盲评关键子集 |
| 版权与隐私 | 无法开源全文或泄露材料 | 只发布元数据、哈希与构建脚本；禁止敏感材料 |
| 范围蔓延 | 核心评估器无法按期完成 | T1/T2/T3 门槛；M3 未达标时停止图谱与 RL |

### 降级触发

| 触发条件 | 动作 |
|---|---|
| W3 证据锚不达标 | 语料缩到 100–150 篇，优先保证物理定位 |
| W6 CEC 关键槽位不达标 | 关键字段人工补齐，自动抽取降为候选 |
| W8 κ < 0.60 | 停止 T2/T3，重写锚点与标注指南后复评 |
| 攻击检出率 < 85% | 定位 L1/L2 漏检，不通过降低难度掩盖 |
| Hy3 端点无法冻结 | 正式实验延期；只报告框架联调，不混入主结果 |
| 专家不足 | 保留双人关键子集，其余单人 + 争议抽检 |

---

## 12. 交付物与仓库结构

### 12.1 交付物

| 类别 | 内容 |
|---|---|
| 应用 | S0–S4 源码、Hy3 适配器、证据审阅界面、拒答与审核流程 |
| 数据 | Corpus manifest、MedicalQuestionSet、EvaluatorChallengeSet、变形 manifest |
| 评估 | D1–D10、SEER、Hard Gate、L1 规则、Judge Prompt、人工标注指南 |
| 结果 | 逐题输出、分项分数、95% CI、成本、失败标签、典型 Case |
| 复现 | 配置、依赖锁、快照哈希、运行命令、失败日志、发布清单 |
| 展示 | README、中文详细方案、≤ 2 分钟 Demo/GIF |

### 12.2 建议目录

~~~text
hy-agent/
├── README.md
├── assets/
│   └── opencompass-evaluation-platform.png
├── app/
│   ├── direct/
│   ├── vector_rag/
│   ├── hybrid_rag/
│   ├── verified_agent/
│   └── adapters/
│       └── mito_agent/
├── corpus/
│   ├── ingestion/
│   ├── evidence_span/
│   ├── indexes/
│   └── manifests/
├── evidence/
│   ├── schema/
│   ├── extraction/
│   ├── normalization/
│   └── retrieval/
├── evaluation/
│   ├── datasets/
│   ├── rubric/
│   ├── rules/
│   ├── judge/
│   ├── mutations/
│   ├── human/
│   └── summarizers/
├── results/
│   ├── episodes/
│   ├── scores/
│   ├── validation/
│   └── cases/
├── docs/
│   ├── ANNOTATION_GUIDE_ZH.md
│   └── RELEASE_CHECKLIST.md
├── configs/
└── tests/
~~~

### 12.3 最小复现入口

~~~bash
python -m evaluation.verify_env --manifest configs/pilot_v1.json
python -m evaluation.run --app verified_agent --dataset medical_question_set/sealed
python -m evaluation.validate --challenge-set evaluator_challenge_set --evaluators ev0,ev1,ev2,ev3,ev4
~~~

命令是目标接口约定，只有对应模块和测试落地后才写入 README 的“可运行命令”区。

---

## 13. 两分钟 Demo 脚本

~~~text
0:00–0:15  真实科研问题：展示通用回答中的实验条件混淆
0:15–0:30  打开已部署 Mito-Agent，展示条件解析与证据矩阵
0:30–0:45  在现有评测平台提交同一问题并展示完整阶段进度
0:45–1:10  点击一条主张，跳到原文页码/图号/证据片段
           展示支持、反驳、证据不足三分区
1:10–1:30  展示逐主张核验、一次补检和正确拒答
1:30–1:45  注入伪造 DOI 或物种偷换，展示 L1/Hard Gate 捕获
1:45–1:55  展示六类底座画像与医学场景指标分开报告
1:55–2:00  hy-agent 仓库、结果链接和复现入口
~~~

Demo 的视觉中心应是“点击回原文”和“评估器识别流畅错误”，而不是堆叠模型或算法名称。

---

## 14. 启动清单

### W1 必须确认

- [ ] Mito-Agent 的机器调用入口、鉴权方式、请求/响应 Schema 与 Trace 导出；
- [ ] 登录账号、密码和 API Key 已迁入私密凭据管理，公开仓库与文档中无明文；
- [ ] OpenCompass 平台可通过 Adapter 运行一条 Mito-Agent Episode，并保留失败日志；
- [ ] Hy3 可用端点、模型 ID、服务版本、预算和比赛使用规则；
- [ ] 两名领域研究者及可投入标注时数；
- [ ] MinerU/OvisOCR2 的版本、健康检查与样例证据；
- [ ] 语料来源、全文许可、检索式与公开策略；
- [ ] 现有 3000 篇候选文献的真实清单、去重和撤稿状态；
- [ ] Pilot 的实际截止日期与团队规模；
- [ ] 公开仓库中不包含 API Key、受限全文或未公开实验材料。

### 主结论发布前

- [ ] 评估器先于系统比较完成有效性验证；
- [ ] MedicalQuestionSet 与 Challenge Set 完全分离；
- [ ] 密封测试未进入 Prompt、调参、图融合或奖励；
- [ ] 所有正式结果绑定 Episode 和 Artifact Manifest；
- [ ] 负结果、超时、空输出、拒答均保留；
- [ ] 引用列表通过自动校验和人工抽检；
- [ ] 方案中“已完成”均有仓库文件或运行证据链接；
- [ ] 研究用途、临床边界和图像能力边界在 README 首屏可见。

---

## 附录 A：策略取舍记录

| 内容 | 结论 | 进入正文的位置 |
|---|---|---|
| 评估优先、先验证尺子 | 直接采纳 | 1、3、7 |
| CEC 与物理证据锚 | 直接采纳 | 3、4、5 |
| 规则 + Judge + Hard Gate + 专家 | 直接采纳 | 5、6 |
| 两个样本集分离 | 直接采纳 | 6 |
| 对抗、等价和局部干预 | 直接采纳 | 5、7 |
| OpenCompass 风格执行 | 修改后采纳 | 8 |
| 九维 Rubric | 修改为十维，补任务正确性与核心覆盖 | 6 |
| A–F 多组件消融 | 修改为 S0–S6 单因素递增 | 7 |
| 医学证据超图 | 降级为条件化证据模型，过门槛再图化 | 5 |
| 图文混合检索 | 首期只做图题、图注、表格和 OCR | 2、5 |
| KGE、RotatE、GRPO | 延后到 T3 | 5 |
| 顶刊白名单语料 | 舍弃 | 5、11 |
| 通用 Benchmark 合成医学总分 | 舍弃 | 8 |
| “一定优于基线” | 舍弃，改成可证伪假设 | 9 |

---

## 附录 B：术语表

| 规范术语 | 定义 |
|---|---|
| MitoEvidence | 可追溯快速证据综述 Agent |
| MitoEvidence-Eval | 证据中心评估方法，也是主要研究产出 |
| CEC | Claim–Evidence–Condition，主张、原文证据与成立条件 |
| EvidenceSpan | 可定位到章节、页码、图表号和 bbox 的原文片段 |
| EvidenceReviewEpisode | 可冻结、可回放的一次完整运行与评估单元 |
| AcceptableEnvelope | 必答内容、允许结论、替代证据、必保留条件和禁止外推 |
| SEER | Severe Evidence Error Rate，严重证据错误率 |
| Hard Gate | 不能由其他分数抵消的严重错误否决规则 |
| ABSTAIN | 证据不足时拒绝作确定性回答 |
| HUMAN_REVIEW | 低置信、冲突或门禁问题转人工 |
| MedicalQuestionSet | 用于比较系统版本的真实科研问题集 |
| EvaluatorChallengeSet | 用于验证评估器的变形挑战集 |
| mutation manifest | 记录变形对象、前后值和预期影响的清单 |
| T1 / T2 / T3 | 必做 / 门槛增强 / 研究展望 |

---

## 附录 C：参考依据

以下资料用于支撑设计方向，不代表其外部结果可以直接迁移到本项目：

1. [Tencent Hy3 官方仓库](https://github.com/Tencent-Hunyuan/Hy3)
2. [OpenCompass 官方文档](https://doc.opencompass.org.cn/)
3. [PRISMA 2020](https://www.bmj.com/content/372/bmj.n71)
4. [OpenScholar：科学文献检索与带引文综合](https://www.nature.com/articles/s41586-025-10072-4)
5. [TrialMind：临床证据综合流程与人机协同](https://www.nature.com/articles/s41746-025-01840-7)
6. [ALCE：带引文文本生成评估](https://aclanthology.org/2023.emnlp-main.398/)
7. [RAGChecker：检索与生成的细粒度失败归因](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html)
8. [Biolink Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC9372416/)
9. [KARL：知识密集 Agent 的强化学习](https://aclanthology.org/2026.acl-long.2196/)
10. [ToolOmni：工具检索与执行的多目标 GRPO](https://aclanthology.org/2026.acl-long.1736/)
11. [AgentGL：基于强化学习的 Agentic Graph Learning](https://aclanthology.org/2026.acl-long.1161/)
12. [ACL 2026 Program Chairs 报告](https://aclanthology.org/2026.acl-long.0.pdf)

---

<div align="center">

### 让每个医学主张，都能回到论文原文和实验条件。

**Evidence first · Evaluation before optimization · Negative results included**

</div>

