<div align="center">

# Hy-Agent · MitoEvidence

### 面向 β 细胞线粒体研究的可追溯证据综述与九维评测

![Model](https://img.shields.io/badge/MODEL-Hy3-111111?style=flat-square)
![Version](https://img.shields.io/badge/MVP-v0.3.1-555555?style=flat-square)
![Tests](https://img.shields.io/badge/TESTS-582%20PASSING-111111?style=flat-square)
![Gold](https://img.shields.io/badge/EXPERT%20GOLD-127-555555?style=flat-square)
[![Mito Agent](https://img.shields.io/badge/MITO--AGENT-DEPLOYED-111111?style=flat-square)](https://agent.blueskun.com:8444/)

</div>

<p align="center">
  <a href="#1-项目简介">项目简介</a> ·
  <a href="#2-当前状态">当前状态</a> ·
  <a href="#3-总体架构">总体架构</a> ·
  <a href="#4-评测与实验">评测与实验</a> ·
  <a href="#5-快速复现">快速复现</a> ·
  <a href="#6-文档入口">文档入口</a>
</p>

---

## 1. 项目简介

MitoEvidence 包含两部分：面向医学实验与文献综述的 **Mito-Agent**，以及独立的
**MitoEvidence-Eval**。项目聚焦胰岛、β 细胞、线粒体与显微成像研究，把科研问题转化为
带原文锚点、实验条件、引用身份和审核状态的可追溯证据综述。

> **回答得像专家，不等于证据可信。每个关键结论都应能回到原文；物种、细胞、实验条件与
> 效应方向必须可核验；严重证据错误不能被语言流畅度抵消。**

工程闭环包括：

`问题结构化 → 冻结语料检索 → 证据约束生成 → 原子主张/证据绑定 → XML 锚点重定位 → 九维评估 → 审计包`

[Mito-Agent 在线实例](https://agent.blueskun.com:8444/)已经部署。仓库提供可复现的 Hy3
调用、评测、实验和审计代码；在线实例与本仓库实验结果分别报告，不把部署状态当作有效性结论。

## 2. 当前状态

截至 2026-08-31，当前可核查基线如下：

| 项目 | 已完成状态 | 证据入口 |
|---|---|---|
| 应用闭环 | Hy3 MVP `v0.3.1`；真实五题 Pilot 完成 `5/5` | `app/`、`scripts/run_pilot_suite.py` |
| 专家参考 | 127 条项目负责人确认的**单一专家共识金标**；原始快照、字段 designation 与逐文件哈希可审计 | `annotation_prelabel/expert_gold_manifest.json`、`evaluator/expert_gold.py` |
| 评测框架 | D1–D9、NA 重归一、四类致命错误上限、PASS/REVIEW/REJECT | `evaluator/rubric.py`、`evaluator/assembly.py` |
| Hy3 Judge | Function Calling、JSON Schema 备选、本地校验、自一致性与升级队列 | `evaluator/judge/` |
| 真实有效性 Pilot | 术语正误对 180/180 次、Claim 准入 50/50 次真实 Hy3 调用完成 | `app/terminology_pair_pilot.py`、`app/claim_admission_pilot.py` |
| A/B/C/D 消融 | 正式 v4 共 60/60 cells 完成并通过独立产物审计，`production_ready=true`；D 组 answerability 11/15（0.7333），κ=0.5833 | `app/ablation.py`、`evaluator/ablation_artifacts.py`、`docs/experiment_results_20260831.md` |
| 实验协议 | 系统—专家参考一致度、判别力、稳定性、对抗性与消融审计入口已实现 | `evaluator/experiment_protocol.py` |
| 工程验证 | Python 3.11 下 582 项离线测试通过 | `tests/` |

127 条金标来自四类任务，不能当作 127 个同构评分样本混算：

| 金标组成 | 数量 | 可用范围与边界 |
|---|---:|---|
| Pilot 问题 | 5 | 含 30 条 required claims；`evidence_papers/evidence_spans` 为空 |
| Claim 审核 | 50 | accept 8、accept_with_edits 25、reject 14、uncertain 3 |
| 术语正误对 | 60 | wrong/correct 成对完整；没有另设 approve/reject 标签 |
| 综述池评估 | 12 | 共 2,043 条参考文献；7 篇有本地 XML 与冻结哈希 |

原始 JSONL 中的 `ai_*`、`annotator`、`review_status` 等历史字段保持原样；仓库通过独立
manifest 记录项目负责人确认的 `expert_consensus_gold` designation，不静默改写来源快照。

### 五题真实 Hy3 Pilot

固定应用版本 `v0.3.1` 在同一套件中完成 5/5：

| Pilot | 召回段落 | 输出主张 | 状态 |
|---|---:|---:|---|
| PILOT-01 | 12 | 7 | 完成 |
| PILOT-02 | 12 | 4 | 完成 |
| PILOT-03 | 12 | 8 | 完成 |
| PILOT-04 | 12 | 3 | 完成 |
| PILOT-05 | 0 | 0 | 正确越界拒答 |

系统与专家金标 `answerability` 的原始一致率为 **0.60**，Cohen's **κ=0.375**，`n=5`。
这是 **Hy3 系统对单份专家参考的一致度**，只用于描述小样本 Pilot；它**不是专家间一致性**。
当前没有专家 A/B 的独立逐项标签，因此专家间 κ、Gwet's AC1/AC2 和 ICC 均不可计算，必须报告
`unavailable`，不能复制同一份金标来构造第二位专家。

### 真实术语与 Claim Pilot

| 任务 | 规模 | 结果 | 使用边界 |
|---|---:|---|---|
| 术语/条件错误成对判别 | 60 对 × 3 次 | 173/180 正确，7 次 abstain，0 次选择攻击表述；多数票 58/60；重复一致率 96.67% | 正确句通常更长，长度基线已达 90%；不是全文证据核验 |
| Claim 准入四分类 | 50 × 1 次 | accuracy=0.32，κ=0.0357，macro-F1=0.2172；多数类基线=0.50 | reject 召回仅 7.14%，不能作为自动排除门禁 |

两组都是实际 Hy3 结果，也都保留不理想指标。它们产生于旧 v1 artifact contract，分析器明确标为
`legacy_v1_nonformal_limited_cell_provenance`，不会静默升级为新版完整逐调用证明。详细指标、
偏差分析和 SHA-256 见 [`docs/experiment_results_20260831.md`](docs/experiment_results_20260831.md)。

### 正式 A/B/C/D v4 消融

正式套件 `pilot-abcd-hy3-v4-formal-r1-20260831` 使用 5 道专家金标问题、每题 3 次重复和
4 个实验组，共 **60/60 cells 成功**；输入与证据快照、生成器/Judge 身份、seed、缓存命名空间、
C→D 精确绑定及敏感信息复扫均通过，独立审计结论为 `production_ready=true`。

| 组别 | answerability 一致 | Cohen's κ | 重复两两一致率 |
|---|---:|---:|---:|
| A 无检索 | 9/15（0.6000） | 0.4118 | 0.8000 |
| B 稀疏检索 | 10/15（0.6667） | 0.5000 | 0.7333 |
| C 证据图重排 | 7/15（0.4667） | 0.2857 | 0.7333 |
| D 同一 C 草稿 + Judge 门控 | **11/15（0.7333）** | **0.5833** | **0.8667** |

D 的点估计和重复稳定性最高，且三组配对比较均无“对照正确、D 错误”的记录；但样本仅
15 对，D 对 A、B、C 的双侧精确 McNemar 检验分别为 `p=0.50`、`p=1.00`、`p=0.125`，
均未达到统计显著，不能宣称 D 显著优于其他组。此前 v3 r3 运行因“只允许一次生成”与已冻结的
有界 Schema 修复机制冲突，在 8/60 cells 后终止；它只保留为**实验协议诊断**，不并入性能比较。

## 3. 总体架构

<p align="center">
  <a href="assets/mitoevidence-architecture.html">
    <img src="assets/mitoevidence-architecture.jpg" alt="MitoEvidence 从科研问题到可信评测的五步闭环架构" width="100%" />
  </a>
</p>

<p align="center"><sub>五步主流程与可信证据、离线评测两层支撑能力</sub></p>

| 层级 | 核心能力 | 审计输出 |
|---|---|---|
| 应用层 | Hy3 规划、冻结语料 BM25、证据约束综合、越界拒答 | 运行清单、响应、Token 与延迟 |
| 证据层 | Claim、EvidenceSpan、实验条件、XML 文本锚点 | 来源标识、段落位置与内容哈希 |
| 规则层 | 引用身份、结构、数字/单位、术语三态检查 | 可复算的确定性结果 |
| Judge 层 | 逐主张支持关系、条件与方向判断、自一致性 | 结构化判定、置信度与升级队列 |
| 门禁层 | 九维计分、NA、致命错误上限、人工复核状态 | 维度分、失败原因与发布决策 |

<p align="center">
  <img src="assets/evaluation-platform.png" alt="离线评测平台任务详情页" width="100%" />
</p>

<p align="center"><sub>离线评测平台的任务详情与题级分析界面</sub></p>

## 4. 评测与实验

### 九维评测

| 维度 | 内容 | 权重 |
|---|---|---:|
| D1 | 引用真实性 | 10 |
| D2 | 主张—证据一致性 | 20 |
| D3 | 关键证据覆盖与引用完整性 | 15 |
| D4 | 实验条件与效应方向准确性 | 12 |
| D5 | 机制综合、冲突处理与因果校准 | 15 |
| D6 | 不确定性、拒答校准与安全边界 | 10 |
| D7 | 可追溯性与流程可复现性 | 8 |
| D8 | 专业术语、数字与单位准确性 | 5 |
| D9 | 用户可理解性与格式规范 | 5 |

核心结论使用伪造引用、关键主张大面积无可定位证据、物种或效应方向偷换，以及输出患者个体化
诊疗决策，会触发独立于加权总分的致命错误上限。工具超时、Schema 失败和缺失输出不会从分母删除。

### A/B/C/D Pilot 消融

| 组别 | 配置 | 目的 |
|---|---|---|
| A | 无检索 | 测量底座直接生成能力 |
| B | 冻结语料稀疏 TF-IDF | 测量基础检索增益；不伪称 dense embedding |
| C | 冻结证据图重排 | 测量结构化证据选择增益 |
| D | 同一 C 草稿 + Hy3 Judge 门控 | 测量逐主张审核、降级与拒答增益 |

runner 会记录每个 cell 的成功或失败，并要求 D 绑定精确的 C artifact hash。旧 v2 真实套件的
20/20 cell 均结构完整且无审计错误，但因缺少完整逐调用身份，只能标为 nonformal。
正式 v4 已在官方 Hy3 端点完成 60/60 cells，并在非空 seed、输入/证据快照、成功/失败敏感信息
复扫和全部跨组绑定同时通过后得到 `production_ready=true`。它把已冻结的有界结构修复纳入应用
方法，分别报告一次通过与修复后完成情况；v3 r3 仅作为协议冲突诊断，不作为消融结果。

判别力、对抗性和稳定性同样按已落地协议运行。当前五题 Pilot 没有原文 EvidenceSpan，也没有
输出级专家九维总分，因此不能补造完整 D2/D3 金标指标，亦不能报告自动总分对专家总分的
Spearman、MAE 或 ICC。

## 5. 快速复现

```bash
# 1. 建立隔离环境并安装锁定依赖
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt

# 2. 运行全部离线测试（不调用网络或 Hy3）
.venv/bin/python -m pytest -q

# 3. 审计 127 条专家共识金标及其哈希
.venv/bin/python scripts/audit_expert_gold.py \
  --out results/expert_gold/audit.json

# 4. 获取 manifest 冻结的 OA XML；全文不会进入 Git
.venv/bin/python scripts/fetch_frozen_corpus.py \
  --output results/frozen_corpus_fetch.json

# 5. 运行真实 Hy3 五题套件；凭据只从环境变量读取
.venv/bin/python scripts/run_pilot_suite.py \
  --suite-id pilot5-hy3-v1

# 6. 运行真实术语与 Claim Pilot
.venv/bin/python scripts/run_terminology_pair_pilot.py \
  --suite-id terminology-pair-hy3-v2 --limit 60 --repeats 3 --base-seed 20260831
.venv/bin/python scripts/run_claim_admission_pilot.py \
  --suite-id claim-admission-hy3-v2 --limit 50 --repeats 1 --base-seed 20260831

# 7. 运行可审计的 A/B/C/D v4；缓存命名空间必须按 suite 隔离
SUITE_ID=pilot-abcd-hy3-v4-$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python scripts/run_pilot_ablation.py \
  --suite-id "$SUITE_ID" --replicates 3 --top-k 12 \
  --judge-k 7 --judge-temperature 0.7 \
  --judge-base-seed 20260831 --generator-base-seed 20260831 \
  --generator-cache-namespace "mitoevidence-$SUITE_ID"
```

详细安装、配置、数据契约、九维计分和命令说明见
[`docs/implementation_guide.md`](docs/implementation_guide.md)。运行产物默认写入被 Git 忽略的
`results/`；受版权限制的全文同样不会提交。

## 6. 文档入口

| 文档 | 内容 |
|---|---|
| [`docs/implementation_guide.md`](docs/implementation_guide.md) | 完整工程实现、配置、目录结构与复现命令 |
| [`docs/completion_status_20260830.md`](docs/completion_status_20260830.md) | 已完成证据、真实 Pilot 结果与不可宣称边界 |
| [`docs/experiment_results_20260831.md`](docs/experiment_results_20260831.md) | 真实 Hy3 指标、偏差、失败模式、产物哈希与证明范围 |
| [`docs/proposal.md`](docs/proposal.md) | 方案设计、量表依据与预注册口径 |
| [`docs/verification_report.md`](docs/verification_report.md) | 文献、数据源和方法学核验记录 |
| [`eval/rubric.md`](eval/rubric.md) | 九维量表细则与待冻结项 |

当前最准确的项目表述是：

> **MitoEvidence-Hy3 已完成应用和评估工程闭环，以 127 条项目负责人确认的单一专家共识金标
> 作为唯一参考，并跑通真实五题、术语正误对、Claim 准入 Pilot 及 60/60 cells 的正式 v4
> A/B/C/D 消融。D 组 answerability 点估计最高，但配对检验尚不显著；完整输出级 D1–D9
> 好/中/差判别与 12 类完整对抗实验仍未完成。这些小样本结果不等价于大规模医学有效性验证，
> 缺失的专家字段和专家间一致性不会被推断或补造。**

---

<div align="center">
  <strong>Ground every claim. Trace every source. Evaluate every failure.</strong>
</div>
