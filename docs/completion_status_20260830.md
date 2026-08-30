# MitoEvidence-Hy3 完成状态与实验边界

> 状态更新：2026-08-31
> 当前结论：应用、评估与实验审计代码已形成可运行基线；项目负责人确认的 127 条历史
> “预标”记录即项目负责人确认的单一专家共识金标，不再等待另一轮人工标注。

## 1. 已完成

| 工作线 | 当前状态 | 可核查位置 |
|---|---|---|
| 应用闭环 | Hy3 规划、冻结全文检索、证据约束生成、Claim—EvidenceSpan、XML 锚点与哈希审计；真实五题 5/5 完成 | `app/`、`scripts/run_pilot_suite.py` |
| 专家参考 | 5 道 Pilot、50 条 Claim、60 组术语正误对、12 篇综述池；共 127 条项目负责人确认的单一专家共识金标，逐文件 SHA-256 固定 | `annotation_prelabel/expert_gold_manifest.json` |
| 九维评估 | D1–D9、NA、事件上限、四类致命错误与 PASS/REVIEW/REJECT | `evaluator/`、`configs/rubric_v0_1.yaml` |
| 真实术语实验 | 60 对 × 3 次，180/180 成功；多数票 58/60，重复两两一致率 96.67% | `app/terminology_pair_pilot.py`、结果报告 |
| 真实 Claim 实验 | 50/50 成功；四分类准确率 0.32、κ=0.0357，暴露自动准入能力不足 | `app/claim_admission_pilot.py`、结果报告 |
| A/B/C/D | 正式 v4 共 60/60 cells 成功并通过独立审计，`production_ready=true`；D 组 answerability 11/15（0.7333），κ=0.5833 | `app/ablation.py`、`evaluator/ablation_artifacts.py`、`experiment_results_20260831.md` |
| 安全与可追溯 | 成功/失败产物独立敏感信息扫描，顶层与 cell 快照审计，路径/软链接逃逸防护，旧格式显式降级 | `evaluator/artifact_security.py`、`evaluator/pilot_identity.py` |
| 工程回归 | Python 3.11 下 582 项离线测试通过 | `tests/` |

完整指标、哈希和解释边界见
[`experiment_results_20260831.md`](experiment_results_20260831.md)。

## 2. 必须保持的金标口径

- 原始 JSONL 的 `ai_*`、`annotator`、`review_status` 是历史字段名；不改写原始快照，
  由独立 manifest 将其指定为 `expert_consensus_gold`。
- 每个 item 只有一份合并结果，没有专家 A/B 的独立标签；专家间 κ、Gwet 或 ICC 均为
  `unavailable`。只能报告自动系统相对单份专家参考的一致度。
- 5 道 Pilot 的 `evidence_papers/evidence_spans` 为空，不能据此补造完整 D2/D3 证据金标。
- 被测模型只能读取中性问题、Claim 输入或正误候选，不能读取专家决定、required claims、
  理由、缺陷码或正确侧标签。

## 3. 当前真实结果

- 五题应用 answerability：原始一致率 0.60，Cohen's κ=0.375（n=5）；
- 术语成对判别：173/180 正确、7 次 abstain、0 次选择攻击表述；长度基线已经达到 90%，
  因而 96.67% 的条目多数票准确率不能脱离偏差分析单独宣传；
- Claim 四分类：准确率 0.32，低于 0.50 的多数类基线；reject 召回率只有 7.14%；
- A/B/C/D v2：20/20 cell 文件完整、0 error，但属于 legacy structural audit，
  `production_ready=false`，不作为正式消融结论；
- A/B/C/D v4 正式套件：60/60 cells 成功，独立产物审计 `production_ready=true`。A、B、C、D
  的 answerability 一致率依次为 9/15、10/15、7/15、11/15，κ 依次为 0.4118、0.5000、
  0.2857、0.5833；D 的点估计最高；
- D 对 A、B、C 的双侧精确配对 McNemar 检验分别为 `p=0.50`、`p=1.00`、`p=0.125`，
  均不显著，因此不能宣称 D 具有统计显著优势。

术语和 Claim 是真实 Hy3 运行，但使用旧 v1 artifact contract。分析器保留兼容并明确标为
`legacy_v1_nonformal_limited_cell_provenance`；不会静默把旧结果升级为完整可复现证明。

v3 r3 在 8/60 cells 后因“只允许一次生成”与有界 Schema 修复机制冲突而停止；该套件只用于
定位和修复实验协议矛盾，属于**协议诊断**，不并入 v4 性能结果。v4 冻结修复策略后重新完整运行，
一次通过与修复后完成情况均单独记录。

## 4. v4 发布门禁

新 A/B/C/D 只有同时满足以下条件才可标记 `production_ready=true`：

1. model 精确为 `hy3`，provider 为 Tencent TokenHub，端点是已确认的官方 HTTPS 主机；
2. generator 和 Judge 均记录完整无凭据 endpoint、配置、prompt/schema/output hash、temperature、
   非空 base seed、逐 cell/sample 派生 seed 和唯一缓存命名空间；
3. generator 的最多尝试次数由配置冻结；同时报告一次通过率与有界修复后的完成率；
4. `suite_state.json` 与 `suite_summary.json` 字节一致，输入和证据快照存在、非软链接且哈希一致；
5. D 精确绑定同一 C 的请求、计划、检索、passage、草稿和 Claim；
6. 成功与失败文件都通过写入前检查和审计器独立复扫；
7. 实验输入与专家 manifest 中的 Pilot 数据集逐项一致。

自定义 OpenAI 兼容端点、其他模型、离线 fixture 和 v1/v2 旧格式仍可用于工程调试，但自动
降级为 nonformal，不能通过修改产物中的自由文字冒充正式实验。

## 5. 尚未完成

1. 输出级专家 D1–D9 分数并不存在，因而完整自动分—专家分相关与 ICC 不能计算；
2. 完整输出级 D1–D9 好/中/差三档判别、12 类完整对抗集、干净样本误报率与扩展题集；
3. 量表冻结、2 分钟 Demo/GIF 和开源许可证。

最准确的项目表述是：

> **MitoEvidence-Hy3 已完成可追溯应用和评估工程闭环，并以项目负责人确认的 127 条单一
> 专家共识金标作为唯一参考。真实 Hy3 Pilot 已揭示术语判别能力与 Claim 门禁失败模式，正式
> v4 A/B/C/D 已完成 60/60 cells 且通过生产审计；D 点估计最高，但配对 McNemar 检验不显著。
> 完整输出级 D1–D9 好/中/差及 12 类完整对抗实验仍未完成，现有结果仍是小样本方法验证，
> 不等价于大规模医学有效性验证。**
