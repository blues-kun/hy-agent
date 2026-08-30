# MitoEvidence-Hy3 完成状态与实验边界

> 快照日期：2026-08-30
> 当前结论：可运行、可审计的 Hy3 证据综述与评估 MVP 已完成；现有 127 条记录经
> 项目负责人确认为本项目唯一的专家共识金标。真实 Hy3 五题 Pilot 已跑通，正式
> A/B/C/D Pilot 消融正在执行。样本规模与金标字段缺失必须如实保留。

## 1. 已完成的工程能力

| 工作线 | 已完成内容 | 可核查位置 |
|---|---|---|
| 应用闭环 | Hy3 规划、冻结全文检索、证据约束生成、Claim—EvidenceSpan、XML 锚点与哈希审计包 | `app/`、`scripts/run_pilot_suite.py` |
| 专家共识金标 | 5 道 Pilot、50 条 Claim、60 组术语正误对、12 篇综述池；逐文件 SHA-256 与字段 designation | `annotation_prelabel/expert_gold_manifest.json`、`evaluator/expert_gold.py` |
| 证据底座 | 12 篇种子综述、2,043 条原始参考文献、1,623 条去重候选；7 篇 OA XML 冻结 | `eval/data/evidence_pool_manifest.json` |
| 九维评估 | D1—D9、NA、事件上限、四类致命错误、PASS/REVIEW/REJECT | `evaluator/rubric.py`、`evaluator/assembly.py` |
| Hy3 Judge | Function Calling、JSON Schema 备选、本地校验、自一致性与升级队列 | `evaluator/judge/` |
| 实验协议 | 专家参考一致度、判别力、稳定性、对抗性、A/B/C/D 完整网格审计 | `evaluator/experiment_protocol.py` |
| Pilot 消融 | A 无检索；B 稀疏 TF-IDF 全文向量；C 冻结证据图重排；D 同一 C 草稿的 Hy3 Judge 门控 | `app/ablation.py`、`scripts/run_pilot_ablation.py` |

原始 JSONL 仍保留 `ai_*`、`annotator` 和 `review_status` 等历史字段，不静默改写来源。
项目负责人确认通过独立 manifest 叠加 `expert_consensus_gold` designation；被测应用只读取
中性问题与范围，不能读取专家答案、required claims 或 prohibited inferences。

## 2. 已完成的验证和真实运行

### 2.1 离线回归

- Python 3.11；
- 424 项离线测试全部通过；
- 7/7 份 OA XML 与冻结 SHA-256 一致；
- A/B/C/D 的失败 cell 必须留在分母，D 必须绑定精确 C artifact hash；
- B 明确是稀疏 TF-IDF，不伪称 dense embedding；C 图只读取冻结文本/元数据，不读取金标；
- 安全扫描未发现提交中的 API Key、私钥、本机绝对路径、全文 XML/PDF 或运行目录。

### 2.2 专家金标审计

共 127 条且 ID 唯一：

- Pilot：5 题、30 条 required claims，但 `evidence_papers/evidence_spans` 均为空；
- Claim：50 条，决策分布为 accept 8、accept_with_edits 25、reject 14、uncertain 3；
- 术语：60 组完整 wrong/correct 对，但没有另设 approve/reject 字段；
- 综述池：12 篇、2,043 条参考文献，7 篇有本地 XML 与哈希。

每项只有一份合并专家结果，没有专家 A/B 的独立逐项标签。因此专家间 κ、Gwet 或 ICC
在本项目中不可计算，必须报告 `unavailable`；可以计算的是自动系统对单份专家参考的一致度。

### 2.3 真实 Hy3 五题 Pilot

固定应用版本 `v0.3.1` 在同一套件中完成 5/5：

| Pilot | 召回段落 | 输出主张 | 状态 |
|---|---:|---:|---|
| PILOT-01 | 12 | 7 | 完成 |
| PILOT-02 | 12 | 4 | 完成 |
| PILOT-03 | 12 | 8 | 完成 |
| PILOT-04 | 12 | 3 | 完成 |
| PILOT-05 | 0 | 0 | 正确越界拒答 |

与专家 `answerability` 的原始一致率为 0.60，Cohen κ=0.375（n=5，仅作 Pilot 描述）：
两道专家标为 `answerable` 的题被系统保守降为 `partial`；两道 `partial` 和一道
`out_of_scope` 判定一致。该 κ 是“Hy3—单份专家参考”一致度，不是专家间可靠性。

真实运行还暴露并修复了两个工程问题：越界输出仍产生 claims，以及复杂综合题在
4,096 思考预算下无正文。当前越界强制 `claims=[]`，综合阶段预算为 8,192，并接入
JSON Schema 备选通道；失败过程保留为回归证据，没有从分母删除。

## 3. 尚不能宣称完成的部分

1. 正式 A/B/C/D 多重复结果、置信区间和逐题九维评分尚未生成；当前运行是 5 题 Pilot。
2. 判别力仍需把60组术语正误对转成受控好/中/差或配对输出并完成评分。
3. 稳定性需要同一输出重复评估；当前只有单次五题生成。
4. Pilot 金标没有原文 EvidenceSpan，因此不能计算完整 D2/D3 证据金标指标；不得补造锚点。
5. 没有输出级专家九维分数，自动总分对专家总分的 Spearman/MAE/ICC 不适用。
6. 本地术语词表不是完整 MeSH/GO 镜像。
7. 2 分钟 Demo/GIF 和开源许可证尚未完成。

## 4. 接下来的固定顺序

1. 完成 5题×A/B/C/D×1重复的真实 Pilot 网格，失败记录不删除；
2. 用专家 answerability、required claims、prohibited inferences 和术语正误对进行适用范围内评分；
3. 对60组术语对运行判别力与对抗检出，不把缺少的输出级专家分数默认成正确；
4. 对固定输出重复运行 Judge，报告稳定性和采样分歧；
5. 再决定是否扩展重复数和题量；
6. 生成逐题结果、失败 Case、README 发布版和2分钟演示。

当前最准确的项目表述是：

> **MitoEvidence-Hy3 已完成应用和评估工程闭环，并以127条项目负责人确认的专家共识
> 记录作为唯一参考；真实五题已跑通。当前结果是小样本 Pilot，不等价于大规模医学
> 有效性验证，缺失的专家字段与专家间一致性不会被推断或补造。**
