# AI 预标材料：禁止直接作为金标

> **醒目声明：本目录中的判断均为 AI 预标，不是专家金标。** 现有预标记录的
> `annotator` 必须保持为 `claude-fable-5-thinking (AI预标)`，`review_status`
> 必须保持为 `ai_prelabel_pending_human`。在两名专家完成相互独立的盲标并经第三人
> 裁决以前，不得把这些记录用于报告专家一致性、校准评估器、训练奖励模型或宣称科学结论。

## 1. 正确用途与禁用用途

这些材料只用于：

- 帮助项目负责人发现待核验的主张、术语、文献和评测设计缺口；
- 在两位专家交卷并锁定之后，作为裁决阶段的附加参考；
- 记录 AI 预标与最终专家结论之间的差异，分析自动预标的失败模式。

这些材料不得用于：

- 计算专家 A 与专家 B 的 Cohen's κ、加权 κ、Gwet's AC2 或原始一致率；
- 在盲标前向专家展示 AI 的决定、理由、置信度、建议修改或缺陷代码；
- 把 `accept`、`reject`、`answerable` 等 AI 字段直接复制成最终标签；
- 将待审核记录写入 gold、validation 或 test 数据集；
- 将综述本身或综述参考文献列表直接当作某一实验结论的一手证据。

## 2. 人工审核的强制顺序

1. **冻结来源快照**：记录题目、原文、文献元数据、文件哈希和 item ID；来源事实可见，AI 判断不可见。
2. **专家 A 独立盲标**：仅使用 `human_review_workflow/blind_annotation_A.json`；完成后签名并锁定。
3. **专家 B 独立盲标**：仅使用 `human_review_workflow/blind_annotation_B.json`；不得查看 A 的结果；完成后签名并锁定。
4. **计算一致性**：只比较 A、B 两份已锁定人工记录；AI 预标不得作为第三名评分者，也不得计入分母。
5. **第三人裁决**：仅对分歧项使用 `human_review_workflow/adjudication_template.json`，同时查看来源、A/B 理由；AI 预标只能在裁决者形成初步判断后作为补充材料打开。
6. **生成金标**：保留 A、B、裁决三份不可覆盖的审计记录；最终 gold 记录须具有独立的新 ID、专家签名、时间戳和来源快照哈希。

详细操作与锁定规则见
[`human_review_workflow/README.md`](human_review_workflow/README.md)。

## 3. 子目录说明

| 子目录 | 内容 | 当前性质 |
|---|---|---|
| `pilot_questions/` | 5 个 Pilot 问题、范围、AI 建议的原子主张和风险点 | AI 预标；证据论文和 EvidenceSpan 尚待专家补齐 |
| `claim_review_sample/` | 50 条本地图谱 Claim 的 AI 质量预审 | AI 预标；不得直接决定 Claim 准入 |
| `terminology_blacklist/` | 60 条术语、方向和推断错误模式候选 | AI 预标；“黑名单”条目必须由专家逐条批准 |
| `review_pool_assessment/` | 对 12 篇种子综述的逐篇 AI 预评估，含全文可用性与推荐用途 | AI 预标；综述只能作为检索入口或二级证据，不能自动替代原始研究 |
| `human_review_workflow/` | 专家 A/B 独立盲标模板、第三人裁决模板和操作说明 | 空白人工模板；不含 AI 判断 |

`claim_review_sample/__pycache__/` 是本地运行缓存，不是标注材料，提交或打包时应排除。

## 4. 公共字段

| 字段 | 含义与约束 |
|---|---|
| `annotator` | 本目录 AI 预标固定为 `claude-fable-5-thinking (AI预标)`；人工模板使用独立的专家代码 |
| `review_status` | AI 预标固定为 `ai_prelabel_pending_human`；只有独立人工工作流才能产生 `human_blind_locked` 或 `adjudicated` |
| `ai_confidence` | AI 对自身预标的主观置信度，不等于科学证据等级，不得替代专家判断 |
| `needs_human_verification` | 必须由专家逐项处理的问题清单；非空时不得进入金标 |
| `source_reviews` / `pmid` / `doi` | 文献定位信息，不等于主张已经被原文支持 |
| `evidence_text` / `evidence_spans` | 证据文本或锚点；最终金标还需验证原文、章节、条件和效应方向 |
| `review_status`（人工） | A/B 交卷锁定后为 `human_blind_locked`；第三人完成后为 `adjudicated`，状态变化必须追加记录而非覆盖 |

各子目录的专用字段：

- Pilot：`answerability`、`required_claims`、`required_context_slots`、`prohibited_inferences`；
- Claim：`ai_decision`、`defect_codes`、`recorded_conditions`、`usable_for_beta_cell_evidence`；
- 术语：`wrong`、`correct`、`maps_to_dimension`、`detector`；
- 综述池：`fulltext_status`、`pool_role`、`recommended_uses`、`caveats`。

## 5. 当前数量与边界

- Pilot 问题：5 条；
- Claim 预审：50 条；
- 术语/错误模式：60 条；
- 种子综述逐篇评估：12 条；
- 总计：127 条 AI 预标记录（不同任务类型不能直接合并计算一致性）。

12 篇种子综述合计提供 2,043 条原始参考文献，去重后候选池为 1,623 条；这只是
**候选证据池**，不是 1,623 条金标。当前 manifest 实际落盘并带 SHA-256 的 XML
全文为 7/12 篇；核验报告中早期“8/12 可走 fullTextXML”的路线判断不得覆盖实际
manifest 结果。

## 6. 进入金标前的最低门槛

每条最终记录至少满足：

- 专家 A/B 独立交卷且文件已锁定；
- 分歧项已经第三人裁决；
- DOI/PMID/PMCID 与题名核验通过；
- EvidenceSpan 可在冻结全文中重定位，或明确标记全文不可得；
- 物种、样本/细胞类型、干预、剂量、时间、方法、结局和效应方向按适用性补齐；
- 事实、相关性、因果、推断和不确定性没有混写；
- 保存来源快照哈希、标注人代码、时间戳和变更记录。
