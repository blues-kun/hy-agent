# 双专家盲标与第三人裁决操作说明

本目录只保存人工工作流模板，不保存 AI 判断。专家 A、专家 B 和裁决者必须使用不同的
账号/代码，文件必须独立保存，禁止多人覆盖同一文件。

## 1. 角色

| 角色 | 可见内容 | 不可见内容 |
|---|---|---|
| 协调员 | 来源材料、item ID、文件哈希、A/B锁定结果 | 不参与科学标签判定 |
| 专家 A | 经白名单生成的来源事实、模板 A、标注指南 | AI预标、专家B结果、裁决结果 |
| 专家 B | 与A相同的来源事实、模板 B、标注指南 | AI预标、专家A结果、裁决结果 |
| 第三裁决者 | 来源事实、已锁定A/B结果、分歧列表 | 裁决前不应以AI预标作为首要依据 |

## 2. 盲标输入必须经过白名单净化

不能把 `annotation_prelabel/` 下的现有 JSONL 原样发给专家，因为里面包含 AI 决定和理由。
协调员应只输出以下来源字段：

- Pilot：`question_id`、`question`；必要时附未经AI改写的任务边界原文；
- Claim：`review_id`、`statement_id`、`paper_id`、`paper_short`、`section`、`triple`、`evidence_text`、原始元数据与全文定位；
- 术语：`term_id`、待判断的原句/术语、原始语境；不得提供 `correct`、`why`、`detector` 等AI建议；
- Review pool：`assessment_id`、PMID/DOI/PMCID、题名、年份、参考文献数、manifest中的全文状态/路径/哈希。

必须移除：`ai_*`、`annotator`、`review_status`、`needs_human_verification`、`suggested_edits`、
AI形成的 `required_claims`、`prohibited_inferences`、`caveats`、`recommended_uses` 和所有AI理由。

## 3. 操作步骤

1. 协调员冻结来源快照，计算 SHA-256，生成不含AI判断的 neutral source packet。
2. 复制 `blind_annotation_A.json` 和 `blind_annotation_B.json`，分别填写专家代码与任务批次。
3. A/B独立标注。任何讨论都必须等两份文件完成并锁定之后进行。
4. 每位专家完成后填写 `completed_at_utc`、`source_snapshot_sha256` 和签名，将
   `review_status` 设为 `human_blind_locked`；锁定后不得原地修改，发现错误需追加 amendment。
5. 协调员对共同字段做机械比较，生成分歧 item ID；此时才把A/B两份交给第三裁决者。
6. 裁决者复制 `adjudication_template.json`，逐项记录A决定、B决定、最终决定、依据与是否仍不可判定。
7. AI预标仅可在裁决者已形成初步判断后用于失败分析；若AI预标改变裁决，必须在
   `ai_prelabel_consulted` 和 `impact_of_ai_prelabel` 中透明记录。
8. 金标由独立导出步骤生成；A/B/裁决文件永久保留且不可覆盖。

## 4. 各类记录至少要标什么

### Pilot question

- `answerability`：`answerable | partial | insufficient | out_of_scope`；
- 可核验的原子主张、必须保留的条件槽位；
- 已知冲突、禁止推断和安全拒答边界；
- 对每条主张给出 DOI/PMID、EvidenceSpan 或明确的证据缺口。

### Claim

- `decision`：`accept | accept_with_edits | reject | uncertain`；
- 证据是结果段原创发现、背景转述还是推断；
- 物种、组织/细胞、干预、剂量、时间、方法、结局、效应方向；
- 是否可用于 β 细胞证据综述及其限定条件。

### Terminology / error pattern

- `decision`：`approve_rule | revise_rule | reject_rule | uncertain`；
- 错误表述、允许表述、适用/不适用条件；
- 规则检测还是语义Judge检测；
- 至少一个独立原文示例。

### Review pool

- `decision`：`include_seed | include_context_only | exclude | uncertain`；
- 主题覆盖与排除范围；
- 全文/许可/锚点可用性；
- 可用于检索种子、背景、冲突发现或何种任务；
- 是否必须回溯原始研究。

## 5. 一致性统计口径

- 只在相同 item、相同标签版本和相同冻结来源上比较 A/B；
- 不同任务类型分别计算，不能把 Pilot、Claim、术语和综述池混成一个 κ；
- 序数评分报告加权 κ、Gwet's AC2、原始一致率和边际分布；
- 名义标签报告 Cohen's κ/Gwet's AC1 和混淆矩阵；
- 条件槽位等多标签字段逐字段报告精确率/召回率或 Jaccard；
- AI预标既不是专家C，也不是金标，禁止计入任何专家一致性分母。

## 6. 锁定与审计

模板中的 `decision`、`rationale` 和 `field_corrections` 初始必须为空。锁定记录至少包含：

- `annotator_code`（不得使用真实姓名公开发布）；
- `guideline_version`；
- `source_snapshot_sha256`；
- `started_at_utc`、`completed_at_utc`；
- `review_status=human_blind_locked`；
- 签名或外部不可变审计ID。

裁决记录必须保留A、B原始决定；如果证据不足以裁决，最终结果应为 `unresolved`，不能强行多数表决。
