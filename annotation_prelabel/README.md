# 专家共识标注（金标；目录名为历史遗留）

> **2026-08-30 口径更正：项目负责人确认，本目录四个 JSONL 中的现有判断就是本项目
> 可用的专家标注结果，后续没有另一轮 A/B 标注或第三人裁决。实验按这些原值使用，禁止补造
> 标签。** `annotation_prelabel` 目录名，以及 `annotator`、`review_status`、`ai_*` 等字段名
> 是历史遗留，为保留原始快照和兼容既有代码不做原地改写；权威 designation 与 SHA-256 见
> [`expert_gold_manifest.json`](expert_gold_manifest.json)。

## 使用口径

- 这 127 条是四种不同任务的**单份合并专家结果**，不是 127 个同质样本：Pilot 题目 5 条、
  Claim 审核 50 条、术语规则 60 条、综述池审核 12 条。
- `ai_decision`、`ai_confidence`、`ai_reasoning` 按历史列名原样读取；在对应数据集内分别作为
  已提供的专家决定、置信度与理由。不要为了字段好看而复制或改写这些值。
- `needs_human_verification` / `required_human_checks` 现在表示专家结果中**保留的未解决事项或
  使用限制**，不是等待未来标注的任务单。实验遇到这些缺口必须按不确定/不可核验处理，不能
  假定为通过。
- `human_review_workflow/` 只有空白旧模板，不属于金标，不应等待这些模板交卷，也不得把空值
  复制成第二位专家标签。
- 生成模型时不得把 required claims、答案可回答性、禁止推断等金标字段放进 prompt；它们只在
  生成完成后的盲评阶段使用，避免标签泄漏。

## 可用标签与边界

| 数据集 | 条数 | 可直接使用的专家金标 | 已知完整性边界 |
|---|---:|---|---|
| `pilot_questions/pilot_5_questions.jsonl` | 5 | `answerability`、30 条 `required_claims`、上下文槽位、冲突与禁止推断 | 5/5 的 `evidence_papers` 和 `evidence_spans` 均为空；只能作为题目/主张级金标，不能伪装成完整 `QuestionGold` |
| `claim_review_sample/claim_review_sample.jsonl` | 50 | 历史列 `ai_decision`、置信度、理由、缺陷码、修改建议与可用性 | 12 条 `recorded_conditions` 为空；3 条 `usable_for_beta_cell_evidence=null`，必须保留 null |
| `terminology_blacklist/terminology_blacklist.jsonl` | 60 | `wrong` / `correct` 规则对、理由、例子、维度与 detector | 全部规则对齐全，但没有单独的 approve/reject 列；22 条无本地语料观测，20 条无未解决事项 |
| `review_pool_assessment/review_pool_assessment.jsonl` | 12 | 历史列 `ai_decision`、池角色、用途、限制与检查项 | 书目 PMID/DOI/题名/年份为 12/12，PMCID 为 9/12；本地 XML 路径与 SHA-256 为 7/12 |

综述池 12 条的 `reference_count` 合计 2,043；去重后的候选池数量是另一层数据统计，不能把
2,043 或候选池条数当作专家标注记录数。

## 专家一致性是否可计算

**不可计算。** 四个 JSONL 每个 item 只有一份合并结果，没有专家 A、专家 B 的逐项独立标签、
评审者代码或配对评分。现有 `annotator` 也只有一个历史值。因而无法从这些文件计算或恢复：

- Cohen's κ、加权 κ、Gwet's AC1/AC2 或原始 A/B 一致率；
- 专家总分 ICC；
- 分歧率或第三人裁决率。

正式实验应把“专家一致性”明确报告为 `unavailable: no independent per-rater labels`。不得把同一
金标复制两份、把 Hy3 当第二位专家，或把不同任务类型互相配对来生成看似有效的一致性数字。
仍可计算的是自动评估器相对于这份专家共识金标的准确率、相关性、判别力和对抗鲁棒性。

## 离线审计与读取

```bash
.venv/bin/python scripts/audit_expert_gold.py \
  --out results/expert_gold_audit.json
```

审计器会核对四个文件的 SHA-256、条数、ID 唯一性、全部顶层字段的 `present/non_null/non_empty`
计数，并输出上述缺口；任何哈希漂移、漏行、重复 ID 或指定金标字段缺失都会失败。代码读取入口：

```python
from evaluator.expert_gold import load_expert_gold_records, selected_gold_fields

datasets = load_expert_gold_records()
claim = datasets["claim_reviews"][0]
gold = selected_gold_fields("claim_reviews", claim)
```

该入口返回原值，不推导跨任务统一标签，也不填充 null、空数组、证据锚点、专家身份或 A/B 评分。
