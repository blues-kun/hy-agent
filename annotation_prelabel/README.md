# 标注样例

[在线浏览](https://blues-kun.github.io/hy-agent/) · [原始校验清单](expert_gold_manifest.json) · [当前核验训练数据](../mito/data/README.md)

本目录保留上一阶段的 127 条参考标注，原始字段、记录和校验哈希不变。当前核验 SFT 的 443 条训练记录和 51 条开发记录位于 `mito/data/`；两批数据有不同任务组织方式，不直接相加为独立样本总量。

| 数据 | 条数 | 内容 |
| :--- | ---: | :--- |
| [Pilot 问题](pilot_questions/pilot_5_questions.jsonl) | 5 | 可回答性、必需主张、条件与证据缺口 |
| [Claim 审核](claim_review_sample/claim_review_sample.jsonl) | 50 | 准入判断、理由、缺陷及修改建议 |
| [术语正误对](terminology_blacklist/terminology_blacklist.jsonl) | 60 | 正误表述、理由与检测方式 |
| [综述池评估](review_pool_assessment/review_pool_assessment.jsonl) | 12 | 综述用途、来源信息与未解决事项 |

项目负责人已确认这些记录为可用的单份合并专家参考。`ai_*`、`annotator` 和 `review_status` 是保留的历史字段，不据此补造新的审核或第二位独立标注者。空证据字段、未知状态与修改建议按原值读取。

在仓库根目录执行以下命令，检查源文件哈希及展示数据一致性：

```bash
python scripts/build_annotation_review_site.py --check
```

旧评测框架及详细历史文档保留在 [legacy/mitoevidence-20260831](https://github.com/blues-kun/hy-agent/tree/legacy/mitoevidence-20260831)，当前训练不依赖旧框架。
