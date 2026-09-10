# 证据核验标注

输入为问题、实验条件、审核证据摘要、待核验主张及引用 ID；输出为支持关系、条件匹配、错误类型、解释与必要改写。

| 分区 | 主张 | 问题 | 支持 | 不足 | 反驳 | 混合 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [train.jsonl](train.jsonl) | 443 | 43 | 233 | 149 | 56 | 5 |
| [dev.jsonl](dev.jsonl) | 51 | 5 | 33 | 10 | 8 | 0 |

每条主张是一条训练记录；同题多条主张共享背景与证据，不等于独立实验。开发集用于模型选择；本次不发布冻结测试题与答案。

## 文件契约

```text
id / task_id
prompt: [system, user]
completion: [assistant]
metadata: split、材料模式、原始记录定位与哈希
```

`completion[0].content` 是 JSON 字符串，字段由 [FORMAT_CONTRACT.json](FORMAT_CONTRACT.json) 定义。支持标签为 `supported`、`contradicted`、`insufficient`、`mixed`。`metadata` 仅用于溯源与分区检查，不进入模型输入或训练损失。

本次公开导出仅将 `metadata.original_record.file` 的内部绝对路径改为源文件名，保留原有 prompt、completion、labels 与原始内容哈希；没有新增专家审核。文件大小、公开文件 SHA-256、源文件 SHA-256 与类别统计见 [MANIFEST.json](MANIFEST.json)。

默认仅 `train.jsonl` 用于优化，`dev.jsonl` 用于开发评测。仓库中的历史 `annotation_prelabel/` 是另一份材料，不自动与这里的记录合并计数或混训。将证据摘要换为原文后，需要重新确认标签与输入的一致性。

→ [核验训练说明](../docs/verifier-sft.md)
