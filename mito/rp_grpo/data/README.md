# RP 外置数据契约

本目录保留说明。RP 使用的原文段落、测量快照、任务评分标签与完整执行轨迹不随本次源码发布；它们与 `mito/data/` 已公开的核验标注是两项独立数据。

运行工具 SFT、rollout 或 RL 前，将同一版本的数据快照装配到这里：

| 文件 | 用途 |
| --- | --- |
| `tasks.jsonl` | 任务、分区、问题族、配对关系、可见初始观察与视图绑定 |
| `views.jsonl` | 每个授权视图可访问的段落与测量记录 ID |
| `passages.jsonl` | 原文内容、来源、位置与内容哈希 |
| `measurements.csv` | 描述统计所需的版本化测量记录 |
| `scorers.jsonl` | 独立评分目标，仅环境与评分器读取 |
| `MANIFEST.json` | 数据版本、文件哈希与来源分区 |
| `planner_sft.train.jsonl` / `planner_sft.dev.jsonl` | 从完整执行轨迹拆出的助手决策样本 |
| `reference_traces.jsonl` | 可独立重放的参考路径 |
| `REFERENCE_MANIFEST.json` | 参考路径、SFT 文件与数据清单的哈希绑定 |

前六项用于环境；工具 SFT 另需后三项对应的文件。也可通过入口的 `--train-data`、`--scorers` 或 `--tasks` 等参数选择外部目录，相关文件仍需位于同一个清单绑定目录；详见各入口 `--help`。

## 一致性要求

任务与配对变体按来源或实验组划分为训练集（`train`）与验证集（`dev`）；同一问题族不跨区。模型仅接收 `actor_observation` 与真实工具返回，不接收 `pair_id`、侧别、目标原文 ID、评分标签或未来步骤的信息。

`MANIFEST.json` 的版本与 `build_data.py::VERSION` 一致，`files` 保存受检文件名及 SHA-256。环境检查文件内容、视图哈希与任务登记；工具 SFT 还检查 `REFERENCE_MANIFEST.json` 的 `dataset_manifest_sha256`、执行与独立重放状态，并实际重放参考路径。

SFT 每行仅监督最后一个助手动作。`--max-reference-steps` 先按步数预算筛选整条路径；`--skip-overlength` 按 token 长度排除整条超长路径，不保留无终态的半条示范。样本数量以当前验证报告为准。

`build_data.py` 保留源系统导出适配逻辑，需提供兼容的源材料与测量表；它不会下载或凭空生成科研数据。最小合成测试数据仅用于代码检查，不替代训练快照。
