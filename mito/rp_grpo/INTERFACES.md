# 工具与接入契约

权威定义是 `environment.py::TOOL_SCHEMAS`。执行器检查字段白名单、类型与枚举；工具访问冻结快照，不修改实验数据。

| 工具 | 主要参数 | 返回或作用 |
| --- | --- | --- |
| `list_documents` | paper_id / offset / limit | 授权目录、分页、范围完整性与测量目录 |
| `search_evidence` | query / paper_id / limit | 词法检索候选、位置与命中情况 |
| `read_passage` | evidence_id | 原文、来源位置与内容哈希 |
| `read_statement` | evidence_id / statement_index | 原文中的指定句段 |
| `query_metrics` | experiment_id / metric / aggregation / wavelength_nm | 快照中的限定描述统计 |
| `finalize` | answerability / scope / claims / measurements / reason | 提交与已读原文或结果绑定的答案 |
| `stop` | scope / reason | 检查授权范围后结束任务 |

`read_statement` 读取句段，不将图谱可达性等同于因果关系。`query_metrics` 仅支持 schema 中的固定指标和聚合，例如 `median_all_planes` 与 mean/median；数值的测量层级、单位与版本随真实返回提供。

## Python 接口

准备[外置数据](data/README.md)后，在 `mito/` 执行：

```python
from rp_grpo.environment import ResearchEnvironment, load_tasks

env = ResearchEnvironment(max_steps=6)
task = load_tasks(split="train")[0]
observation = env.reset(task)
result = env.step({"tool": "list_documents", "arguments": {"limit": 20}})
# result 包含 observation、done、reward、info。
# actor 只接收 observation；reward/info 保留在训练与审计侧。
```

策略每次输出一个动作 JSON：

```json
{"tool":"search_evidence","arguments":{"query":"待查原文短语","limit":5}}
```

`train_policy.py` 的 `initial_messages`、`append_observation`、`format_prompt` 统一装配上下文。模型使用同一基座 tokenizer、聊天模板和 `enable_thinking=false` 设置，只生成动作，不生成工具返回。

## 状态与记录

`reset` 核查任务登记、视图与文件哈希。初始观察不暴露 task/pair/side/split、评分器或全局语料表；条件差异通过实际工具返回显现。快照缺失使用 `unavailable_in_authorized_snapshot` 表达，不推断材料在其他范围是否存在。

每步保存原始模型文本、解析动作、实际返回、哈希、调用数与时间；B0 额外保存原动作和规则干预。独立重放验证这些记录的一致性。后续原文和目标标签不能提前进入当前观察。

## 应用接入

线上适配器将授权检索与分析结果映射为相同的来源 ID、位置、条件、版本与哈希。大模型直接使用证据和分析结果；工具策略负责下一步行动，核验器独立提供主张检查建议。

执行器将动作绑定任务、轮次、调用 ID 与结果版本，统一处理取消、重试和缓存。工具版本或 adapter 变更后重新执行准入。此训练模块不自动发布模型或连接生产账号；具体工作台接入由应用适配层完成。
