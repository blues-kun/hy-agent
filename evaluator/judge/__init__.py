"""L2 语义 Judge：冻结配置的 Hy3 推理 Judge + 自一致性置信（方案 8.4、7.3）。

分层：
  - config.py            configs/judge_v0_1.yaml 的只读视图；
  - prompts.py           缓存友好的提示模板（稳定 system 前缀 + 逐主张 user 消息）；
  - hy3_client.py        TokenHub OpenAI 兼容客户端：节流、退避、双结构化通道、本地校验；
  - self_consistency.py  k 次采样多数票聚合与人工升级判定。
"""
from evaluator.judge.config import JudgeConfig, default_judge_config, load_judge_config
from evaluator.judge.hy3_client import Hy3Client, Hy3Transport, JudgeCallResult
from evaluator.judge.self_consistency import (
    JudgeAggregate,
    JudgeSample,
    TokenUsage,
    aggregate_samples,
    run_self_consistency,
)

__all__ = [
    "Hy3Client",
    "Hy3Transport",
    "JudgeAggregate",
    "JudgeCallResult",
    "JudgeConfig",
    "JudgeSample",
    "TokenUsage",
    "aggregate_samples",
    "default_judge_config",
    "load_judge_config",
    "run_self_consistency",
]
