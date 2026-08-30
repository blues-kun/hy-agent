"""MitoEvidence-Hy3 评估器包。

模块划分：
  - schemas    数据契约（原子主张、证据片段、金标记录、评估结果）；
  - rubric     九维计分引擎（阈值全部来自 configs/rubric_v0_1.yaml）；
  - rules      确定性规则层（标识符核验、结构必需项、数字与单位）；
  - judge      Hy3 语义判定与自一致性聚合；
  - assembly   九维输入与致命错误的可审计汇总；
  - expert_gold 现有127条专家共识金标的哈希审计与无损读取；
  - blind      可选新A/B可靠性研究的盲标包、锁定校验与裁决入口；
  - validation 判别力、一致性、稳定性和对抗性统计。

量表本身仍是未冻结的 v0.1；现有金标不含独立A/B列，不能据此计算专家间一致性。
"""
from evaluator.rubric import EVALUATOR_VERSION, evaluate, load_rubric

__all__ = ["EVALUATOR_VERSION", "evaluate", "load_rubric"]
__version__ = "0.1.0"
