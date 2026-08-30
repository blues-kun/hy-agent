"""九维计分引擎（方案 8.2 / 8.3）。

职责边界：
  - 本模块只做算术与判定，全部阈值、权重、上限与决策门槛从
    configs/rubric_v0_1.yaml 读取，代码内不硬编码任何量表数值；
  - 各维的连续指标由规则层（evaluator/rules/）与 Judge 层提供，本模块不猜测
    缺失输入：九维必须全部显式给出（NA 也要显式声明），否则报错。
    依据方案 10.3：「工具超时、Schema 失败或缺失输出按预注册规则记为失败，
    不能从分母删除」。

计分公式（方案 8.2）：
    RawScore = 100 * Σ_d w_d * (s_d / 4) / Σ_d w_d
某维记 NA 时从分子分母同时移除，其余维度按原权重比例重归一。

致命错误（方案 8.3）：
    FinalScore = min(RawScore, 所有已触发的分数上限)
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from evaluator.schemas import (
    AtomicClaim,
    DimensionScore,
    EvaluationResult,
    FatalErrorRecord,
    JudgeVerdict,
    ReleaseDecision,
    StrictModel,
)

EVALUATOR_VERSION = "mitoevidence-evaluator-0.1.0"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "rubric_v0_1.yaml"

# 浮点容差：仅用于抵消 n/d 形式指标与十进制阈值之间的表示误差，
# 不改变方案 8.2 的任何判据（例如 p=0.98 必须落入 3 分档而非 2 分档）。
BAND_EPSILON = 1e-9

DIMENSION_ORDER = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")


class RubricConfigError(ValueError):
    """量表配置自身不自洽时抛出，绝不静默容忍。"""


# ---------------------------------------------------------------------------
# 引擎输入
# ---------------------------------------------------------------------------


class DimensionInput(StrictModel):
    """单维的评分输入。

    三种给法（互斥）：
      1. is_na=True                     —— 该维整体不适用，从加权分母移除；
      2. metric_value=<float>           —— 连续指标，由本引擎按 bands 分档；
      3. level=<0..4>                   —— 直接给档位（metric=decision_table 的
                                            D6，或人工评分直接落档时使用）。
    event_flags 承载方案对某些档位附加的事件性条件，键名见配置 event_caps.flag。
    """

    is_na: bool = False
    metric_value: float | None = None
    level: int | None = Field(default=None, ge=0, le=4)
    event_flags: dict[str, bool | int | float] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _exactly_one_source(self) -> DimensionInput:
        if self.is_na:
            if self.metric_value is not None or self.level is not None:
                raise ValueError("is_na=True 时不得同时给出 metric_value 或 level")
            return self
        if (self.metric_value is None) == (self.level is None):
            raise ValueError("必须给出 metric_value 或 level 之一（且只给一个）")
        return self


# ---------------------------------------------------------------------------
# 配置装载
# ---------------------------------------------------------------------------


class RubricConfig:
    """configs/rubric_v0_1.yaml 的只读视图。"""

    def __init__(self, raw: Mapping[str, Any], source_path: Path | None = None, sha256: str = ""):
        self.raw = raw
        self.source_path = source_path
        self.sha256 = sha256
        self.version: str = str(raw["version"])
        self.frozen: bool = bool(raw.get("frozen", False))
        self.scale: Mapping[str, Any] = raw["scale"]
        self.dimensions: Mapping[str, Mapping[str, Any]] = raw["dimensions"]
        self.fatal_errors: Mapping[str, Mapping[str, Any]] = raw["fatal_errors"]
        self.release_decision: Mapping[str, Any] = raw["release_decision"]
        self._validate()

    def _validate(self) -> None:
        missing = [d for d in DIMENSION_ORDER if d not in self.dimensions]
        if missing:
            raise RubricConfigError(f"配置缺少维度定义：{missing}")
        extra = [d for d in self.dimensions if d not in DIMENSION_ORDER]
        if extra:
            raise RubricConfigError(f"配置出现未知维度：{extra}")
        total = sum(int(self.dimensions[d]["weight"]) for d in DIMENSION_ORDER)
        declared = int(self.scale["total_weight"])
        if total != declared:
            raise RubricConfigError(f"权重之和 {total} 与声明的 total_weight {declared} 不一致")
        for dim_id in DIMENSION_ORDER:
            bands = self.dimensions[dim_id].get("bands") or []
            levels = [int(b["level"]) for b in bands]
            if levels != sorted(levels, reverse=True):
                raise RubricConfigError(f"{dim_id} 的 bands 必须按 level 降序排列")
            if bands and levels != [4, 3, 2, 1, 0]:
                raise RubricConfigError(f"{dim_id} 的 bands 必须覆盖 4/3/2/1/0 五档，实际 {levels}")

    @property
    def max_level(self) -> int:
        return int(self.scale["max_level"])

    def weight(self, dim_id: str) -> int:
        return int(self.dimensions[dim_id]["weight"])

    def name_zh(self, dim_id: str) -> str:
        return str(self.dimensions[dim_id]["name_zh"])

    def metric_name(self, dim_id: str) -> str | None:
        value = self.dimensions[dim_id].get("metric_name")
        return None if value is None else str(value)


def load_rubric(path: str | Path | None = None) -> RubricConfig:
    """从 YAML 装载量表配置，并记录文件哈希写入 run manifest。"""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    text = cfg_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return RubricConfig(yaml.safe_load(text), source_path=cfg_path, sha256=digest)


@lru_cache(maxsize=4)
def _cached_rubric(resolved: str) -> RubricConfig:
    return load_rubric(resolved)


def default_rubric() -> RubricConfig:
    """默认配置（带缓存）；测试中如需改配置请直接调用 load_rubric。"""
    return _cached_rubric(str(DEFAULT_CONFIG_PATH))


# ---------------------------------------------------------------------------
# 指标 → 档位
# ---------------------------------------------------------------------------


def band_for(dim_id: str, metric_value: float, config: RubricConfig | None = None) -> int:
    """按配置 bands 把连续指标折算为 0—4 档位。"""
    cfg = config or default_rubric()
    bands = cfg.dimensions[dim_id].get("bands")
    if not bands:
        raise RubricConfigError(f"{dim_id} 未定义 bands，无法由指标分档（应直接给出 level）")
    for band in bands:  # 已校验为 level 降序
        lower = float(band["lower"])
        if band.get("lower_exclusive", False):
            if metric_value > lower + BAND_EPSILON:
                return int(band["level"])
        elif metric_value >= lower - BAND_EPSILON:
            return int(band["level"])
    return int(bands[-1]["level"])


def _flag_triggered(spec: Mapping[str, Any], flags: Mapping[str, Any]) -> bool:
    name = str(spec["flag"])
    if name not in flags:
        return False
    value = flags[name]
    threshold = spec.get("when_at_least")
    if threshold is None:
        return bool(value)
    if isinstance(value, bool):
        # 统一用 ValueError 报告一切评估输入错误，便于调用方单点捕获。
        raise ValueError(  # noqa: TRY004
            f"事件 flag {name} 定义了 when_at_least，应传数值而非布尔值"
        )
    return float(value) >= float(threshold)


def apply_event_caps(
    dim_id: str,
    level: int,
    event_flags: Mapping[str, Any],
    config: RubricConfig | None = None,
) -> tuple[int, list[str]]:
    """施加档位上限。

    方案 8.2 D1 原文即为此机制：「先按比例分档，再应用事件上限，最终取较低档」；
    其余维度中形如「4 分：p≥0.95，无关键槽位错误」的附加条件按同一机制折算。
    """
    cfg = config or default_rubric()
    applied: list[str] = []
    capped = level
    for spec in cfg.dimensions[dim_id].get("event_caps") or []:
        if _flag_triggered(spec, event_flags):
            capped = min(capped, int(spec["max_level"]))
            applied.append(str(spec["flag"]))
    return capped, applied


def score_dimension(
    dim_id: str, item: DimensionInput, config: RubricConfig | None = None
) -> DimensionScore:
    """把单维输入折算为 DimensionScore，保留连续指标、上限前档位与生效上限。"""
    cfg = config or default_rubric()
    common = {
        "dimension": dim_id,
        "name_zh": cfg.name_zh(dim_id),
        "weight": cfg.weight(dim_id),
        "notes": item.notes,
    }
    if item.is_na:
        return DimensionScore(is_na=True, metric_name=cfg.metric_name(dim_id), **common)

    if item.level is not None:
        pre_cap = int(item.level)
    else:
        pre_cap = band_for(dim_id, float(item.metric_value), cfg)
    level, applied = apply_event_caps(dim_id, pre_cap, item.event_flags, cfg)
    return DimensionScore(
        level=level,
        metric_name=cfg.metric_name(dim_id),
        metric_value=item.metric_value,
        level_before_event_caps=pre_cap,
        event_caps_applied=applied,
        **common,
    )


# ---------------------------------------------------------------------------
# 加权折算
# ---------------------------------------------------------------------------


def compute_raw_score(
    dimension_scores: Mapping[str, DimensionScore], config: RubricConfig | None = None
) -> tuple[float, int]:
    """返回 (RawScore, 参与计分的权重之和)。

    NA 维度从分子分母同时移除，等价于其余维度按原权重比例重归一。
    """
    cfg = config or default_rubric()
    max_level = cfg.max_level
    numerator = 0.0
    weight_sum = 0
    for dim_id in DIMENSION_ORDER:
        score = dimension_scores[dim_id]
        if score.is_na:
            continue
        weight_sum += score.weight
        numerator += score.weight * (score.level / max_level)
    if weight_sum == 0:
        raise ValueError("九维全部记 NA，无法计算 RawScore；请检查评估输入")
    return 100.0 * numerator / weight_sum, weight_sum


def apply_fatal_caps(
    raw_score: float,
    fatal_error_keys: Iterable[str],
    config: RubricConfig | None = None,
) -> tuple[float, int | None, list[FatalErrorRecord]]:
    """施加致命错误上限；多个同时触发取最低（方案 8.3）。"""
    cfg = config or default_rubric()
    records: list[FatalErrorRecord] = []
    for key in fatal_error_keys:
        spec = cfg.fatal_errors.get(key)
        if spec is None:
            raise ValueError(
                f"未知致命错误类型 {key!r}；合法取值：{sorted(cfg.fatal_errors)}"
            )
        records.append(
            FatalErrorRecord(
                key=key, label_zh=str(spec["label_zh"]), score_cap=int(spec["score_cap"])
            )
        )
    if not records:
        return raw_score, None, records
    cap = min(r.score_cap for r in records)
    return min(raw_score, float(cap)), cap, records


# ---------------------------------------------------------------------------
# 发布决策
# ---------------------------------------------------------------------------


def decide_release(
    final_score: float,
    fatal_errors: Sequence[FatalErrorRecord],
    dimension_scores: Mapping[str, DimensionScore],
    unresolved_unverifiable: bool = False,
    config: RubricConfig | None = None,
) -> tuple[ReleaseDecision, list[str]]:
    """方案 8.3 发布决策。

    判定顺序及其依据：
      1. 触发任一致命错误，或低于 70 分 → 拒绝发布；
      2. 存在未解决的「不可核验」项 → 必须人工复核（方案原文为无条件表述，
         因此优先于 PASS，见 eval/rubric.md 待澄清 E）；
      3. ≥85 分、无致命错误且 D1/D2/D4/D6 均 ≥3 → 建议发布；
      4. [70, 85) 且无致命错误 → 必须人工复核；
      5. 其余组合（方案未覆盖）→ 配置的 residual_decision，见待澄清 D。
    """
    cfg = config or default_rubric()
    rules = cfg.release_decision
    reject, review, passing = rules["reject"], rules["review"], rules["pass"]
    reasons: list[str] = []

    if fatal_errors:
        keys = ", ".join(r.key for r in fatal_errors)
        return ReleaseDecision.REJECT, [f"触发致命错误：{keys}"]
    if final_score < float(reject["below_final_score"]):
        return ReleaseDecision.REJECT, [
            f"FinalScore {final_score:.2f} 低于 {reject['below_final_score']}"
        ]

    if unresolved_unverifiable and bool(rules.get("unverifiable_overrides_pass", True)):
        return ReleaseDecision.REVIEW, ["存在未解决的「不可核验」项"]

    gate_dims = list(passing["min_level_dimensions"])
    min_level = int(passing["min_level"])
    below = [
        d
        for d in gate_dims
        if dimension_scores[d].is_na or (dimension_scores[d].level or 0) < min_level
    ]
    if final_score >= float(passing["min_final_score"]) and not below:
        return ReleaseDecision.PASS, [
            (
                f"FinalScore {final_score:.2f} ≥ {passing['min_final_score']}，无致命错误，"
                f"{'/'.join(gate_dims)} 均 ≥{min_level} 分"
            )
        ]
    if below:
        detail = ", ".join(
            f"{d}=NA" if dimension_scores[d].is_na else f"{d}={dimension_scores[d].level}"
            for d in below
        )
        reasons.append(f"关键维度未达 {min_level} 分门槛：{detail}")

    if float(review["min_final_score"]) <= final_score < float(review["below_final_score"]):
        reasons.append(
            f"FinalScore {final_score:.2f} 落入 "
            f"[{review['min_final_score']}, {review['below_final_score']}) 复核区间"
        )
        return ReleaseDecision.REVIEW, reasons

    residual = ReleaseDecision(str(rules.get("residual_decision", "REVIEW")))
    reasons.append("方案 8.3 三条规则未覆盖该组合，按 residual_decision 兜底")
    return residual, reasons


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def evaluate(
    question_id: str,
    dimension_inputs: Mapping[str, DimensionInput],
    fatal_error_keys: Iterable[str] = (),
    unresolved_unverifiable: bool = False,
    output_id: str | None = None,
    config: RubricConfig | None = None,
) -> EvaluationResult:
    """完整评分：九维分档 → 加权折算 → 致命错误上限 → 发布决策。"""
    cfg = config or default_rubric()
    missing = [d for d in DIMENSION_ORDER if d not in dimension_inputs]
    if missing:
        raise ValueError(
            f"九维必须全部显式给出（不适用请显式 is_na=True），缺少：{missing}"
        )
    unknown = [d for d in dimension_inputs if d not in DIMENSION_ORDER]
    if unknown:
        raise ValueError(f"出现未知维度：{unknown}")

    scores = {d: score_dimension(d, dimension_inputs[d], cfg) for d in DIMENSION_ORDER}
    raw_score, weight_sum = compute_raw_score(scores, cfg)
    final_score, cap, fatal_records = apply_fatal_caps(raw_score, fatal_error_keys, cfg)
    decision, reasons = decide_release(
        final_score, fatal_records, scores, unresolved_unverifiable, cfg
    )
    return EvaluationResult(
        question_id=question_id,
        output_id=output_id,
        dimension_scores=scores,
        raw_score=round(raw_score, 4),
        final_score=round(final_score, 4),
        applied_score_cap=cap,
        fatal_errors=fatal_records,
        unresolved_unverifiable=unresolved_unverifiable,
        decision=decision,
        decision_reasons=reasons,
        na_dimensions=[d for d in DIMENSION_ORDER if scores[d].is_na],
        effective_weight_sum=weight_sum,
        evaluator_version=EVALUATOR_VERSION,
        rubric_version=cfg.version,
        rubric_config_sha256=cfg.sha256,
    )


# ---------------------------------------------------------------------------
# 各维连续指标的计算辅助
# ---------------------------------------------------------------------------
# 这些函数只使用配置中的量表数值（主张权重、支持值、复合权重、槽位权重），
# 不引入任何新判据。


def weighted_support_precision(
    claims: Sequence[AtomicClaim],
    verdicts: Mapping[str, JudgeVerdict],
    config: RubricConfig | None = None,
) -> float:
    """D2 加权支持精确率（方案 8.2 D2）。

    完全支持 1、部分支持 0.5、不支持/反驳/未知 0；核心主张权重 2、次要主张权重 1。
    未给出判定的主张按「未知」计 0 分且仍留在分母（方案 10.3：缺失输出记为失败，
    不从分母删除）。
    """
    cfg = config or default_rubric()
    spec = cfg.dimensions["D2"]
    support_values = spec["claim_support_values"]
    claim_weights = spec["claim_weights"]
    numerator = 0.0
    denominator = 0.0
    for claim in claims:
        weight = float(claim_weights["core" if claim.is_core else "secondary"])
        denominator += weight
        verdict = verdicts.get(claim.claim_id)
        if verdict is None:
            continue
        numerator += weight * float(support_values[verdict.verdict.value])
    if denominator == 0:
        raise ValueError("D2 主张集合为空，无法计算加权支持精确率")
    return numerator / denominator


def coverage_composite(
    pooled_evidence_recall: float,
    core_claim_citation_completeness: float,
    config: RubricConfig | None = None,
) -> float:
    """D3 复合指标 C = 0.5×Recall + 0.5×Completeness（方案 8.2 D3）。"""
    cfg = config or default_rubric()
    weights = cfg.dimensions["D3"]["composite"]
    return (
        float(weights["pooled_evidence_recall_weight"]) * pooled_evidence_recall
        + float(weights["core_claim_citation_completeness_weight"])
        * core_claim_citation_completeness
    )


def slot_accuracy(
    slot_results: Mapping[str, bool | None], config: RubricConfig | None = None
) -> float:
    """D4 槽位准确率（方案 8.2 D4）。

    slot_results 的取值：True 正确、False 错误、None 记 NA（不进入分母）。
    物种、细胞类型与效应方向按配置双倍计权。
    """
    cfg = config or default_rubric()
    slot_weights = cfg.dimensions["D4"]["slot_weights"]
    numerator = 0.0
    denominator = 0.0
    for slot, correct in slot_results.items():
        if slot not in slot_weights:
            raise ValueError(f"未知条件槽位 {slot!r}；合法取值：{sorted(slot_weights)}")
        if correct is None:
            continue
        weight = float(slot_weights[slot])
        denominator += weight
        if correct:
            numerator += weight
    if denominator == 0:
        raise ValueError("D4 所有槽位均记 NA，应把该维整体标为 is_na=True")
    return numerator / denominator


def checklist_satisfaction_rate(items: Mapping[str, bool | None]) -> float:
    """D5 适用项满足比例（方案 8.2 D5）。None 表示该项不适用，记 NA。"""
    applicable = {k: v for k, v in items.items() if v is not None}
    if not applicable:
        raise ValueError("D5 适用项为空，应把该维整体标为 is_na=True")
    return sum(1 for v in applicable.values() if v) / len(applicable)
