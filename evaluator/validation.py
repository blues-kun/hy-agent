"""Auditable statistics for validating an open-ended evaluation method.

The functions in this module analyse *evaluation results*; they do not judge
scientific claims.  Every metric has an explicit denominator and returns
``None`` with a structured warning when the available observations are too
few, incomplete, constant, or otherwise mathematically degenerate.

Definitions used here
---------------------

``complete_order_accuracy``
    Among every Cartesian ``good x medium x bad`` triplet, the fraction that
    satisfies ``score_good > score_medium > score_bad``.  A tie is not a
    correct strict ordering.  At least two usable observations per tier are
    required so that a single hand-picked triplet is not presented as a
    validation result.

``kendall_tau_b``
    Kendall's tau-b between the expected tier order (bad=0, medium=1,
    good=2) and evaluator score.  Pairs tied on both variables are omitted;
    ties on only one variable enter the corresponding denominator term.

``linear_weighted_kappa_0_4``
    Cohen's weighted kappa with agreement weight
    ``w(i,j) = 1 - abs(i-j)/4`` over the fixed ordinal labels 0..4.

``icc_2_1``
    Two-way random-effects, absolute-agreement, single-measure ICC(2,1):
    ``(MSR-MSE) / (MSR+(k-1)MSE+k*(MSC-MSE)/n)``.  Here ``k=2`` raters.
    At least three complete subjects are required.  A zero denominator is
    reported as undefined rather than coerced to zero.

``grade_change_rate``
    For one repeatedly evaluated output, ``1 - modal_grade_count / n``.
    The aggregate additionally reports the fraction of outputs that received
    more than one grade at least once.

``severe_attack_success_rate``
    Among pairs explicitly marked ``severe_error=true``, the fraction for
    which ``clean_score - attacked_score < 5``.  This measures score gaming;
    it is intentionally separate from the evaluator's explicit attack flag.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from enum import Enum
from itertools import combinations
from typing import Any, Iterable, Sequence

from pydantic import Field, model_validator

from evaluator.schemas import StrictModel


SCHEMA_VERSION = "mitoevidence.validation.v1"
MIN_TIER_SIZE = 2
MIN_RATE_N = 2
MIN_CORRELATION_N = 3
EPSILON = 1e-12


class QualityTier(str, Enum):
    GOOD = "good"
    MEDIUM = "medium"
    BAD = "bad"


class TieredScore(StrictModel):
    output_id: str
    quality_tier: QualityTier
    score: float | None

    @model_validator(mode="after")
    def _validate_record(self) -> "TieredScore":
        if not self.output_id.strip():
            raise ValueError("output_id 不能为空")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("score 必须是有限数值或 null")
        return self


class NominalRatingPair(StrictModel):
    item_id: str
    rater_a: str | None
    rater_b: str | None


class OrdinalRatingPair(StrictModel):
    item_id: str
    rater_a: int | None = Field(default=None, ge=0, le=4)
    rater_b: int | None = Field(default=None, ge=0, le=4)


class TotalScorePair(StrictModel):
    item_id: str
    rater_a: float | None
    rater_b: float | None

    @model_validator(mode="after")
    def _finite_scores(self) -> "TotalScorePair":
        for value in (self.rater_a, self.rater_b):
            if value is not None and not math.isfinite(value):
                raise ValueError("总分必须是有限数值或 null")
        return self


class AgreementInput(StrictModel):
    nominal: list[NominalRatingPair] = Field(default_factory=list)
    ordinal_0_4: list[OrdinalRatingPair] = Field(default_factory=list)
    total_scores: list[TotalScorePair] = Field(default_factory=list)


class StabilityRepeat(StrictModel):
    run_id: str
    score: float | None
    grade: str | None

    @model_validator(mode="after")
    def _finite_score(self) -> "StabilityRepeat":
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("重复评分必须是有限数值或 null")
        return self


class StabilityRecord(StrictModel):
    output_id: str
    repeats: list[StabilityRepeat] = Field(default_factory=list)


class AdversarialPair(StrictModel):
    pair_id: str
    clean_score: float | None
    attacked_score: float | None
    attack_detected: bool | None
    clean_flagged: bool | None
    severe_error: bool

    @model_validator(mode="after")
    def _finite_scores(self) -> "AdversarialPair":
        for value in (self.clean_score, self.attacked_score):
            if value is not None and not math.isfinite(value):
                raise ValueError("对抗样本分数必须是有限数值或 null")
        return self


class ValidationInput(StrictModel):
    """Explicit JSON input contract consumed by :func:`analyze_validation`."""

    discrimination: list[TieredScore] = Field(default_factory=list)
    agreement: AgreementInput = Field(default_factory=AgreementInput)
    stability: list[StabilityRecord] = Field(default_factory=list)
    adversarial: list[AdversarialPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_within_sections(self) -> "ValidationInput":
        collections = {
            "discrimination.output_id": [x.output_id for x in self.discrimination],
            "agreement.nominal.item_id": [x.item_id for x in self.agreement.nominal],
            "agreement.ordinal_0_4.item_id": [
                x.item_id for x in self.agreement.ordinal_0_4
            ],
            "agreement.total_scores.item_id": [
                x.item_id for x in self.agreement.total_scores
            ],
            "stability.output_id": [x.output_id for x in self.stability],
            "adversarial.pair_id": [x.pair_id for x in self.adversarial],
        }
        for name, ids in collections.items():
            duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
            if duplicates:
                raise ValueError(f"{name} 必须唯一；重复值：{duplicates}")
        return self


def _warning(
    warnings: list[dict[str, Any]],
    code: str,
    section: str,
    message: str,
    item_ids: Iterable[str] = (),
) -> None:
    warnings.append(
        {
            "code": code,
            "section": section,
            "message": message,
            "item_ids": list(item_ids),
        }
    )


def _percentile(values: Sequence[float], probability: float) -> float | None:
    """Type-7/linear percentile, identical to interpolation on (n-1)*p."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return 1-based average ranks with exact-value tie handling."""

    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or not x:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator <= EPSILON:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y, strict=True)) / denominator


def _kendall_tau_b(expected: Sequence[int], observed: Sequence[float]) -> tuple[float | None, dict[str, int]]:
    concordant = discordant = tied_expected_only = tied_observed_only = tied_both = 0
    for i, j in combinations(range(len(expected)), 2):
        dx = expected[i] - expected[j]
        dy = observed[i] - observed[j]
        if dx == 0 and dy == 0:
            tied_both += 1
        elif dx == 0:
            tied_expected_only += 1
        elif dy == 0:
            tied_observed_only += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    left = concordant + discordant + tied_expected_only
    right = concordant + discordant + tied_observed_only
    denominator = math.sqrt(left * right)
    value = None if denominator <= EPSILON else (concordant - discordant) / denominator
    counts = {
        "concordant": concordant,
        "discordant": discordant,
        "tied_expected_only": tied_expected_only,
        "tied_observed_only": tied_observed_only,
        "tied_both": tied_both,
    }
    return value, counts


def _analyse_discrimination(
    records: Sequence[TieredScore], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    missing = [record.output_id for record in records if record.score is None]
    if missing:
        _warning(
            warnings,
            "MISSING_SCORE_EXCLUDED",
            "discrimination",
            "缺失 score 的输出已从判别力统计中排除。",
            missing,
        )
    valid = [record for record in records if record.score is not None]
    groups: dict[QualityTier, list[float]] = {tier: [] for tier in QualityTier}
    for record in valid:
        groups[record.quality_tier].append(float(record.score))

    sizes = {tier.value: len(groups[tier]) for tier in QualityTier}
    complete_accuracy: float | None = None
    correct_triplets: int | None = None
    total_triplets: int | None = None
    if any(len(groups[tier]) < MIN_TIER_SIZE for tier in QualityTier):
        _warning(
            warnings,
            "INSUFFICIENT_TIER_SAMPLE",
            "discrimination.complete_order_accuracy",
            f"每个好/中/差组至少需要 {MIN_TIER_SIZE} 个有效输出；当前 {sizes}。",
        )
    else:
        good = groups[QualityTier.GOOD]
        medium = groups[QualityTier.MEDIUM]
        bad = groups[QualityTier.BAD]
        total_triplets = len(good) * len(medium) * len(bad)
        correct_triplets = sum(
            sum(score_good > score_medium for score_good in good)
            * sum(score_bad < score_medium for score_bad in bad)
            for score_medium in medium
        )
        complete_accuracy = correct_triplets / total_triplets

    tau: float | None = None
    tau_counts = {
        "concordant": 0,
        "discordant": 0,
        "tied_expected_only": 0,
        "tied_observed_only": 0,
        "tied_both": 0,
    }
    if len(valid) < MIN_CORRELATION_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "discrimination.kendall_tau_b",
            f"Kendall tau-b 至少需要 {MIN_CORRELATION_N} 个完整输出。",
        )
    else:
        order = {QualityTier.BAD: 0, QualityTier.MEDIUM: 1, QualityTier.GOOD: 2}
        tau, tau_counts = _kendall_tau_b(
            [order[record.quality_tier] for record in valid],
            [float(record.score) for record in valid],
        )
        if tau is None:
            _warning(
                warnings,
                "CONSTANT_OR_DEGENERATE",
                "discrimination.kendall_tau_b",
                "期望等级或评估分数为常数，tau-b 分母为 0。",
            )

    return {
        "n_input": len(records),
        "n_complete": len(valid),
        "tier_sizes": sizes,
        "complete_order_accuracy": complete_accuracy,
        "correct_strict_triplets": correct_triplets,
        "total_triplets": total_triplets,
        "kendall_tau_b": tau,
        "kendall_pair_counts": tau_counts,
        "definitions": {
            "complete_order_accuracy": "P(score_good > score_medium > score_bad) over all Cartesian triplets; ties fail",
            "kendall_tau_b": "(C-D)/sqrt((C+D+T_expected)*(C+D+T_score))",
        },
    }


def _confusion(labels: Sequence[str], pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    return {
        left: {
            right: sum(1 for a, b in pairs if a == left and b == right)
            for right in labels
        }
        for left in labels
    }


def _nominal_agreement(
    records: Sequence[NominalRatingPair], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    missing = [
        record.item_id
        for record in records
        if record.rater_a is None or record.rater_b is None
    ]
    if missing:
        _warning(
            warnings,
            "INCOMPLETE_PAIR_EXCLUDED",
            "agreement.nominal",
            "任一评审缺失的名义评分对已排除。",
            missing,
        )
    pairs = [
        (str(record.rater_a), str(record.rater_b))
        for record in records
        if record.rater_a is not None and record.rater_b is not None
    ]
    labels = sorted({label for pair in pairs for label in pair})
    matrix = _confusion(labels, pairs)
    raw = None
    kappa = None
    observed = None
    expected = None
    if len(pairs) < MIN_RATE_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "agreement.nominal",
            f"名义一致性至少需要 {MIN_RATE_N} 个完整评分对。",
        )
    else:
        n = len(pairs)
        observed = sum(a == b for a, b in pairs) / n
        raw = observed
        counts_a = Counter(a for a, _ in pairs)
        counts_b = Counter(b for _, b in pairs)
        expected = sum(counts_a[label] * counts_b[label] for label in labels) / (n * n)
        if abs(1.0 - expected) <= EPSILON:
            _warning(
                warnings,
                "CONSTANT_OR_DEGENERATE",
                "agreement.nominal.cohen_kappa",
                "两位评审的边际分布使期望一致率为 1，Cohen kappa 未定义。",
            )
        else:
            kappa = (observed - expected) / (1.0 - expected)
    return {
        "n_input": len(records),
        "n_complete": len(pairs),
        "labels": labels,
        "confusion_matrix": matrix,
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "observed_agreement": observed,
        "expected_agreement": expected,
    }


def _ordinal_agreement(
    records: Sequence[OrdinalRatingPair], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    missing = [
        record.item_id
        for record in records
        if record.rater_a is None or record.rater_b is None
    ]
    if missing:
        _warning(
            warnings,
            "INCOMPLETE_PAIR_EXCLUDED",
            "agreement.ordinal_0_4",
            "任一评审缺失的 0-4 评分对已排除。",
            missing,
        )
    pairs = [
        (int(record.rater_a), int(record.rater_b))
        for record in records
        if record.rater_a is not None and record.rater_b is not None
    ]
    labels = list(range(5))
    matrix = [[sum(a == i and b == j for a, b in pairs) for j in labels] for i in labels]
    raw = None
    weighted_kappa = None
    observed_weighted = None
    expected_weighted = None
    if len(pairs) < MIN_RATE_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "agreement.ordinal_0_4",
            f"线性加权 kappa 至少需要 {MIN_RATE_N} 个完整评分对。",
        )
    else:
        n = len(pairs)
        raw = sum(a == b for a, b in pairs) / n
        weights = [[1.0 - abs(i - j) / 4.0 for j in labels] for i in labels]
        observed_weighted = sum(weights[a][b] for a, b in pairs) / n
        counts_a = Counter(a for a, _ in pairs)
        counts_b = Counter(b for _, b in pairs)
        expected_weighted = sum(
            weights[i][j] * counts_a[i] * counts_b[j]
            for i in labels
            for j in labels
        ) / (n * n)
        if abs(1.0 - expected_weighted) <= EPSILON:
            _warning(
                warnings,
                "CONSTANT_OR_DEGENERATE",
                "agreement.ordinal_0_4.linear_weighted_kappa",
                "加权期望一致率为 1，线性加权 kappa 未定义。",
            )
        else:
            weighted_kappa = (observed_weighted - expected_weighted) / (
                1.0 - expected_weighted
            )
    return {
        "n_input": len(records),
        "n_complete": len(pairs),
        "labels": labels,
        "confusion_matrix": matrix,
        "raw_agreement": raw,
        "linear_weighted_kappa": weighted_kappa,
        "observed_weighted_agreement": observed_weighted,
        "expected_weighted_agreement": expected_weighted,
        "weight_formula": "1 - abs(rater_a-rater_b)/4",
    }


def _icc_2_1(values: Sequence[tuple[float, float]]) -> tuple[float | None, dict[str, float | int | None]]:
    n = len(values)
    k = 2
    flat = [score for pair in values for score in pair]
    grand = statistics.fmean(flat)
    row_means = [statistics.fmean(pair) for pair in values]
    column_means = [statistics.fmean(pair[index] for pair in values) for index in range(k)]
    ss_rows = k * sum((value - grand) ** 2 for value in row_means)
    ss_columns = n * sum((value - grand) ** 2 for value in column_means)
    ss_error = sum(
        (values[row][column] - row_means[row] - column_means[column] + grand) ** 2
        for row in range(n)
        for column in range(k)
    )
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + k * (ms_columns - ms_error) / n
    )
    value = None if abs(denominator) <= EPSILON else (ms_rows - ms_error) / denominator
    audit: dict[str, float | int | None] = {
        "n_subjects": n,
        "n_raters": k,
        "ms_rows": ms_rows,
        "ms_columns": ms_columns,
        "ms_error": ms_error,
        "denominator": denominator,
    }
    return value, audit


def _total_score_agreement(
    records: Sequence[TotalScorePair], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    missing = [
        record.item_id
        for record in records
        if record.rater_a is None or record.rater_b is None
    ]
    if missing:
        _warning(
            warnings,
            "INCOMPLETE_PAIR_EXCLUDED",
            "agreement.total_scores",
            "任一评审缺失的总分对已排除。",
            missing,
        )
    pairs = [
        (float(record.rater_a), float(record.rater_b))
        for record in records
        if record.rater_a is not None and record.rater_b is not None
    ]
    mae = None
    if len(pairs) < MIN_RATE_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "agreement.total_scores.mae",
            f"MAE 至少需要 {MIN_RATE_N} 个完整评分对。",
        )
    else:
        mae = statistics.fmean(abs(a - b) for a, b in pairs)

    spearman = None
    if len(pairs) < MIN_CORRELATION_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "agreement.total_scores.spearman",
            f"Spearman 至少需要 {MIN_CORRELATION_N} 个完整评分对。",
        )
    else:
        ranks_a = _average_ranks([pair[0] for pair in pairs])
        ranks_b = _average_ranks([pair[1] for pair in pairs])
        spearman = _pearson(ranks_a, ranks_b)
        if spearman is None:
            _warning(
                warnings,
                "CONSTANT_OR_DEGENERATE",
                "agreement.total_scores.spearman",
                "至少一位评审的总分为常数，Spearman 未定义。",
            )

    icc = None
    icc_audit: dict[str, float | int | None] = {
        "n_subjects": len(pairs),
        "n_raters": 2,
        "ms_rows": None,
        "ms_columns": None,
        "ms_error": None,
        "denominator": None,
    }
    if len(pairs) < MIN_CORRELATION_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "agreement.total_scores.icc_2_1",
            f"ICC(2,1) 至少需要 {MIN_CORRELATION_N} 个完整对象。",
        )
    else:
        icc, icc_audit = _icc_2_1(pairs)
        if icc is None:
            _warning(
                warnings,
                "CONSTANT_OR_DEGENERATE",
                "agreement.total_scores.icc_2_1",
                "ICC(2,1) 分母为 0，通常由完全常数评分造成。",
            )

    return {
        "n_input": len(records),
        "n_complete": len(pairs),
        "spearman_rho": spearman,
        "mean_absolute_error": mae,
        "icc_2_1": icc,
        "icc_audit": icc_audit,
        "icc_formula": "(MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n); two-way random, absolute agreement, single measure",
    }


def _analyse_agreement(
    data: AgreementInput, warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "nominal": _nominal_agreement(data.nominal, warnings),
        "ordinal_0_4": _ordinal_agreement(data.ordinal_0_4, warnings),
        "total_scores": _total_score_agreement(data.total_scores, warnings),
    }


def _analyse_stability(
    records: Sequence[StabilityRecord], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    if not records:
        _warning(
            warnings,
            "MISSING_SECTION_DATA",
            "stability",
            "未提供重复评估记录，稳定性指标均为 null。",
        )
    output_rows: list[dict[str, Any]] = []
    usable_stds: list[float] = []
    outputs_with_grade_change = 0
    outputs_with_grade_data = 0
    for record in records:
        missing_scores = [repeat.run_id for repeat in record.repeats if repeat.score is None]
        if missing_scores:
            _warning(
                warnings,
                "MISSING_SCORE_EXCLUDED",
                f"stability.{record.output_id}",
                "缺失 score 的重复运行已从分数稳定性统计排除。",
                missing_scores,
            )
        scores = [float(repeat.score) for repeat in record.repeats if repeat.score is not None]
        grades = [
            str(repeat.grade)
            for repeat in record.repeats
            if repeat.grade is not None and str(repeat.grade).strip()
        ]
        score_std = score_median = score_p95 = None
        if len(scores) < MIN_RATE_N:
            _warning(
                warnings,
                "INSUFFICIENT_REPEATS",
                f"stability.{record.output_id}.scores",
                f"每个输出至少需要 {MIN_RATE_N} 次有效分数。",
            )
        else:
            score_std = statistics.stdev(scores)
            score_median = statistics.median(scores)
            score_p95 = _percentile(scores, 0.95)
            usable_stds.append(score_std)

        grade_change_rate = None
        grade_instability = None
        if len(grades) < MIN_RATE_N:
            _warning(
                warnings,
                "INSUFFICIENT_REPEATS",
                f"stability.{record.output_id}.grades",
                f"等级变化率至少需要 {MIN_RATE_N} 次有效等级。",
            )
        else:
            counts = Counter(grades)
            grade_change_rate = 1.0 - max(counts.values()) / len(grades)
            grade_instability = len(counts) > 1
            outputs_with_grade_data += 1
            outputs_with_grade_change += int(grade_instability)

        output_rows.append(
            {
                "output_id": record.output_id,
                "n_runs_input": len(record.repeats),
                "n_scores": len(scores),
                "n_grades": len(grades),
                "score_std_sample": score_std,
                "score_median": score_median,
                "score_p95_type7": score_p95,
                "grade_change_rate": grade_change_rate,
                "grade_instability": grade_instability,
            }
        )

    aggregate_std_median = None
    aggregate_std_p95 = None
    if len(usable_stds) >= MIN_RATE_N:
        aggregate_std_median = statistics.median(usable_stds)
        aggregate_std_p95 = _percentile(usable_stds, 0.95)
    elif usable_stds:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "stability.aggregate.score_std",
            f"聚合输出标准差至少需要 {MIN_RATE_N} 个可用输出。",
        )
    output_grade_change_rate = None
    if outputs_with_grade_data >= MIN_RATE_N:
        output_grade_change_rate = outputs_with_grade_change / outputs_with_grade_data
    elif records:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "stability.aggregate.output_grade_change_rate",
            f"聚合等级变化率至少需要 {MIN_RATE_N} 个具有完整等级重复的输出。",
        )

    return {
        "n_outputs": len(records),
        "outputs": output_rows,
        "aggregate": {
            "n_outputs_with_score_std": len(usable_stds),
            "median_of_output_stds": aggregate_std_median,
            "p95_of_output_stds_type7": aggregate_std_p95,
            "n_outputs_with_grade_data": outputs_with_grade_data,
            "outputs_with_any_grade_change": outputs_with_grade_change,
            "output_grade_change_rate": output_grade_change_rate,
        },
        "definitions": {
            "score_std_sample": "sample standard deviation (n-1 denominator)",
            "grade_change_rate": "1 - modal grade count / number of valid grades",
            "output_grade_change_rate": "outputs with >1 observed grade / outputs with valid grade repeats",
        },
    }


def _analyse_adversarial(
    records: Sequence[AdversarialPair], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    detection = [record.attack_detected for record in records if record.attack_detected is not None]
    false_positive = [record.clean_flagged for record in records if record.clean_flagged is not None]
    score_rows: list[dict[str, Any]] = []
    drops: list[float] = []
    severe_drops: list[float] = []
    missing_score_ids: list[str] = []
    for record in records:
        if record.clean_score is None or record.attacked_score is None:
            missing_score_ids.append(record.pair_id)
            drop = None
        else:
            drop = float(record.clean_score - record.attacked_score)
            drops.append(drop)
            if record.severe_error:
                severe_drops.append(drop)
        score_rows.append(
            {
                "pair_id": record.pair_id,
                "clean_score": record.clean_score,
                "attacked_score": record.attacked_score,
                "score_drop_clean_minus_attacked": drop,
                "severe_error": record.severe_error,
                "attack_detected": record.attack_detected,
                "clean_flagged": record.clean_flagged,
            }
        )
    if missing_score_ids:
        _warning(
            warnings,
            "INCOMPLETE_PAIR_EXCLUDED",
            "adversarial.score_drop",
            "缺失 clean_score 或 attacked_score 的对抗对已从分差统计排除。",
            missing_score_ids,
        )

    def rate_or_none(values: Sequence[bool], section: str, label: str) -> float | None:
        if len(values) < MIN_RATE_N:
            _warning(
                warnings,
                "INSUFFICIENT_SAMPLE",
                section,
                f"{label}至少需要 {MIN_RATE_N} 个明确布尔判定。",
            )
            return None
        return sum(values) / len(values)

    detection_rate = rate_or_none(
        detection, "adversarial.attack_detection_rate", "攻击检出率"
    )
    false_positive_rate = rate_or_none(
        false_positive, "adversarial.clean_false_positive_rate", "干净样本误报率"
    )

    severe_success = None
    if len(severe_drops) < MIN_RATE_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "adversarial.severe_attack_success_rate",
            f"严重错误攻击成功率至少需要 {MIN_RATE_N} 个带完整分数的严重错误对。",
        )
    else:
        severe_success = sum(drop < 5.0 for drop in severe_drops) / len(severe_drops)

    drop_summary = {
        "n": len(drops),
        "mean": statistics.fmean(drops) if drops else None,
        "median": statistics.median(drops) if drops else None,
        "p05_type7": _percentile(drops, 0.05),
        "p95_type7": _percentile(drops, 0.95),
        "minimum": min(drops) if drops else None,
        "maximum": max(drops) if drops else None,
    }
    if len(drops) < MIN_RATE_N:
        _warning(
            warnings,
            "INSUFFICIENT_SAMPLE",
            "adversarial.score_drop",
            f"稳健解释分差至少需要 {MIN_RATE_N} 个完整对；描述统计保留为 null/有限样本。",
        )
        if len(drops) == 1:
            # A single value is not presented as an aggregate validation result.
            drop_summary = {key: (1 if key == "n" else None) for key in drop_summary}

    return {
        "n_pairs": len(records),
        "attack_detection": {
            "numerator": sum(detection) if detection else 0,
            "denominator": len(detection),
            "rate": detection_rate,
        },
        "clean_false_positive": {
            "numerator": sum(false_positive) if false_positive else 0,
            "denominator": len(false_positive),
            "rate": false_positive_rate,
        },
        "score_drop_clean_minus_attacked": drop_summary,
        "severe_attack_success": {
            "criterion": "severe_error=true and clean_score-attacked_score < 5",
            "numerator": sum(drop < 5.0 for drop in severe_drops),
            "denominator": len(severe_drops),
            "rate": severe_success,
        },
        "pairs": score_rows,
    }


def analyze_validation(data: ValidationInput) -> dict[str, Any]:
    """Compute all validity analyses without network access or hidden state."""

    warnings: list[dict[str, Any]] = []
    result = {
        "schema_version": SCHEMA_VERSION,
        "methodology": {
            "purpose": "descriptive validation of an evaluation method",
            "minimum_tier_size": MIN_TIER_SIZE,
            "minimum_rate_n": MIN_RATE_N,
            "minimum_correlation_n": MIN_CORRELATION_N,
            "missing_policy": "pairwise/section-wise exclusion with structured warnings; no imputation",
        },
        "discrimination": _analyse_discrimination(data.discrimination, warnings),
        "agreement": _analyse_agreement(data.agreement, warnings),
        "stability": _analyse_stability(data.stability, warnings),
        "adversarial": _analyse_adversarial(data.adversarial, warnings),
        "warnings": warnings,
    }
    return result
