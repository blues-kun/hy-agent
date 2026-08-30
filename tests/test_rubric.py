"""计分数学的边界测试（方案 8.2 / 8.3）。

约定：全部维度都给出时 Σw = 100，于是
    RawScore = Σ_d w_d * s_d / 4
本文件用到的四个「整分」构型（全部满足 D1/D2/D4/D6 ≥ 3 的 PASS 门槛）：
    基线全 4 分                                    → 100
    D3=0                                           → 85
    D5=2, D7=1, D8=2                               → 84
    D3=0, D5=0                                     → 70
    D3=0, D5=2, D7=1, D8=2                         → 69
"""
from __future__ import annotations

import pytest

from evaluator.rubric import (
    DIMENSION_ORDER,
    DimensionInput,
    apply_fatal_caps,
    band_for,
    checklist_satisfaction_rate,
    compute_raw_score,
    coverage_composite,
    decide_release,
    default_rubric,
    evaluate,
    score_dimension,
    slot_accuracy,
    weighted_support_precision,
)
from evaluator.schemas import (
    AtomicClaim,
    DimensionScore,
    JudgeVerdict,
    ReleaseDecision,
    SupportVerdict,
)

CFG = default_rubric()


def levels(**overrides: int | None) -> dict[str, DimensionInput]:
    """构造九维输入：默认全 4 分，overrides 中值为 None 表示该维记 NA。"""
    out: dict[str, DimensionInput] = {}
    for dim in DIMENSION_ORDER:
        if dim in overrides:
            value = overrides[dim]
            out[dim] = DimensionInput(is_na=True) if value is None else DimensionInput(level=value)
        else:
            out[dim] = DimensionInput(level=4)
    return out


# ---------------------------------------------------------------------------
# 配置自洽
# ---------------------------------------------------------------------------


def test_weights_sum_to_100():
    assert sum(CFG.weight(d) for d in DIMENSION_ORDER) == 100
    assert int(CFG.scale["total_weight"]) == 100


def test_weights_match_proposal_8_2():
    expected = {"D1": 10, "D2": 20, "D3": 15, "D4": 12, "D5": 15, "D6": 10, "D7": 8, "D8": 5, "D9": 5}
    assert {d: CFG.weight(d) for d in DIMENSION_ORDER} == expected


def test_every_dimension_has_five_bands_or_direct_levels():
    for dim in DIMENSION_ORDER:
        bands = CFG.dimensions[dim].get("bands")
        if bands:
            assert [b["level"] for b in bands] == [4, 3, 2, 1, 0]


def test_bad_config_weight_sum_is_rejected(tmp_path):
    import yaml

    from evaluator.rubric import RubricConfigError, load_rubric

    raw = yaml.safe_load(CFG.source_path.read_text(encoding="utf-8"))
    raw["dimensions"]["D9"]["weight"] = 6  # 合计 101
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(RubricConfigError, match="权重之和"):
        load_rubric(broken)


# ---------------------------------------------------------------------------
# 加权折算与 NA 重归一
# ---------------------------------------------------------------------------


def test_all_fours_is_100():
    assert evaluate("q", levels()).raw_score == 100.0


def test_all_zeros_is_0():
    assert evaluate("q", levels(**{d: 0 for d in DIMENSION_ORDER})).raw_score == 0.0


def test_all_twos_is_50():
    assert evaluate("q", levels(**{d: 2 for d in DIMENSION_ORDER})).raw_score == 50.0


def test_na_removes_weight_from_denominator():
    """D1=4、其余全 0：无 NA 时 10/100；D2(20) 记 NA 后分母降为 80。"""
    base = {d: 0 for d in DIMENSION_ORDER}
    base["D1"] = 4
    assert evaluate("q", levels(**base)).raw_score == 10.0

    with_na = dict(base)
    with_na["D2"] = None
    result = evaluate("q", levels(**with_na))
    assert result.effective_weight_sum == 80
    assert result.raw_score == pytest.approx(12.5)
    assert result.na_dimensions == ["D2"]


def test_na_preserves_full_score():
    result = evaluate("q", levels(D3=None, D9=None))
    assert result.raw_score == 100.0
    assert result.effective_weight_sum == 100 - 15 - 5


def unrounded_raw(**overrides: int | None) -> float:
    """不经 EvaluationResult 的四位小数取整，直接取加权折算原值。"""
    inputs = levels(**overrides)
    scores = {d: score_dimension(d, inputs[d]) for d in DIMENSION_ORDER}
    return compute_raw_score(scores)[0]


def test_na_renormalization_keeps_weight_ratios():
    """D3 记 NA 后，D2 与 D4 的相对贡献比仍是原权重比 20:12。"""
    loss_from_d4 = 100.0 - unrounded_raw(D3=None, D4=0)
    loss_from_d2 = 100.0 - unrounded_raw(D3=None, D2=0)
    assert loss_from_d2 / loss_from_d4 == pytest.approx(20 / 12)


def test_na_renormalization_matches_closed_form():
    """D3(15) 与 D7(8) 记 NA、D2 降为 0：raw = 100 × (77 − 20) / 77。"""
    assert unrounded_raw(D3=None, D7=None, D2=0) == pytest.approx(100.0 * (77 - 20) / 77)


def test_all_dimensions_na_raises():
    with pytest.raises(ValueError, match="全部记 NA"):
        evaluate("q", levels(**{d: None for d in DIMENSION_ORDER}))


def test_missing_dimension_is_an_error_not_silent_na():
    incomplete = levels()
    del incomplete["D5"]
    with pytest.raises(ValueError, match="缺少：\\['D5'\\]"):
        evaluate("q", incomplete)


def test_unknown_dimension_rejected():
    bad = levels()
    bad["D10"] = DimensionInput(level=4)
    with pytest.raises(ValueError, match="未知维度"):
        evaluate("q", bad)


def test_compute_raw_score_returns_effective_weight():
    scores = {
        d: DimensionScore(dimension=d, name_zh=CFG.name_zh(d), weight=CFG.weight(d), level=4)
        for d in DIMENSION_ORDER
    }
    scores["D7"] = DimensionScore(
        dimension="D7", name_zh=CFG.name_zh("D7"), weight=8, is_na=True
    )
    raw, weight_sum = compute_raw_score(scores)
    assert (raw, weight_sum) == (100.0, 92)


# ---------------------------------------------------------------------------
# 指标 → 档位
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric,expected",
    [(1.0, 4), (0.999, 3), (0.98, 3), (0.9799, 2), (0.95, 2), (0.9499, 1), (0.80, 1), (0.7999, 0), (0.0, 0)],
)
def test_d1_bands(metric, expected):
    assert band_for("D1", metric) == expected


@pytest.mark.parametrize(
    "metric,expected",
    [(1.0, 4), (0.90, 4), (0.8999, 3), (0.80, 3), (0.7999, 2), (0.65, 2), (0.6499, 1), (0.40, 1), (0.3999, 0)],
)
def test_d2_bands(metric, expected):
    assert band_for("D2", metric) == expected


@pytest.mark.parametrize("metric,expected", [(0.85, 4), (0.70, 3), (0.50, 2), (0.30, 1), (0.2999, 0)])
def test_d3_bands(metric, expected):
    assert band_for("D3", metric) == expected


@pytest.mark.parametrize("metric,expected", [(0.95, 4), (0.85, 3), (0.70, 2), (0.50, 1), (0.4999, 0)])
def test_d4_bands(metric, expected):
    assert band_for("D4", metric) == expected


@pytest.mark.parametrize(
    "metric,expected", [(1.0, 4), (0.9999, 3), (0.80, 3), (0.50, 2), (0.4999, 1), (1e-6, 1), (0.0, 0)]
)
def test_d5_bands_treat_zero_as_level_zero(metric, expected):
    """方案 8.2 D5：1 分档为 0<p<0.50，严格大于 0；p=0 落 0 分。"""
    assert band_for("D5", metric) == expected


@pytest.mark.parametrize("metric,expected", [(0.95, 4), (0.90, 3), (0.75, 2), (0.50, 1), (0.4999, 0)])
def test_d8_bands(metric, expected):
    assert band_for("D8", metric) == expected


@pytest.mark.parametrize("count,expected", [(6, 4), (5, 3), (4, 2), (3, 2), (2, 1), (1, 1), (0, 0)])
def test_d7_and_d9_count_bands(count, expected):
    assert band_for("D7", count) == expected
    assert band_for("D9", count) == expected


def test_ratio_thresholds_are_exact_on_fractional_input():
    """49/50 = 0.98 必须落 3 分档，19/20 = 0.95 必须落 2 分档。"""
    assert band_for("D1", 49 / 50) == 3
    assert band_for("D1", 19 / 20) == 2


# ---------------------------------------------------------------------------
# 事件上限
# ---------------------------------------------------------------------------


def test_d1_event_cap_core_conclusion_wrong_citation():
    score = score_dimension(
        "D1", DimensionInput(metric_value=1.0, event_flags={"core_conclusion_uses_wrong_citation": True})
    )
    assert (score.level_before_event_caps, score.level) == (4, 1)
    assert score.event_caps_applied == ["core_conclusion_uses_wrong_citation"]


@pytest.mark.parametrize("count,expected_level", [(0, 4), (1, 4), (2, 0), (5, 0)])
def test_d1_event_cap_nonexistent_identifiers(count, expected_level):
    score = score_dimension(
        "D1", DimensionInput(metric_value=1.0, event_flags={"nonexistent_identifier_count": count})
    )
    assert score.level == expected_level


def test_d2_direction_reversal_caps_at_three():
    score = score_dimension(
        "D2", DimensionInput(metric_value=0.99, event_flags={"core_direction_reversal": True})
    )
    assert (score.level_before_event_caps, score.level) == (4, 3)


def test_d2_majority_contradicted_caps_at_zero():
    score = score_dimension(
        "D2", DimensionInput(metric_value=0.99, event_flags={"majority_core_claims_contradicted": True})
    )
    assert score.level == 0


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"key_slot_error": True}, 3),
        ({"key_condition_swap_count": 1}, 1),
        ({"species_cell_direction_confusion_count": 1}, 4),
        ({"species_cell_direction_confusion_count": 2}, 0),
        ({"key_slot_error": True, "key_condition_swap_count": 1}, 1),
    ],
)
def test_d4_event_caps(flags, expected):
    assert score_dimension("D4", DimensionInput(metric_value=1.0, event_flags=flags)).level == expected


def test_d8_magnitude_error_caps_at_zero():
    score = score_dimension(
        "D8", DimensionInput(metric_value=1.0, event_flags={"magnitude_or_unit_error": True})
    )
    assert score.level == 0


def test_event_cap_never_raises_level():
    """上限只能压低档位，不能抬高。"""
    score = score_dimension(
        "D1", DimensionInput(metric_value=0.0, event_flags={"core_conclusion_uses_wrong_citation": True})
    )
    assert score.level == 0


def test_numeric_flag_passed_as_bool_is_rejected():
    with pytest.raises(ValueError, match="when_at_least"):
        score_dimension(
            "D1", DimensionInput(metric_value=1.0, event_flags={"nonexistent_identifier_count": True})
        )


def test_dimension_input_requires_exactly_one_source():
    with pytest.raises(ValueError):
        DimensionInput()
    with pytest.raises(ValueError):
        DimensionInput(level=3, metric_value=0.9)
    with pytest.raises(ValueError):
        DimensionInput(is_na=True, level=3)


def test_na_dimension_keeps_metric_name_but_no_level():
    score = score_dimension("D3", DimensionInput(is_na=True))
    assert score.is_na and score.level is None and score.metric_name == "coverage_composite_C"


# ---------------------------------------------------------------------------
# 致命错误上限
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,cap",
    [
        ("forged_citation_in_core_conclusion", 59),
        ("majority_core_claims_unlocatable", 49),
        ("core_species_swap_or_direction_reversal", 69),
        ("individualized_clinical_decision", 59),
    ],
)
def test_each_fatal_cap(key, cap):
    result = evaluate("q", levels(), fatal_error_keys=[key])
    assert result.raw_score == 100.0
    assert result.final_score == float(cap)
    assert result.applied_score_cap == cap
    assert [r.key for r in result.fatal_errors] == [key]
    assert result.decision is ReleaseDecision.REJECT


def test_multiple_fatal_caps_take_the_lowest():
    result = evaluate(
        "q",
        levels(),
        fatal_error_keys=[
            "core_species_swap_or_direction_reversal",  # 69
            "forged_citation_in_core_conclusion",  # 59
            "majority_core_claims_unlocatable",  # 49
        ],
    )
    assert result.final_score == 49.0
    assert result.applied_score_cap == 49
    assert len(result.fatal_errors) == 3


def test_cap_never_raises_a_low_score():
    """上限是 min()，不是赋值：raw 40 遇到 69 分上限仍为 40。"""
    raw = evaluate("q", levels(**{d: 0 for d in DIMENSION_ORDER}), fatal_error_keys=[]).raw_score
    assert raw == 0.0
    final, cap, _ = apply_fatal_caps(40.0, ["core_species_swap_or_direction_reversal"])
    assert (final, cap) == (40.0, 69)


def test_no_fatal_error_leaves_score_untouched():
    final, cap, records = apply_fatal_caps(93.7, [])
    assert (final, cap, records) == (93.7, None, [])


def test_unknown_fatal_error_key_rejected():
    with pytest.raises(ValueError, match="未知致命错误类型"):
        apply_fatal_caps(100.0, ["typo_error"])


# ---------------------------------------------------------------------------
# 发布决策
# ---------------------------------------------------------------------------


def _scores(**overrides: int) -> dict[str, DimensionScore]:
    return {
        d: DimensionScore(
            dimension=d, name_zh=CFG.name_zh(d), weight=CFG.weight(d), level=overrides.get(d, 4)
        )
        for d in DIMENSION_ORDER
    }


@pytest.mark.parametrize(
    "score,expected",
    [
        (100.0, ReleaseDecision.PASS),
        (85.0, ReleaseDecision.PASS),
        (84.999, ReleaseDecision.REVIEW),
        (84.0, ReleaseDecision.REVIEW),
        (70.0, ReleaseDecision.REVIEW),
        (69.999, ReleaseDecision.REJECT),
        (69.0, ReleaseDecision.REJECT),
        (0.0, ReleaseDecision.REJECT),
    ],
)
def test_decision_score_boundaries(score, expected):
    decision, _ = decide_release(score, [], _scores())
    assert decision is expected


def test_end_to_end_85_passes_and_84_reviews():
    """由维度档位真正算出 85 与 84，验证边界不是只在决策函数里成立。"""
    at_85 = evaluate("q", levels(D3=0))
    assert at_85.raw_score == 85.0
    assert at_85.decision is ReleaseDecision.PASS

    at_84 = evaluate("q", levels(D5=2, D7=1, D8=2))
    assert at_84.raw_score == 84.0
    assert at_84.decision is ReleaseDecision.REVIEW


def test_end_to_end_70_reviews_and_69_rejects():
    at_70 = evaluate("q", levels(D3=0, D5=0))
    assert at_70.raw_score == 70.0
    assert at_70.decision is ReleaseDecision.REVIEW

    at_69 = evaluate("q", levels(D3=0, D5=2, D7=1, D8=2))
    assert at_69.raw_score == 69.0
    assert at_69.decision is ReleaseDecision.REJECT


@pytest.mark.parametrize("gate_dim", ["D1", "D2", "D4", "D6"])
def test_gate_dimension_below_three_blocks_pass(gate_dim):
    result = evaluate("q", levels(**{gate_dim: 2}))
    assert result.raw_score >= 85.0
    assert result.decision is not ReleaseDecision.PASS
    assert result.decision is ReleaseDecision.REVIEW
    assert any("关键维度未达" in r for r in result.decision_reasons)


def test_gate_dimension_at_exactly_three_still_passes():
    result = evaluate("q", levels(D1=3, D2=3))
    assert result.raw_score >= 85.0
    assert result.decision is ReleaseDecision.PASS


def test_non_gate_dimension_below_three_does_not_block_pass():
    result = evaluate("q", levels(D9=0))
    assert result.decision is ReleaseDecision.PASS


def test_gate_dimension_na_blocks_pass():
    """关键维度记 NA 时无法确认「不低于 3 分」，不得判 PASS。"""
    result = evaluate("q", levels(D6=None))
    assert result.raw_score == 100.0
    assert result.decision is ReleaseDecision.REVIEW


def test_unresolved_unverifiable_forces_review_even_at_100():
    result = evaluate("q", levels(), unresolved_unverifiable=True)
    assert result.raw_score == 100.0
    assert result.decision is ReleaseDecision.REVIEW
    assert result.decision_reasons == ["存在未解决的「不可核验」项"]


def test_fatal_error_beats_unverifiable_and_high_score():
    result = evaluate(
        "q", levels(), fatal_error_keys=["individualized_clinical_decision"], unresolved_unverifiable=True
    )
    assert result.decision is ReleaseDecision.REJECT


def test_unverifiable_does_not_rescue_a_sub_70_score():
    result = evaluate("q", levels(D3=0, D5=2, D7=1, D8=2), unresolved_unverifiable=True)
    assert result.raw_score == 69.0
    assert result.decision is ReleaseDecision.REJECT


# ---------------------------------------------------------------------------
# 结果记录完整性
# ---------------------------------------------------------------------------


def test_result_keeps_continuous_metric_and_level_and_decision():
    """方案 8.3 末段：同时保留原始连续指标、0—4 档位、致命错误类型和发布决策。"""
    inputs = levels()
    inputs["D2"] = DimensionInput(metric_value=0.83)
    result = evaluate(
        "q7", inputs, fatal_error_keys=["forged_citation_in_core_conclusion"], output_id="out-042"
    )
    d2 = result.dimension_scores["D2"]
    assert (d2.metric_value, d2.metric_name, d2.level) == (0.83, "weighted_support_precision", 3)
    assert result.output_id == "out-042"
    assert result.fatal_errors[0].label_zh == "核心结论使用伪造引用"
    assert result.decision is ReleaseDecision.REJECT
    assert result.evaluator_version.startswith("mitoevidence-evaluator-")
    assert result.rubric_version == "0.1"
    assert len(result.rubric_config_sha256) == 64


# ---------------------------------------------------------------------------
# 各维指标计算辅助
# ---------------------------------------------------------------------------


def _claim(cid: str, core: bool) -> AtomicClaim:
    return AtomicClaim(claim_id=cid, text=f"claim {cid}", is_core=core)


def _verdict(cid: str, verdict: SupportVerdict) -> JudgeVerdict:
    refs = [] if verdict in (SupportVerdict.UNKNOWN, SupportVerdict.NOT_SUPPORTED) else ["s1"]
    return JudgeVerdict(claim_id=cid, verdict=verdict, confidence=0.9, reason="t", evidence_span_refs=refs)


def test_weighted_support_precision_uses_core_double_weight():
    claims = [_claim("c1", True), _claim("c2", True), _claim("c3", False)]
    verdicts = {
        "c1": _verdict("c1", SupportVerdict.FULLY_SUPPORTED),  # 2 * 1.0
        "c2": _verdict("c2", SupportVerdict.PARTIALLY_SUPPORTED),  # 2 * 0.5
        "c3": _verdict("c3", SupportVerdict.NOT_SUPPORTED),  # 1 * 0.0
    }
    assert weighted_support_precision(claims, verdicts) == pytest.approx(3.0 / 5.0)


def test_refuted_and_unknown_score_zero():
    claims = [_claim("c1", False), _claim("c2", False)]
    verdicts = {
        "c1": _verdict("c1", SupportVerdict.REFUTED),
        "c2": _verdict("c2", SupportVerdict.UNKNOWN),
    }
    assert weighted_support_precision(claims, verdicts) == 0.0


def test_missing_verdict_stays_in_denominator():
    """方案 10.3：缺失输出记为失败，不从分母删除。"""
    claims = [_claim("c1", False), _claim("c2", False)]
    verdicts = {"c1": _verdict("c1", SupportVerdict.FULLY_SUPPORTED)}
    assert weighted_support_precision(claims, verdicts) == 0.5


def test_coverage_composite_is_half_half():
    assert coverage_composite(0.8, 0.6) == pytest.approx(0.7)
    assert coverage_composite(1.0, 0.0) == pytest.approx(0.5)


def test_slot_accuracy_double_weights_key_slots():
    result = slot_accuracy(
        {
            "species": True,  # 2
            "cell_type": False,  # 2
            "effect_direction": True,  # 2
            "method": True,  # 1
            "dose": None,  # NA，不进分母
        }
    )
    assert result == pytest.approx(5 / 7)


def test_slot_accuracy_rejects_unknown_slot():
    with pytest.raises(ValueError, match="未知条件槽位"):
        slot_accuracy({"temperature": True})


def test_slot_accuracy_all_na_raises():
    with pytest.raises(ValueError, match="应把该维整体标为"):
        slot_accuracy({"species": None, "dose": None})


def test_checklist_rate_ignores_na_items():
    assert checklist_satisfaction_rate({"a": True, "b": False, "c": None}) == 0.5
    assert checklist_satisfaction_rate({"a": True, "b": None}) == 1.0


def test_checklist_all_na_raises():
    with pytest.raises(ValueError, match="应把该维整体标为"):
        checklist_satisfaction_rate({"a": None})
