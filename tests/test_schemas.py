"""数据契约测试（方案 8.1 冻结定义 + 9.2 金标 Schema + 核验报告的文本锚点结论）。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluator.schemas import (
    Answerability,
    AtomicClaim,
    Citation,
    ConditionSlot,
    DimensionScore,
    EffectDirection,
    EvidencePaper,
    EvidenceSpan,
    ExperimentConditions,
    IdentifierVerification,
    JudgeVerdict,
    QuestionGold,
    SourceAccess,
    SupportVerdict,
    TextAnchor,
    VerificationStatus,
)


def anchor(exact: str = "线粒体碎片化指数显著升高") -> TextAnchor:
    return TextAnchor(prefix="我们观察到 INS-1E 细胞", exact=exact, postfix="（p<0.01）。")


def span(span_id: str = "s1", access: SourceAccess = SourceAccess.FULLTEXT) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        paper_id="p1",
        doi_or_pmid="10.1038/s41746-025-02005-2",
        section="Results",
        page_or_figure="Fig. 3B",
        anchor=anchor(),
        source_access=access,
    )


# ---------------------------------------------------------------------------
# 文本锚点（不使用字符 offset）
# ---------------------------------------------------------------------------


def test_evidence_span_uses_text_anchor_not_char_offset():
    """核验报告 3.1：Europe PMC Annotations API 无字符 offset，定位只能用文本锚点。"""
    fields = set(EvidenceSpan.model_fields)
    assert "anchor" in fields
    assert not fields & {"start_offset", "end_offset", "char_offset", "offset"}
    assert set(TextAnchor.model_fields) == {"prefix", "exact", "postfix"}


def test_anchor_requires_non_blank_exact():
    with pytest.raises(ValidationError, match="不能为空白"):
        TextAnchor(exact="   ")


def test_anchor_prefix_and_postfix_default_to_empty():
    a = TextAnchor(exact="Drp1 敲低减少碎片化")
    assert (a.prefix, a.postfix) == ("", "")


# ---------------------------------------------------------------------------
# source_access 规则（方案 8.1）
# ---------------------------------------------------------------------------


def test_fulltext_span_requires_anchor():
    with pytest.raises(ValidationError, match="必须提供 anchor"):
        EvidenceSpan(
            span_id="s1", paper_id="p1", doi_or_pmid="10.1/x", source_access=SourceAccess.FULLTEXT
        )


def test_abstract_only_span_requires_anchor():
    with pytest.raises(ValidationError, match="必须提供 anchor"):
        EvidenceSpan(
            span_id="s1", paper_id="p1", doi_or_pmid="10.1/x", source_access=SourceAccess.ABSTRACT_ONLY
        )


def test_metadata_only_span_may_omit_anchor():
    s = EvidenceSpan(
        span_id="s1", paper_id="p1", doi_or_pmid="10.1/x", source_access=SourceAccess.METADATA_ONLY
    )
    assert s.anchor is None


def test_metadata_only_cannot_support_any_scientific_claim():
    """方案 8.1：metadata_only 只能证明论文存在，不能支撑科学主张。"""
    s = span(access=SourceAccess.METADATA_ONLY)
    assert s.can_support_scientific_claim() is False
    assert all(not s.can_support_slot(slot) for slot in ConditionSlot)


def test_abstract_only_cannot_backfill_dose_time_method():
    """方案 8.1：abstract_only 不能补写摘要未报告的剂量、时间或方法细节。"""
    s = span(access=SourceAccess.ABSTRACT_ONLY)
    assert s.can_support_scientific_claim() is True
    assert s.can_support_slot(ConditionSlot.DOSE) is False
    assert s.can_support_slot(ConditionSlot.TIME) is False
    assert s.can_support_slot(ConditionSlot.METHOD) is False
    assert s.can_support_slot(ConditionSlot.SPECIES) is True
    assert s.can_support_slot(ConditionSlot.EFFECT_DIRECTION) is True


def test_fulltext_supports_every_slot():
    s = span()
    assert all(s.can_support_slot(slot) for slot in ConditionSlot)


# ---------------------------------------------------------------------------
# 原子主张与条件槽位
# ---------------------------------------------------------------------------


def test_condition_slots_match_proposal_d4_list():
    assert set(ExperimentConditions.model_fields) == {slot.value for slot in ConditionSlot}


def test_effect_direction_domain():
    assert {d.value for d in EffectDirection} == {
        "increase",
        "decrease",
        "no_effect",
        "mixed",
        "unknown",
    }


def test_filled_slots_excludes_none():
    conditions = ExperimentConditions(
        species="Mus musculus", cell_type="INS-1E", effect_direction=EffectDirection.INCREASE
    )
    assert set(conditions.filled_slots()) == {
        ConditionSlot.SPECIES,
        ConditionSlot.CELL_TYPE,
        ConditionSlot.EFFECT_DIRECTION,
    }


def test_claim_defaults_to_secondary():
    claim = AtomicClaim(claim_id="c1", text="高糖增加线粒体碎片化")
    assert claim.is_core is False
    assert claim.citations == []
    assert claim.conditions.species is None


def test_claim_text_cannot_be_blank():
    with pytest.raises(ValidationError, match="不能为空"):
        AtomicClaim(claim_id="c1", text="  ")


def test_claim_carries_citation_with_span_refs():
    claim = AtomicClaim(
        claim_id="c1",
        text="高糖增加线粒体碎片化",
        is_core=True,
        citations=[Citation(doi_or_pmid="10.1/x", paper_id="p1", evidence_span_ids=["s1"])],
    )
    assert claim.citations[0].evidence_span_ids == ["s1"]


# ---------------------------------------------------------------------------
# 五值判定
# ---------------------------------------------------------------------------


def test_support_verdict_is_the_frozen_five_value_set():
    assert {v.value for v in SupportVerdict} == {
        "fully_supported",
        "partially_supported",
        "not_supported",
        "refuted",
        "unknown",
    }


@pytest.mark.parametrize(
    "verdict", [SupportVerdict.FULLY_SUPPORTED, SupportVerdict.PARTIALLY_SUPPORTED, SupportVerdict.REFUTED]
)
def test_positive_and_refuting_verdicts_require_evidence_refs(verdict):
    with pytest.raises(ValidationError, match="必须给出 evidence_span_refs"):
        JudgeVerdict(claim_id="c1", verdict=verdict, confidence=0.9, reason="r")


def test_unknown_verdict_needs_no_evidence():
    v = JudgeVerdict(claim_id="c1", verdict=SupportVerdict.UNKNOWN, confidence=0.2, reason="全文不可得")
    assert v.evidence_span_refs == []


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_a_probability(confidence):
    with pytest.raises(ValidationError):
        JudgeVerdict(claim_id="c1", verdict=SupportVerdict.UNKNOWN, confidence=confidence, reason="r")


# ---------------------------------------------------------------------------
# 金标记录（方案 9.2）
# ---------------------------------------------------------------------------


def test_question_gold_fields_match_proposal_9_2():
    assert set(QuestionGold.model_fields) == {
        "question_id",
        "question",
        "scope",
        "answerability",
        "required_claims",
        "optional_claims",
        "evidence_papers",
        "evidence_spans",
        "required_context_slots",
        "known_conflicts",
        "prohibited_inferences",
    }


def test_answerability_domain():
    assert {a.value for a in Answerability} == {
        "answerable",
        "partial",
        "insufficient",
        "out_of_scope",
    }


def test_required_claims_must_be_core():
    with pytest.raises(ValidationError, match="必须标记 is_core=True"):
        QuestionGold(
            question_id="Q1",
            question="高糖是否增加 β 细胞线粒体碎片化？",
            answerability=Answerability.ANSWERABLE,
            required_claims=[AtomicClaim(claim_id="c1", text="是", is_core=False)],
        )


def test_minimal_gold_record_round_trips():
    gold = QuestionGold(
        question_id="Q1",
        question="高糖条件下 β 细胞线粒体碎片化与胰岛素分泌障碍有哪些已发表证据？",
        scope="INS-1E / 小鼠原代胰岛，2015—2026",
        answerability=Answerability.PARTIAL,
        required_claims=[AtomicClaim(claim_id="c1", text="高糖增加碎片化", is_core=True)],
        evidence_papers=[
            EvidencePaper(
                paper_id="p1", doi_or_pmid="10.1/x", is_key_evidence=True, is_conflict_or_negative=True
            )
        ],
        evidence_spans=[span()],
        required_context_slots=[ConditionSlot.SPECIES, ConditionSlot.DOSE],
        known_conflicts=["急性高糖与慢性高糖结论不一致"],
        prohibited_inferences=["不得外推到人体在体"],
    )
    restored = QuestionGold.model_validate_json(gold.model_dump_json())
    assert restored == gold
    assert restored.required_context_slots == [ConditionSlot.SPECIES, ConditionSlot.DOSE]


def test_extra_fields_are_rejected():
    """方案 5.3：模型输出必须由本地 Pydantic 重新校验，不合规即拒绝。"""
    with pytest.raises(ValidationError):
        AtomicClaim(claim_id="c1", text="x", confidence_hack=0.99)


# ---------------------------------------------------------------------------
# 核验与结果契约
# ---------------------------------------------------------------------------


def test_verification_status_is_three_state():
    assert {s.value for s in VerificationStatus} == {"verified", "mismatch", "unresolved"}


def test_identifier_verification_records_reason_and_source():
    record = IdentifierVerification(
        input_id="10.1/x",
        normalized_id="10.1/x",
        id_type="doi",
        status=VerificationStatus.UNRESOLVED,
        reason="service_unavailable: 传输失败",
        source="crossref",
    )
    assert record.status is VerificationStatus.UNRESOLVED
    assert record.title is None


def test_dimension_score_na_must_not_carry_level():
    with pytest.raises(ValidationError, match="记 NA 时不得同时给出 level"):
        DimensionScore(dimension="D3", name_zh="x", weight=15, is_na=True, level=4)


def test_dimension_score_non_na_must_carry_level():
    with pytest.raises(ValidationError, match="必须给出 level"):
        DimensionScore(dimension="D3", name_zh="x", weight=15)


@pytest.mark.parametrize("level", [-1, 5])
def test_dimension_score_level_bounds(level):
    with pytest.raises(ValidationError):
        DimensionScore(dimension="D3", name_zh="x", weight=15, level=level)
