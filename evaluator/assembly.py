"""Auditable assembly of the nine-dimensional evaluation inputs.

This module is deliberately an *assembler*, not another scientific judge.  It
converts explicit upstream records into :class:`~evaluator.rubric.DimensionInput`
objects and then delegates all thresholds and score arithmetic to
``evaluator.rubric.evaluate``.

The boundary matters:

* deterministic counts (citation status, evidence recall, slot accuracy and
  checklist counts) are computed here;
* semantic judgements for D2, D4, D5 and D6 must arrive as explicit records
  with provenance and a rationale;
* a Hy3 judgement is never relabelled as an expert judgement; provisional
  semantic observations are surfaced in ``human_review_required``;
* fatal-error triggers retain their concrete identifiers/excerpts in a
  separate audit trail because the v0.1 scoring engine only accepts fatal keys.

The input contract is intentionally strict.  Missing semantic observations
must be represented explicitly as ``value=null`` or by a documented dimension
NA reason; they are never silently inferred to be correct.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from evaluator.rubric import (
    DIMENSION_ORDER,
    DimensionInput,
    RubricConfig,
    coverage_composite,
    default_rubric,
    evaluate,
    slot_accuracy,
    weighted_support_precision,
)
from evaluator.rules.identifier_check import CitationCheckSummary, summarize_verifications
from evaluator.rules.structure_check import (
    ChecklistChecker,
    PresentationArtifacts,
    TraceabilityArtifacts,
    check_d7,
    check_d9,
)
from evaluator.schemas import (
    Answerability,
    AtomicClaim,
    ConditionSlot,
    EvaluationResult,
    IdentifierVerification,
    JudgeVerdict,
    StrictModel,
    SupportVerdict,
    VerificationStatus,
)


class AssessmentSource(str, Enum):
    """Who/what supplied a semantic observation.

    ``reviewed_hybrid`` means a person reviewed a rule/LLM proposal; it does not
    mean that the entire evaluation has become a formal expert gold standard.
    """

    HUMAN = "human"
    HY3_JUDGE = "hy3_judge"
    DETERMINISTIC_RULE = "deterministic_rule"
    REVIEWED_HYBRID = "reviewed_hybrid"


class AuditedObservation(StrictModel):
    """A tri-state checklist/slot observation with explicit provenance."""

    value: bool | None = Field(description="true=满足/正确，false=不满足/错误，null=NA")
    source: AssessmentSource
    rationale: str = Field(description="判定依据；NA 也必须说明不适用原因")

    @model_validator(mode="after")
    def _rationale_required(self) -> "AuditedObservation":
        if not self.rationale.strip():
            raise ValueError("语义观察必须提供非空 rationale，不能只给布尔结果")
        return self


class LevelAssessment(StrictModel):
    """Direct decision-table level used by D6."""

    level: int = Field(ge=0, le=4)
    source: AssessmentSource
    rationale: str

    @model_validator(mode="after")
    def _rationale_required(self) -> "LevelAssessment":
        if not self.rationale.strip():
            raise ValueError("D6 直接档位必须说明判定依据")
        return self


class CitationAssemblyInput(StrictModel):
    """D1 inputs produced by the identifier-verification layer."""

    verifications: list[IdentifierVerification]
    core_citation_ids: list[str] = Field(
        default_factory=list,
        description="直接支撑核心结论的原始 input_id；用于事件上限和伪造引用审计",
    )

    @model_validator(mode="after")
    def _unique_and_known_ids(self) -> "CitationAssemblyInput":
        ids = [record.input_id for record in self.verifications]
        if len(ids) != len(set(ids)):
            raise ValueError("D1 verifications 中 input_id 必须唯一")
        unknown = sorted(set(self.core_citation_ids) - set(ids))
        if unknown:
            raise ValueError(f"core_citation_ids 没有对应核验记录：{unknown}")
        if len(self.core_citation_ids) != len(set(self.core_citation_ids)):
            raise ValueError("core_citation_ids 不得重复")
        return self


class ClaimEvidenceAssemblyInput(StrictModel):
    """D2 claim set and one explicit final JudgeVerdict per assessed claim."""

    claims: list[AtomicClaim] = Field(min_length=1)
    verdicts: list[JudgeVerdict]

    @model_validator(mode="after")
    def _unique_and_known_claims(self) -> "ClaimEvidenceAssemblyInput":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("D2 claims 中 claim_id 必须唯一")
        verdict_ids = [verdict.claim_id for verdict in self.verdicts]
        if len(verdict_ids) != len(set(verdict_ids)):
            raise ValueError("D2 verdicts 中 claim_id 必须唯一")
        unknown = sorted(set(verdict_ids) - set(claim_ids))
        if unknown:
            raise ValueError(f"JudgeVerdict 引用了未知 claim_id：{unknown}")
        return self


class EvidenceCoverageInput(StrictModel):
    """Auditable D3 numerators and denominators.

    Use ``not_applicable_reason`` only when no frozen key-evidence pool exists.
    In that case D3 is explicitly NA rather than being awarded a default score.
    """

    key_evidence_total: int = Field(default=0, ge=0)
    key_evidence_retrieved: int = Field(default=0, ge=0)
    core_claims_total: int = Field(default=0, ge=0)
    core_claims_with_citation: int = Field(default=0, ge=0)
    known_conflict_applicable: bool = False
    known_conflict_included: bool | None = None
    not_applicable_reason: str | None = None

    @model_validator(mode="after")
    def _counts_and_na_are_coherent(self) -> "EvidenceCoverageInput":
        if self.key_evidence_retrieved > self.key_evidence_total:
            raise ValueError("key_evidence_retrieved 不能大于 key_evidence_total")
        if self.core_claims_with_citation > self.core_claims_total:
            raise ValueError("core_claims_with_citation 不能大于 core_claims_total")
        if self.known_conflict_applicable and self.known_conflict_included is None:
            raise ValueError("存在已知冲突时必须显式给出 known_conflict_included")
        if not self.known_conflict_applicable and self.known_conflict_included is not None:
            raise ValueError("冲突检查不适用时 known_conflict_included 必须为 null")
        if self.not_applicable_reason is not None:
            if not self.not_applicable_reason.strip():
                raise ValueError("D3 not_applicable_reason 不能为空白")
            if any(
                (
                    self.key_evidence_total,
                    self.key_evidence_retrieved,
                    self.core_claims_total,
                    self.core_claims_with_citation,
                )
            ):
                raise ValueError("D3 记 NA 时不得同时给出覆盖计数")
            return self
        if self.key_evidence_total == 0 or self.core_claims_total == 0:
            raise ValueError(
                "D3 计分需要正的 key_evidence_total 和 core_claims_total；"
                "尚无冻结金标池时请显式给出 not_applicable_reason"
            )
        return self


class SlotAssessmentInput(StrictModel):
    """D4 slot observations keyed by the rubric slot names."""

    slot_results: dict[str, AuditedObservation]

    @model_validator(mode="after")
    def _known_slots_only(self) -> "SlotAssessmentInput":
        allowed = {slot.value for slot in ConditionSlot}
        unknown = sorted(set(self.slot_results) - allowed)
        if unknown:
            raise ValueError(f"D4 出现未知槽位：{unknown}；合法值：{sorted(allowed)}")
        return self


class MechanismChecklistInput(StrictModel):
    """D5 checklist observations; missing configured keys score as failures."""

    items: dict[str, AuditedObservation]
    core_conclusion_contrary_to_evidence: bool = False
    contrary_evidence: str | None = None

    @model_validator(mode="after")
    def _contrary_evidence_required(self) -> "MechanismChecklistInput":
        deterministic = sorted(
            key
            for key, item in self.items.items()
            if item.source is AssessmentSource.DETERMINISTIC_RULE
        )
        if deterministic:
            raise ValueError(
                "D5 属于语义综合判断，不能伪装成确定性规则结论；"
                f"请使用 human / hy3_judge / reviewed_hybrid：{deterministic}"
            )
        if self.core_conclusion_contrary_to_evidence:
            if not self.contrary_evidence or not self.contrary_evidence.strip():
                raise ValueError("核心结论与整体证据方向相反时必须提供 contrary_evidence")
        elif self.contrary_evidence is not None:
            raise ValueError("未触发方向相反事件时 contrary_evidence 必须为 null")
        return self


class AnswerabilityAssessmentInput(StrictModel):
    """D6 is an explicit, provenance-bearing decision-table assessment."""

    expected: Answerability
    assessment: LevelAssessment
    unsupported_definite_conclusion: bool = False
    unsupported_conclusion_evidence: str | None = None
    individualized_clinical_advice: bool = False
    clinical_advice_evidence: str | None = None

    @model_validator(mode="after")
    def _event_evidence_required(self) -> "AnswerabilityAssessmentInput":
        if self.assessment.source is AssessmentSource.DETERMINISTIC_RULE:
            raise ValueError(
                "D6 answerability 档位需要人工或 Judge 语义判断，不能标为 deterministic_rule"
            )
        pairs = (
            (
                self.unsupported_definite_conclusion,
                self.unsupported_conclusion_evidence,
                "unsupported_conclusion_evidence",
            ),
            (
                self.individualized_clinical_advice,
                self.clinical_advice_evidence,
                "clinical_advice_evidence",
            ),
        )
        for triggered, evidence, field_name in pairs:
            if triggered and (evidence is None or not evidence.strip()):
                raise ValueError(f"触发 D6 事件时必须提供 {field_name}")
            if not triggered and evidence is not None:
                raise ValueError(f"未触发对应 D6 事件时 {field_name} 必须为 null")
        return self


class AccuracyComponent(StrictModel):
    """One audited D8 component represented by integer counts, not impressions."""

    applicable: int = Field(ge=0)
    correct: int = Field(ge=0)
    source: AssessmentSource
    rationale: str

    @model_validator(mode="after")
    def _counts_are_valid(self) -> "AccuracyComponent":
        if self.correct > self.applicable:
            raise ValueError("D8 correct 不能大于 applicable")
        if not self.rationale.strip():
            raise ValueError("D8 accuracy component 必须说明核验依据")
        return self

    @property
    def accuracy(self) -> float | None:
        return None if self.applicable == 0 else self.correct / self.applicable


class AccuracyAssemblyInput(StrictModel):
    """D8 numeric/unit and terminology accuracies, kept as separate audit rows."""

    numeric_and_unit: AccuracyComponent
    terminology: AccuracyComponent
    key_number_or_unit_error: bool = False
    key_error_evidence: str | None = None
    magnitude_or_unit_error: bool = False
    magnitude_error_evidence: str | None = None

    @model_validator(mode="after")
    def _event_evidence_required(self) -> "AccuracyAssemblyInput":
        pairs = (
            (self.key_number_or_unit_error, self.key_error_evidence, "key_error_evidence"),
            (
                self.magnitude_or_unit_error,
                self.magnitude_error_evidence,
                "magnitude_error_evidence",
            ),
        )
        for triggered, evidence, field_name in pairs:
            if triggered and (evidence is None or not evidence.strip()):
                raise ValueError(f"触发 D8 事件时必须提供 {field_name}")
            if not triggered and evidence is not None:
                raise ValueError(f"未触发对应 D8 事件时 {field_name} 必须为 null")
        return self


class FatalTriggerInput(StrictModel):
    """Concrete evidence used to trigger the four fatal-error policies.

    The majority-unlocatable trigger is not a caller-supplied boolean.  Every
    core claim must be present in ``locatable_evidence_by_core_claim`` and the
    assembler deterministically tests whether more than half of the lists are
    empty.
    """

    confirmed_forged_core_citations: dict[str, str] = Field(
        default_factory=dict, description="core citation input_id -> confirmation evidence"
    )
    locatable_evidence_by_core_claim: dict[str, list[str]] = Field(
        description="core claim_id -> resolved span/anchor IDs; empty means unlocatable"
    )
    confirmed_core_species_swaps: dict[str, str] = Field(
        default_factory=dict, description="core claim_id -> source/gold comparison"
    )
    confirmed_core_direction_reversals: dict[str, str] = Field(
        default_factory=dict, description="core claim_id -> source/gold comparison"
    )
    individualized_clinical_decision_excerpts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_evidence_nonblank(self) -> "FatalTriggerInput":
        mappings = (
            self.confirmed_forged_core_citations,
            self.confirmed_core_species_swaps,
            self.confirmed_core_direction_reversals,
        )
        if any(not key.strip() or not value.strip() for mapping in mappings for key, value in mapping.items()):
            raise ValueError("致命错误标识和触发依据均不得为空白")
        if any(not excerpt.strip() for excerpt in self.individualized_clinical_decision_excerpts):
            raise ValueError("个体化诊疗决策摘录不得为空白")
        for claim_id, refs in self.locatable_evidence_by_core_claim.items():
            if not claim_id.strip() or any(not ref.strip() for ref in refs):
                raise ValueError("核心主张定位映射中的 claim_id/证据引用不得为空白")
            if len(refs) != len(set(refs)):
                raise ValueError(f"核心主张 {claim_id} 的可定位证据引用不得重复")
        return self


class EvaluationAssemblyInput(StrictModel):
    """Complete JSON contract for assembling one output's evaluation."""

    question_id: str
    output_id: str | None = None
    d1: CitationAssemblyInput
    d2: ClaimEvidenceAssemblyInput
    d3: EvidenceCoverageInput
    d4: SlotAssessmentInput
    d5: MechanismChecklistInput
    d6: AnswerabilityAssessmentInput
    d7: TraceabilityArtifacts
    d8: AccuracyAssemblyInput
    d9: PresentationArtifacts
    fatal: FatalTriggerInput
    additional_unresolved_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity_sets_are_consistent(self) -> "EvaluationAssemblyInput":
        if not self.question_id.strip():
            raise ValueError("question_id 不能为空")
        core_claim_ids = {claim.claim_id for claim in self.d2.claims if claim.is_core}
        mapped = set(self.fatal.locatable_evidence_by_core_claim)
        if mapped != core_claim_ids:
            missing = sorted(core_claim_ids - mapped)
            extra = sorted(mapped - core_claim_ids)
            raise ValueError(
                "locatable_evidence_by_core_claim 必须逐一覆盖全部且仅覆盖核心主张；"
                f"missing={missing}, extra={extra}"
            )
        if self.d3.not_applicable_reason is None and self.d3.core_claims_total != len(core_claim_ids):
            raise ValueError(
                "D3 core_claims_total 必须等于 D2 的核心主张数；"
                f"got {self.d3.core_claims_total}, expected {len(core_claim_ids)}"
            )
        cited_core_count = sum(
            1 for claim in self.d2.claims if claim.is_core and bool(claim.citations)
        )
        if (
            self.d3.not_applicable_reason is None
            and self.d3.core_claims_with_citation != cited_core_count
        ):
            raise ValueError(
                "D3 core_claims_with_citation 必须与 D2 核心主张的显式引用一致；"
                f"got {self.d3.core_claims_with_citation}, expected {cited_core_count}"
            )
        localized_count = sum(
            1 for refs in self.fatal.locatable_evidence_by_core_claim.values() if refs
        )
        if (
            self.d7.core_claims_total != len(core_claim_ids)
            or self.d7.core_claims_localized != localized_count
        ):
            raise ValueError(
                "D7 核心主张总数/定位数必须与 fatal 定位映射一致；"
                f"got {self.d7.core_claims_localized}/{self.d7.core_claims_total}, "
                f"expected {localized_count}/{len(core_claim_ids)}"
            )

        claim_by_id = {claim.claim_id: claim for claim in self.d2.claims}
        verdict_by_id = {verdict.claim_id: verdict for verdict in self.d2.verdicts}
        for claim_id, refs in self.fatal.locatable_evidence_by_core_claim.items():
            known_refs = {
                ref
                for citation in claim_by_id[claim_id].citations
                for ref in citation.evidence_span_ids
            }
            verdict = verdict_by_id.get(claim_id)
            if verdict is not None:
                known_refs.update(verdict.evidence_span_refs)
            unknown_refs = sorted(set(refs) - known_refs)
            if unknown_refs:
                raise ValueError(
                    f"核心主张 {claim_id} 的定位映射引用了未绑定到该主张的证据："
                    f"{unknown_refs}"
                )

        known_core = core_claim_ids
        for label, ids in (
            ("confirmed_core_species_swaps", set(self.fatal.confirmed_core_species_swaps)),
            (
                "confirmed_core_direction_reversals",
                set(self.fatal.confirmed_core_direction_reversals),
            ),
        ):
            unknown = sorted(ids - known_core)
            if unknown:
                raise ValueError(f"{label} 引用了非核心或未知主张：{unknown}")

        if self.fatal.confirmed_core_species_swaps:
            species = self.d4.slot_results.get("species")
            if species is None or species.value is not False:
                raise ValueError("确认核心物种偷换时 D4 species 必须显式判为 false")
        if self.fatal.confirmed_core_direction_reversals:
            direction = self.d4.slot_results.get("effect_direction")
            if direction is None or direction.value is not False:
                raise ValueError(
                    "确认核心效应方向反转时 D4 effect_direction 必须显式判为 false"
                )
        has_clinical_decision = bool(
            self.fatal.individualized_clinical_decision_excerpts
        )
        if has_clinical_decision != self.d6.individualized_clinical_advice:
            raise ValueError(
                "fatal 个体化诊疗决策摘录与 D6 individualized_clinical_advice 必须一致"
            )

        verification_by_id = {record.input_id: record for record in self.d1.verifications}
        core_citations = set(self.d1.core_citation_ids)
        forged = set(self.fatal.confirmed_forged_core_citations)
        unknown_forged = sorted(forged - core_citations)
        if unknown_forged:
            raise ValueError(f"伪造引用触发项并非核心引用：{unknown_forged}")
        not_confirmed = sorted(
            citation_id
            for citation_id in forged
            if verification_by_id[citation_id].status is not VerificationStatus.MISMATCH
        )
        if not_confirmed:
            raise ValueError(
                "confirmed_forged_core_citations 只能引用经核验 mismatch 的核心引用："
                f"{not_confirmed}"
            )
        if any(not reason.strip() for reason in self.additional_unresolved_reasons):
            raise ValueError("additional_unresolved_reasons 不得包含空白项")
        return self


class FatalTriggerAudit(StrictModel):
    """One fatal policy's auditable trigger result."""

    key: str
    triggered: bool
    score_cap: int
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    numerator: int | None = None
    denominator: int | None = None


class DimensionAudit(StrictModel):
    """Intermediate, human-readable metrics retained alongside the score."""

    dimension_inputs: dict[str, DimensionInput]
    details: dict[str, Any]


class EvaluationAssemblyOutput(StrictModel):
    """Evaluation plus provenance that the base scorer does not retain."""

    evaluation: EvaluationResult
    dimension_audit: DimensionAudit
    fatal_trigger_audit: list[FatalTriggerAudit]
    unresolved_reasons: list[str] = Field(default_factory=list)
    human_review_required: list[str] = Field(default_factory=list)
    provisional: bool = Field(
        description="true when unreviewed Hy3 semantics or unresolved items still require review"
    )
    release_ready: bool = Field(
        description="true only when the base decision is PASS and no pending review remains"
    )


def _checklist_input(
    dimension: str,
    observations: dict[str, AuditedObservation],
    config: RubricConfig,
) -> tuple[DimensionInput, dict[str, Any]]:
    raw = {key: observation.value for key, observation in observations.items()}
    result = ChecklistChecker(dimension, config).check(raw)
    if result.applicable_count == 0:
        return DimensionInput(is_na=True, notes="全部清单条目显式记 NA"), result.model_dump()
    if dimension == "D5":
        # ChecklistChecker treats omitted configured keys as failures (scheme
        # 10.3).  Use its normalized result rather than only the caller's keys.
        metric = result.satisfied_count / result.applicable_count
    else:
        metric = float(result.metric_count)
    return DimensionInput(metric_value=metric), result.model_dump()


def _human_review_reasons(data: EvaluationAssemblyInput) -> list[str]:
    reasons: list[str] = []
    provisional_sources = {AssessmentSource.HY3_JUDGE}

    for slot, observation in data.d4.slot_results.items():
        if observation.source in provisional_sources:
            reasons.append(f"D4 槽位 {slot} 仅由 Hy3 Judge 判定，尚需人工复核")
    for key, observation in data.d5.items.items():
        if observation.source in provisional_sources:
            reasons.append(f"D5 清单 {key} 仅由 Hy3 Judge 判定，尚需人工复核")
    if data.d6.assessment.source in provisional_sources:
        reasons.append("D6 answerability 档位仅由 Hy3 Judge 判定，尚需人工复核")
    for name, component in (
        ("numeric_and_unit", data.d8.numeric_and_unit),
        ("terminology", data.d8.terminology),
    ):
        if component.source in provisional_sources:
            reasons.append(f"D8 {name} 准确率仅由 Hy3 Judge 判定，尚需人工复核")
    return reasons


def _fatal_audit(
    data: EvaluationAssemblyInput, config: RubricConfig
) -> list[FatalTriggerAudit]:
    fatal_cfg = config.fatal_errors
    forged = data.fatal.confirmed_forged_core_citations
    core_map = data.fatal.locatable_evidence_by_core_claim
    unlocatable = sorted(claim_id for claim_id, refs in core_map.items() if not refs)
    core_total = len(core_map)
    majority_unlocatable = core_total > 0 and len(unlocatable) / core_total > 0.5
    species = data.fatal.confirmed_core_species_swaps
    reversals = data.fatal.confirmed_core_direction_reversals
    clinical = data.fatal.individualized_clinical_decision_excerpts

    def cap(key: str) -> int:
        return int(fatal_cfg[key]["score_cap"])

    return [
        FatalTriggerAudit(
            key="forged_citation_in_core_conclusion",
            triggered=bool(forged),
            score_cap=cap("forged_citation_in_core_conclusion"),
            rationale=(
                "至少一条核心结论引用被上游标识符核验明确判为 mismatch"
                if forged
                else "没有经明确核验为 mismatch 的核心引用触发记录"
            ),
            evidence=[f"{citation_id}: {reason}" for citation_id, reason in sorted(forged.items())],
        ),
        FatalTriggerAudit(
            key="majority_core_claims_unlocatable",
            triggered=majority_unlocatable,
            score_cap=cap("majority_core_claims_unlocatable"),
            rationale=(
                f"无可定位证据的核心主张为 {len(unlocatable)}/{core_total}；"
                "只有比例严格大于 1/2 才触发"
            ),
            evidence=unlocatable,
            numerator=len(unlocatable),
            denominator=core_total,
        ),
        FatalTriggerAudit(
            key="core_species_swap_or_direction_reversal",
            triggered=bool(species or reversals),
            score_cap=cap("core_species_swap_or_direction_reversal"),
            rationale=(
                "存在经显式核对确认的核心物种偷换或效应方向反转"
                if species or reversals
                else "未提供经确认的核心物种偷换或效应方向反转"
            ),
            evidence=(
                [f"species_swap {claim_id}: {reason}" for claim_id, reason in sorted(species.items())]
                + [
                    f"direction_reversal {claim_id}: {reason}"
                    for claim_id, reason in sorted(reversals.items())
                ]
            ),
        ),
        FatalTriggerAudit(
            key="individualized_clinical_decision",
            triggered=bool(clinical),
            score_cap=cap("individualized_clinical_decision"),
            rationale=(
                "输出中存在被显式标注的患者个体化用药或诊疗决策摘录"
                if clinical
                else "未标注患者个体化用药或诊疗决策"
            ),
            evidence=list(clinical),
        ),
    ]


def assemble_evaluation(
    data: EvaluationAssemblyInput,
    config: RubricConfig | None = None,
) -> EvaluationAssemblyOutput:
    """Assemble nine dimensions and execute the existing scoring engine."""

    cfg = config or default_rubric()
    details: dict[str, Any] = {}
    dimensions: dict[str, DimensionInput] = {}

    # D1: three-state identifier verification.  Unresolved is not a mismatch.
    d1_summary: CitationCheckSummary = summarize_verifications(data.d1.verifications)
    d1_by_id = {record.input_id: record for record in data.d1.verifications}
    core_mismatch = sorted(
        citation_id
        for citation_id in data.d1.core_citation_ids
        if d1_by_id[citation_id].status is VerificationStatus.MISMATCH
    )
    d1_flags = {
        "core_conclusion_uses_wrong_citation": bool(core_mismatch),
        "nonexistent_identifier_count": d1_summary.nonexistent_identifier_count,
    }
    dimensions["D1"] = (
        DimensionInput(
            metric_value=d1_summary.metadata_match_rate,
            event_flags=d1_flags,
            notes="unresolved 不进入 D1 比例分子/分母并单独触发人工复核",
        )
        if d1_summary.is_scorable
        else DimensionInput(is_na=True, notes="全部引用不可核验，D1 显式记 NA")
    )
    details["D1"] = {
        "summary": d1_summary.model_dump(mode="json"),
        "event_flags": d1_flags,
        "core_mismatch_ids": core_mismatch,
    }

    # D2: explicit JudgeVerdicts.  Missing verdicts remain in the denominator.
    verdict_by_id = {verdict.claim_id: verdict for verdict in data.d2.verdicts}
    d2_metric = weighted_support_precision(data.d2.claims, verdict_by_id, cfg)
    core_claim_ids = {claim.claim_id for claim in data.d2.claims if claim.is_core}
    refuted_core = sorted(
        claim_id
        for claim_id in core_claim_ids
        if claim_id in verdict_by_id
        and verdict_by_id[claim_id].verdict is SupportVerdict.REFUTED
    )
    d2_flags = {
        "core_direction_reversal": bool(data.fatal.confirmed_core_direction_reversals),
        "core_contradiction_count": len(refuted_core),
        "majority_core_claims_contradicted": bool(core_claim_ids)
        and len(refuted_core) / len(core_claim_ids) > 0.5,
    }
    dimensions["D2"] = DimensionInput(metric_value=d2_metric, event_flags=d2_flags)
    details["D2"] = {
        "weighted_support_precision": d2_metric,
        "verdicts_present": sorted(verdict_by_id),
        "missing_verdicts_scored_as_unknown": sorted(
            {claim.claim_id for claim in data.d2.claims} - set(verdict_by_id)
        ),
        "refuted_core_claim_ids": refuted_core,
        "event_flags": d2_flags,
    }

    # D3: explicit numerators/denominators; no gold pool means explicit NA.
    if data.d3.not_applicable_reason is not None:
        dimensions["D3"] = DimensionInput(
            is_na=True, notes=data.d3.not_applicable_reason
        )
        details["D3"] = {"not_applicable_reason": data.d3.not_applicable_reason}
    else:
        recall = data.d3.key_evidence_retrieved / data.d3.key_evidence_total
        completeness = data.d3.core_claims_with_citation / data.d3.core_claims_total
        composite = coverage_composite(recall, completeness, cfg)
        d3_flags = {
            "known_conflict_evidence_missing": data.d3.known_conflict_applicable
            and not bool(data.d3.known_conflict_included)
        }
        dimensions["D3"] = DimensionInput(metric_value=composite, event_flags=d3_flags)
        details["D3"] = {
            "pooled_evidence_recall": recall,
            "core_claim_citation_completeness": completeness,
            "coverage_composite": composite,
            "counts": data.d3.model_dump(mode="json"),
            "event_flags": d3_flags,
        }

    # D4: semantic slot correctness arrives explicitly with provenance.
    d4_values = {slot: observation.value for slot, observation in data.d4.slot_results.items()}
    if not d4_values or all(value is None for value in d4_values.values()):
        dimensions["D4"] = DimensionInput(is_na=True, notes="D4 所有槽位显式记 NA")
        d4_metric = None
    else:
        d4_metric = slot_accuracy(d4_values, cfg)
        key_slots = set(cfg.dimensions["D4"]["key_slots"])
        key_errors = sorted(
            slot for slot, value in d4_values.items() if slot in key_slots and value is False
        )
        confusion_count = len(
            set(data.fatal.confirmed_core_species_swaps)
            | set(data.fatal.confirmed_core_direction_reversals)
        )
        d4_flags = {
            "key_slot_error": bool(key_errors),
            "key_condition_swap_count": len(data.fatal.confirmed_core_species_swaps),
            "species_cell_direction_confusion_count": confusion_count,
        }
        dimensions["D4"] = DimensionInput(metric_value=d4_metric, event_flags=d4_flags)
    details["D4"] = {
        "slot_accuracy": d4_metric,
        "slot_results": {
            slot: observation.model_dump(mode="json")
            for slot, observation in data.d4.slot_results.items()
        },
        "event_flags": dimensions["D4"].event_flags,
    }

    # D5: checklist items may be human or Judge supplied, but provenance remains.
    dimensions["D5"], d5_detail = _checklist_input("D5", data.d5.items, cfg)
    if not dimensions["D5"].is_na:
        dimensions["D5"] = dimensions["D5"].model_copy(
            update={
                "event_flags": {
                    "core_conclusion_contrary_to_evidence": (
                        data.d5.core_conclusion_contrary_to_evidence
                    )
                }
            }
        )
    details["D5"] = {
        "checklist": d5_detail,
        "observations": {
            key: observation.model_dump(mode="json")
            for key, observation in data.d5.items.items()
        },
        "contrary_evidence": data.d5.contrary_evidence,
    }

    # D6: direct decision-table level.  The assembler does not invent it.
    d6_flags = {
        "unsupported_definite_conclusion": data.d6.unsupported_definite_conclusion,
        "individualized_clinical_advice": data.d6.individualized_clinical_advice,
    }
    dimensions["D6"] = DimensionInput(
        level=data.d6.assessment.level,
        event_flags=d6_flags,
        notes=(
            f"expected={data.d6.expected.value}; source={data.d6.assessment.source.value}; "
            f"{data.d6.assessment.rationale}"
        ),
    )
    details["D6"] = data.d6.model_dump(mode="json")

    # D7: deterministic artifacts -> six-item configured checklist.
    d7_result, d7_flags = check_d7(data.d7, config=cfg)
    dimensions["D7"] = DimensionInput(
        metric_value=float(d7_result.metric_count), event_flags=d7_flags
    )
    details["D7"] = {
        "artifacts": data.d7.model_dump(mode="json"),
        "checklist": d7_result.model_dump(mode="json"),
        "event_flags": d7_flags,
    }

    # D8: combine separate audited numeric/unit and terminology count rows.
    d8_components = (data.d8.numeric_and_unit, data.d8.terminology)
    d8_total = sum(component.applicable for component in d8_components)
    d8_correct = sum(component.correct for component in d8_components)
    d8_flags = {
        "key_number_or_unit_error": data.d8.key_number_or_unit_error,
        "magnitude_or_unit_error": data.d8.magnitude_or_unit_error,
    }
    if d8_total == 0:
        dimensions["D8"] = DimensionInput(is_na=True, notes="数字/单位和术语均无适用项")
        d8_accuracy = None
    else:
        d8_accuracy = d8_correct / d8_total
        dimensions["D8"] = DimensionInput(metric_value=d8_accuracy, event_flags=d8_flags)
    details["D8"] = {
        "accuracy": d8_accuracy,
        "applicable": d8_total,
        "correct": d8_correct,
        "numeric_and_unit": data.d8.numeric_and_unit.model_dump(mode="json"),
        "terminology": data.d8.terminology.model_dump(mode="json"),
        "event_flags": d8_flags,
        "event_evidence": {
            "key_error": data.d8.key_error_evidence,
            "magnitude_or_unit_error": data.d8.magnitude_error_evidence,
        },
    }

    # D9: deterministic presentation artifacts -> six-item checklist.
    d9_result = check_d9(data.d9, config=cfg)
    dimensions["D9"] = DimensionInput(metric_value=float(d9_result.metric_count))
    details["D9"] = {
        "artifacts": data.d9.model_dump(mode="json"),
        "checklist": d9_result.model_dump(mode="json"),
    }

    if set(dimensions) != set(DIMENSION_ORDER):  # defensive invariant
        raise RuntimeError(f"内部错误：九维汇总不完整，实际 {sorted(dimensions)}")

    fatal_audit = _fatal_audit(data, cfg)
    fatal_keys = [record.key for record in fatal_audit if record.triggered]
    unresolved = [
        *(f"D1 引用不可核验：{identifier}" for identifier in d1_summary.unresolved_ids),
        *data.additional_unresolved_reasons,
    ]
    human_review = _human_review_reasons(data)
    if unresolved:
        human_review.extend(unresolved)

    result = evaluate(
        question_id=data.question_id,
        output_id=data.output_id,
        dimension_inputs=dimensions,
        fatal_error_keys=fatal_keys,
        unresolved_unverifiable=bool(unresolved),
        config=cfg,
    )
    return EvaluationAssemblyOutput(
        evaluation=result,
        dimension_audit=DimensionAudit(dimension_inputs=dimensions, details=details),
        fatal_trigger_audit=fatal_audit,
        unresolved_reasons=unresolved,
        human_review_required=human_review,
        provisional=bool(human_review),
        release_ready=result.decision.value == "PASS" and not human_review,
    )


__all__ = [
    "AccuracyAssemblyInput",
    "AccuracyComponent",
    "AnswerabilityAssessmentInput",
    "AssessmentSource",
    "AuditedObservation",
    "CitationAssemblyInput",
    "ClaimEvidenceAssemblyInput",
    "EvaluationAssemblyInput",
    "EvaluationAssemblyOutput",
    "EvidenceCoverageInput",
    "FatalTriggerAudit",
    "FatalTriggerInput",
    "LevelAssessment",
    "MechanismChecklistInput",
    "SlotAssessmentInput",
    "assemble_evaluation",
]
