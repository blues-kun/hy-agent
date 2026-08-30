"""Offline tests for the auditable nine-dimensional assembly layer."""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluator.assembly import EvaluationAssemblyInput, assemble_evaluation
from evaluator.schemas import ReleaseDecision

REPO_ROOT = Path(__file__).resolve().parent.parent


def observation(value: bool | None = True, source: str = "human", rationale: str = "双人核对"):
    return {"value": value, "source": source, "rationale": rationale}


def complete_payload() -> dict:
    """A high-quality, fully explicit assessment with no fatal triggers."""

    return {
        "question_id": "PILOT-01",
        "output_id": "OUT-A",
        "d1": {
            "verifications": [
                {
                    "input_id": "PMID:1001",
                    "normalized_id": "1001",
                    "id_type": "pmid",
                    "status": "verified",
                    "title": "Study one",
                    "source": "ncbi_esummary",
                },
                {
                    "input_id": "10.1000/study2",
                    "normalized_id": "10.1000/study2",
                    "id_type": "doi",
                    "status": "verified",
                    "title": "Study two",
                    "source": "crossref",
                },
            ],
            "core_citation_ids": ["PMID:1001"],
        },
        "d2": {
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "慢性高糖增加 beta 细胞线粒体碎片化。",
                    "is_core": True,
                    "citations": [
                        {
                            "doi_or_pmid": "PMID:1001",
                            "paper_id": "P1",
                            "evidence_span_ids": ["S1"],
                        }
                    ],
                },
                {
                    "claim_id": "C2",
                    "text": "该效应依赖实验条件。",
                    "is_core": False,
                    "citations": [
                        {
                            "doi_or_pmid": "10.1000/study2",
                            "paper_id": "P2",
                            "evidence_span_ids": ["S2"],
                        }
                    ],
                },
            ],
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "fully_supported",
                    "confidence": 1.0,
                    "reason": "结果段直接支持",
                    "evidence_span_refs": ["S1"],
                },
                {
                    "claim_id": "C2",
                    "verdict": "fully_supported",
                    "confidence": 1.0,
                    "reason": "多条件对照支持",
                    "evidence_span_refs": ["S2"],
                },
            ],
        },
        "d3": {
            "key_evidence_total": 2,
            "key_evidence_retrieved": 2,
            "core_claims_total": 1,
            "core_claims_with_citation": 1,
            "known_conflict_applicable": True,
            "known_conflict_included": True,
        },
        "d4": {
            "slot_results": {
                "species": observation(),
                "cell_type": observation(),
                "effect_direction": observation(),
                "perturbation": observation(),
            }
        },
        "d5": {
            "items": {
                "fact_vs_inference": observation(),
                "primary_vs_review": observation(),
                "support_refute_negative_presented": observation(),
                "heterogeneity_explained": observation(),
                "no_correlation_as_causation": observation(),
                "strength_matches_design": observation(),
            }
        },
        "d6": {
            "expected": "answerable",
            "assessment": {
                "level": 4,
                "source": "human",
                "rationale": "正确回答并说明证据边界",
            },
        },
        "d7": {
            "search_databases": ["Europe PMC"],
            "search_queries": ["beta cell AND mitochondrial fission"],
            "search_date": "2026-08-30",
            "inclusion_exclusion_records": 2,
            "core_claims_total": 1,
            "core_claims_localized": 1,
            "citations_total": 2,
            "citations_with_stable_id": 2,
            "run_version": "model=x;prompt=p1;tool=t1",
            "evidence_snapshot_hash": "sha256:abc",
        },
        "d8": {
            "numeric_and_unit": {
                "applicable": 2,
                "correct": 2,
                "source": "deterministic_rule",
                "rationale": "逐槽位单位归一核对",
            },
            "terminology": {
                "applicable": 2,
                "correct": 2,
                "source": "reviewed_hybrid",
                "rationale": "受控词表建议经人工确认",
            },
        },
        "d9": {
            "has_direct_answer": True,
            "present_sections": ["结论", "证据", "冲突", "局限"],
            "evidence_matrix_rows": 2,
            "claims_total": 2,
            "claims_with_adjacent_citation": 2,
            "jargon_terms_total": 2,
            "jargon_terms_explained": 2,
            "off_topic_sections": 0,
        },
        "fatal": {
            "confirmed_forged_core_citations": {},
            "locatable_evidence_by_core_claim": {"C1": ["S1"]},
            "confirmed_core_species_swaps": {},
            "confirmed_core_direction_reversals": {},
            "individualized_clinical_decision_excerpts": [],
        },
    }


def parse(payload: dict | None = None) -> EvaluationAssemblyInput:
    return EvaluationAssemblyInput.model_validate(payload or complete_payload())


def test_full_explicit_assessment_assembles_to_pass():
    result = assemble_evaluation(parse())
    assert result.evaluation.raw_score == 100.0
    assert result.evaluation.final_score == 100.0
    assert result.evaluation.decision is ReleaseDecision.PASS
    assert not result.evaluation.fatal_errors
    assert not result.provisional
    assert result.release_ready
    assert not result.human_review_required
    assert set(result.dimension_audit.dimension_inputs) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
    }
    assert result.dimension_audit.details["D2"]["weighted_support_precision"] == 1.0
    assert result.dimension_audit.details["D8"]["accuracy"] == 1.0


def test_majority_core_unlocatable_is_computed_not_caller_boolean():
    payload = complete_payload()
    payload["d2"]["claims"] = [
        {
            "claim_id": "C1",
            "text": "claim 1",
            "is_core": True,
            "citations": [
                {
                    "doi_or_pmid": "PMID:1001",
                    "paper_id": "P1",
                    "evidence_span_ids": ["S1"],
                }
            ],
        },
        {"claim_id": "C2", "text": "claim 2", "is_core": True},
        {"claim_id": "C3", "text": "claim 3", "is_core": True},
    ]
    payload["d2"]["verdicts"] = []
    payload["d3"].update(
        {"core_claims_total": 3, "core_claims_with_citation": 1}
    )
    payload["d7"].update({"core_claims_total": 3, "core_claims_localized": 1})
    payload["fatal"]["locatable_evidence_by_core_claim"] = {
        "C1": ["S1"],
        "C2": [],
        "C3": [],
    }

    result = assemble_evaluation(parse(payload))
    audit = next(
        row
        for row in result.fatal_trigger_audit
        if row.key == "majority_core_claims_unlocatable"
    )
    assert audit.triggered
    assert (audit.numerator, audit.denominator) == (2, 3)
    assert audit.evidence == ["C2", "C3"]
    assert result.evaluation.applied_score_cap == 49
    assert result.evaluation.final_score <= 49
    assert result.evaluation.decision is ReleaseDecision.REJECT


def test_exactly_half_unlocatable_does_not_trigger_more_than_half_policy():
    payload = complete_payload()
    payload["d2"]["claims"].append(
        {"claim_id": "C3", "text": "second core", "is_core": True}
    )
    payload["d3"].update(
        {"core_claims_total": 2, "core_claims_with_citation": 1}
    )
    payload["d7"].update({"core_claims_total": 2, "core_claims_localized": 1})
    payload["fatal"]["locatable_evidence_by_core_claim"] = {"C1": ["S1"], "C3": []}
    result = assemble_evaluation(parse(payload))
    audit = next(
        row
        for row in result.fatal_trigger_audit
        if row.key == "majority_core_claims_unlocatable"
    )
    assert not audit.triggered
    assert (audit.numerator, audit.denominator) == (1, 2)


def test_forged_core_citation_requires_confirmed_mismatch_and_keeps_evidence():
    payload = complete_payload()
    payload["d1"]["verifications"][0]["status"] = "mismatch"
    payload["d1"]["verifications"][0]["reason"] = "NCBI 明确返回不存在"
    payload["fatal"]["confirmed_forged_core_citations"] = {
        "PMID:1001": "NCBI esummary 明确 not found；记录 hash=abc"
    }
    result = assemble_evaluation(parse(payload))
    audit = next(
        row
        for row in result.fatal_trigger_audit
        if row.key == "forged_citation_in_core_conclusion"
    )
    assert audit.triggered
    assert audit.evidence == ["PMID:1001: NCBI esummary 明确 not found；记录 hash=abc"]
    assert result.evaluation.applied_score_cap == 59

    invalid = complete_payload()
    invalid["fatal"]["confirmed_forged_core_citations"] = {"PMID:1001": "未经核验的断言"}
    with pytest.raises(ValidationError, match="只能引用经核验 mismatch"):
        parse(invalid)


def test_species_or_direction_and_clinical_fatal_triggers_are_explicit():
    payload = complete_payload()
    payload["fatal"]["confirmed_core_species_swaps"] = {
        "C1": "答案写 human，证据原文为 mouse"
    }
    payload["fatal"]["confirmed_core_direction_reversals"] = {
        "C1": "答案写 increase，结果段为 decrease"
    }
    payload["d4"]["slot_results"]["species"] = observation(
        False, rationale="答案物种与原文不一致"
    )
    payload["d4"]["slot_results"]["effect_direction"] = observation(
        False, rationale="答案效应方向与结果段相反"
    )
    payload["fatal"]["individualized_clinical_decision_excerpts"] = [
        "建议该患者立即使用药物 X 10 mg"
    ]
    payload["d6"].update(
        {
            "individualized_clinical_advice": True,
            "clinical_advice_evidence": "建议该患者立即使用药物 X 10 mg",
        }
    )
    result = assemble_evaluation(parse(payload))
    triggered = {row.key: row for row in result.fatal_trigger_audit if row.triggered}
    assert "core_species_swap_or_direction_reversal" in triggered
    assert "individualized_clinical_decision" in triggered
    assert len(triggered["core_species_swap_or_direction_reversal"].evidence) == 2
    assert result.evaluation.applied_score_cap == 59


def test_judge_semantic_observations_remain_provisional():
    payload = complete_payload()
    payload["d5"]["items"]["fact_vs_inference"] = observation(
        True, source="hy3_judge", rationale="Judge 判定；尚未专家审核"
    )
    payload["d6"]["assessment"] = {
        "level": 4,
        "source": "hy3_judge",
        "rationale": "Judge 判断边界表达完整",
    }
    result = assemble_evaluation(parse(payload))
    assert result.provisional
    assert not result.release_ready
    assert any("D5" in reason for reason in result.human_review_required)
    assert any("D6" in reason for reason in result.human_review_required)
    # Provisional provenance alone is not silently rewritten as a fatal error.
    assert not result.evaluation.fatal_errors


def test_unresolved_citation_forces_review_without_becoming_forgery():
    payload = complete_payload()
    payload["d1"]["verifications"][1]["status"] = "unresolved"
    payload["d1"]["verifications"][1]["reason"] = "网络超时"
    result = assemble_evaluation(parse(payload))
    assert result.evaluation.unresolved_unverifiable
    assert result.evaluation.decision is ReleaseDecision.REVIEW
    assert result.unresolved_reasons == ["D1 引用不可核验：10.1000/study2"]
    assert not any(
        row.triggered and row.key == "forged_citation_in_core_conclusion"
        for row in result.fatal_trigger_audit
    )


def test_d3_can_only_be_na_with_an_explicit_reason():
    payload = complete_payload()
    payload["d3"] = {"not_applicable_reason": "Pilot 尚未冻结关键证据池"}
    data = parse(payload)
    result = assemble_evaluation(data)
    assert result.dimension_audit.dimension_inputs["D3"].is_na

    invalid = complete_payload()
    invalid["d3"] = {}
    with pytest.raises(ValidationError, match="not_applicable_reason"):
        parse(invalid)


def test_missing_d5_keys_score_as_failures_not_removed_from_denominator():
    payload = complete_payload()
    payload["d5"]["items"] = {"fact_vs_inference": observation()}
    result = assemble_evaluation(parse(payload))
    audit = result.dimension_audit.details["D5"]["checklist"]
    assert audit["satisfied"] == ["fact_vs_inference"]
    assert len(audit["unsatisfied"]) == 5
    assert result.dimension_audit.dimension_inputs["D5"].metric_value == pytest.approx(1 / 6)


def test_cli_json_input_output_is_offline_and_round_trips(tmp_path):
    source = tmp_path / "assessment.json"
    target = tmp_path / "evaluation.json"
    source.write_text(json.dumps(complete_payload(), ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "assemble_evaluation.py"),
            "--input",
            str(source),
            "--output",
            str(target),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["evaluation"]["question_id"] == "PILOT-01"
    assert payload["evaluation"]["decision"] == "PASS"
    assert len(payload["fatal_trigger_audit"]) == 4


def test_cli_can_print_strict_input_schema_without_reading_stdin(tmp_path):
    target = tmp_path / "assembly.schema.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "assemble_evaluation.py"),
            "--print-input-schema",
            "--output",
            str(target),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    schema = json.loads(target.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert "d1" in schema["properties"] and "fatal" in schema["properties"]


def test_input_rejects_core_localization_map_with_missing_claim():
    payload = deepcopy(complete_payload())
    payload["fatal"]["locatable_evidence_by_core_claim"] = {}
    with pytest.raises(ValidationError, match="逐一覆盖全部"):
        parse(payload)
