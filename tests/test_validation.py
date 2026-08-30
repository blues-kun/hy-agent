"""Hand-checkable tests for evaluator.validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluator.validation import ValidationInput, analyze_validation

REPO_ROOT = Path(__file__).resolve().parent.parent


def complete_payload() -> dict:
    return {
        "discrimination": [
            {"output_id": "g1", "quality_tier": "good", "score": 9},
            {"output_id": "g2", "quality_tier": "good", "score": 9},
            {"output_id": "m1", "quality_tier": "medium", "score": 6},
            {"output_id": "m2", "quality_tier": "medium", "score": 6},
            {"output_id": "b1", "quality_tier": "bad", "score": 2},
            {"output_id": "b2", "quality_tier": "bad", "score": 2},
        ],
        "agreement": {
            "nominal": [
                {"item_id": "n1", "rater_a": "yes", "rater_b": "yes"},
                {"item_id": "n2", "rater_a": "yes", "rater_b": "no"},
                {"item_id": "n3", "rater_a": "no", "rater_b": "no"},
                {"item_id": "n4", "rater_a": "no", "rater_b": "no"},
            ],
            "ordinal_0_4": [
                {"item_id": "o1", "rater_a": 0, "rater_b": 0},
                {"item_id": "o2", "rater_a": 1, "rater_b": 2},
                {"item_id": "o3", "rater_a": 3, "rater_b": 3},
                {"item_id": "o4", "rater_a": 4, "rater_b": 3},
            ],
            "total_scores": [
                {"item_id": "s1", "rater_a": 1, "rater_b": 1},
                {"item_id": "s2", "rater_a": 2, "rater_b": 3},
                {"item_id": "s3", "rater_a": 3, "rater_b": 2},
            ],
        },
        "stability": [
            {
                "output_id": "out1",
                "repeats": [
                    {"run_id": "r1", "score": 80, "grade": "REVIEW"},
                    {"run_id": "r2", "score": 82, "grade": "REVIEW"},
                    {"run_id": "r3", "score": 84, "grade": "PASS"},
                ],
            },
            {
                "output_id": "out2",
                "repeats": [
                    {"run_id": "r1", "score": 70, "grade": "REVIEW"},
                    {"run_id": "r2", "score": 70, "grade": "REVIEW"},
                ],
            },
        ],
        "adversarial": [
            {
                "pair_id": "a1",
                "clean_score": 90,
                "attacked_score": 70,
                "attack_detected": True,
                "clean_flagged": False,
                "severe_error": True,
            },
            {
                "pair_id": "a2",
                "clean_score": 90,
                "attacked_score": 88,
                "attack_detected": False,
                "clean_flagged": False,
                "severe_error": True,
            },
            {
                "pair_id": "a3",
                "clean_score": 85,
                "attacked_score": 80,
                "attack_detected": True,
                "clean_flagged": True,
                "severe_error": False,
            },
            {
                "pair_id": "a4",
                "clean_score": 80,
                "attacked_score": 90,
                "attack_detected": False,
                "clean_flagged": False,
                "severe_error": True,
            },
        ],
    }


def test_discrimination_is_strict_and_tau_b_accounts_for_ties():
    result = analyze_validation(ValidationInput.model_validate(complete_payload()))
    discrimination = result["discrimination"]
    assert discrimination["total_triplets"] == 8
    assert discrimination["correct_strict_triplets"] == 8
    assert discrimination["complete_order_accuracy"] == 1.0
    # Within-tier score ties are tied on both axes, so all 12 cross-tier pairs
    # are concordant and tau-b is exactly one.
    assert discrimination["kendall_pair_counts"] == {
        "concordant": 12,
        "discordant": 0,
        "tied_expected_only": 0,
        "tied_observed_only": 0,
        "tied_both": 3,
    }
    assert discrimination["kendall_tau_b"] == 1.0


def test_nominal_ordinal_and_total_score_agreement_match_hand_calculation():
    result = analyze_validation(ValidationInput.model_validate(complete_payload()))
    nominal = result["agreement"]["nominal"]
    assert nominal["raw_agreement"] == pytest.approx(0.75)
    assert nominal["expected_agreement"] == pytest.approx(0.5)
    assert nominal["cohen_kappa"] == pytest.approx(0.5)
    assert nominal["confusion_matrix"] == {
        "no": {"no": 2, "yes": 0},
        "yes": {"no": 1, "yes": 1},
    }

    ordinal = result["agreement"]["ordinal_0_4"]
    assert ordinal["raw_agreement"] == pytest.approx(0.5)
    assert ordinal["observed_weighted_agreement"] == pytest.approx(0.875)
    assert ordinal["expected_weighted_agreement"] == pytest.approx(0.59375)
    assert ordinal["linear_weighted_kappa"] == pytest.approx(9 / 13)

    total = result["agreement"]["total_scores"]
    assert total["mean_absolute_error"] == pytest.approx(2 / 3)
    assert total["spearman_rho"] == pytest.approx(0.5)
    assert total["icc_2_1"] == pytest.approx(0.6)
    assert total["icc_audit"]["ms_rows"] == pytest.approx(1.5)
    assert total["icc_audit"]["ms_columns"] == pytest.approx(0.0)
    assert total["icc_audit"]["ms_error"] == pytest.approx(0.5)


def test_stability_and_adversarial_statistics_are_auditable():
    result = analyze_validation(ValidationInput.model_validate(complete_payload()))
    out1 = result["stability"]["outputs"][0]
    assert out1["score_std_sample"] == pytest.approx(2.0)
    assert out1["score_median"] == pytest.approx(82.0)
    assert out1["score_p95_type7"] == pytest.approx(83.8)
    assert out1["grade_change_rate"] == pytest.approx(1 / 3)
    assert out1["grade_instability"] is True
    aggregate = result["stability"]["aggregate"]
    assert aggregate["output_grade_change_rate"] == pytest.approx(0.5)

    adversarial = result["adversarial"]
    assert adversarial["attack_detection"]["rate"] == pytest.approx(0.5)
    assert adversarial["clean_false_positive"]["rate"] == pytest.approx(0.25)
    assert adversarial["score_drop_clean_minus_attacked"]["mean"] == pytest.approx(4.25)
    assert adversarial["severe_attack_success"] == {
        "criterion": "severe_error=true and clean_score-attacked_score < 5",
        "numerator": 2,
        "denominator": 3,
        "rate": pytest.approx(2 / 3),
    }


def test_missing_small_and_constant_cases_return_null_with_warnings():
    payload = {
        "discrimination": [
            {"output_id": "g1", "quality_tier": "good", "score": 10},
            {"output_id": "m1", "quality_tier": "medium", "score": None},
            {"output_id": "b1", "quality_tier": "bad", "score": 10},
        ],
        "agreement": {
            "nominal": [
                {"item_id": "n1", "rater_a": "PASS", "rater_b": "PASS"},
                {"item_id": "n2", "rater_a": "PASS", "rater_b": "PASS"},
            ],
            "ordinal_0_4": [
                {"item_id": "o1", "rater_a": 4, "rater_b": 4},
                {"item_id": "o2", "rater_a": 4, "rater_b": 4},
            ],
            "total_scores": [
                {"item_id": "s1", "rater_a": 80, "rater_b": 80},
                {"item_id": "s2", "rater_a": 80, "rater_b": 80},
                {"item_id": "s3", "rater_a": 80, "rater_b": 80},
            ],
        },
        "stability": [
            {
                "output_id": "out1",
                "repeats": [{"run_id": "r1", "score": 80, "grade": None}],
            }
        ],
        "adversarial": [
            {
                "pair_id": "a1",
                "clean_score": 90,
                "attacked_score": None,
                "attack_detected": None,
                "clean_flagged": None,
                "severe_error": True,
            }
        ],
    }
    result = analyze_validation(ValidationInput.model_validate(payload))
    assert result["discrimination"]["complete_order_accuracy"] is None
    assert result["discrimination"]["kendall_tau_b"] is None
    assert result["agreement"]["nominal"]["cohen_kappa"] is None
    assert result["agreement"]["ordinal_0_4"]["linear_weighted_kappa"] is None
    assert result["agreement"]["total_scores"]["spearman_rho"] is None
    assert result["agreement"]["total_scores"]["icc_2_1"] is None
    assert result["stability"]["outputs"][0]["score_std_sample"] is None
    assert result["adversarial"]["attack_detection"]["rate"] is None
    assert result["adversarial"]["score_drop_clean_minus_attacked"]["mean"] is None
    warning_codes = {warning["code"] for warning in result["warnings"]}
    assert {
        "MISSING_SCORE_EXCLUDED",
        "INSUFFICIENT_TIER_SAMPLE",
        "CONSTANT_OR_DEGENERATE",
        "INSUFFICIENT_REPEATS",
        "INSUFFICIENT_SAMPLE",
    } <= warning_codes


def test_input_contract_rejects_duplicate_ids_and_out_of_range_ordinal():
    payload = complete_payload()
    payload["discrimination"][1]["output_id"] = "g1"
    with pytest.raises(ValidationError, match="必须唯一"):
        ValidationInput.model_validate(payload)

    payload = complete_payload()
    payload["agreement"]["ordinal_0_4"][0]["rater_a"] = 5
    with pytest.raises(ValidationError):
        ValidationInput.model_validate(payload)


def test_cli_prints_schema_and_writes_result(tmp_path: Path):
    schema_run = subprocess.run(
        [sys.executable, "scripts/analyze_validation.py", "--print-input-schema"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    schema = json.loads(schema_run.stdout)
    assert schema["additionalProperties"] is False
    assert "discrimination" in schema["properties"]

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(complete_payload()), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_validation.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == "mitoevidence.validation.v1"
    assert result["discrimination"]["complete_order_accuracy"] == 1.0
