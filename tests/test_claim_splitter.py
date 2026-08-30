"""Tests for the independent, unfrozen claim-candidate splitter."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluator.claim_splitter import (
    ClaimSplitRequest,
    ReviewRisk,
    SplitBoundary,
    split_claim_candidates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def request(answer: str) -> ClaimSplitRequest:
    return ClaimSplitRequest(output_id="OUT-1", question="What is the evidence?", answer=answer)


def test_two_sentences_have_exact_reproducible_offsets():
    answer = "High glucose increased fragmentation. Insulin secretion decreased。"
    result = split_claim_candidates(request(answer))
    assert result.candidate_count == 2
    assert result.formal_denominator is None
    assert result.human_confirmed_claim_count is None
    for candidate in result.candidates:
        source = candidate.source
        assert answer[source.candidate_start_char : source.candidate_end_char] == candidate.text
        assert source.text_quote.exact == candidate.text
        assert answer[source.sentence_start_char : source.sentence_end_char] == source.sentence_exact


def test_semicolon_is_observable_split_but_and_is_not_blindly_split():
    answer = "Glucose increased fission and reduced secretion; ROS was unchanged."
    result = split_claim_candidates(request(answer))
    assert [item.text for item in result.candidates] == [
        "Glucose increased fission and reduced secretion",
        "ROS was unchanged.",
    ]
    assert all(item.split_boundary is SplitBoundary.SEMICOLON_CLAUSE for item in result.candidates)
    first = result.candidates[0]
    assert ReviewRisk.COORDINATED_PROPOSITIONS in first.review_risks
    assert ReviewRisk.MULTIPLE_EFFECT_DIRECTIONS in first.review_risks
    assert first.requires_human_review


def test_conditions_negation_contrast_and_anaphora_are_review_flags_not_inferences():
    answer = "This did not increase ATP under 5 mM glucose but did so after 30 min."
    item = split_claim_candidates(request(answer)).candidates[0]
    assert item.text == answer
    assert {
        ReviewRisk.ANAPHORA_OR_CONTEXT_DEPENDENCE,
        ReviewRisk.NEGATION_SCOPE,
        ReviewRisk.CONTRAST_OR_EXCEPTION,
        ReviewRisk.MULTIPLE_CONDITIONS_OR_COMPARATORS,
    }.issubset(set(item.review_risks))
    assert item.requires_human_review


def test_simple_single_claim_can_be_candidate_without_automatic_review_flag():
    result = split_claim_candidates(request("High glucose increased mitochondrial fission."))
    assert result.candidate_count == 1
    assert not result.candidates[0].review_risks
    assert not result.candidates[0].requires_human_review
    assert result.requires_human_review
    assert not result.contains_ambiguity_flags
    # It remains unfrozen and cannot become a formal denominator automatically.
    assert result.calibration_status == "unfrozen_requires_20_output_calibration"
    assert result.minimum_calibration_outputs == 20


def test_stable_ids_are_deterministic_and_change_with_output_identity():
    first = split_claim_candidates(request("ATP increased."))
    second = split_claim_candidates(request("ATP increased."))
    other = split_claim_candidates(
        ClaimSplitRequest(output_id="OUT-2", question="What is the evidence?", answer="ATP increased.")
    )
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].candidate_id != other.candidates[0].candidate_id


def test_input_rejects_tested_system_self_reported_claims_and_blank_answer():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ClaimSplitRequest.model_validate(
            {
                "output_id": "OUT-1",
                "question": "Q",
                "answer": "A",
                "claims": [{"text": "system says this is atomic"}],
            }
        )
    with pytest.raises(ValidationError, match="answer cannot be blank"):
        request("   ")


def test_cli_writes_strict_json_and_rejects_extra_claim_list(tmp_path: Path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "output_id": "OUT-CLI",
                "question": "Q",
                "answer": "Claim one. Claim two.",
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "split_claim_candidates.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 2
    assert payload["formal_denominator"] is None
    assert payload["independent_of_tested_system_claims"] is True

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps(
            {
                "output_id": "OUT-CLI",
                "question": "Q",
                "answer": "A",
                "self_reported_claims": ["A"],
            }
        ),
        encoding="utf-8",
    )
    bad = subprocess.run(
        command[:3] + [str(bad_path)] + command[4:],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "extra_forbidden" in bad.stderr
