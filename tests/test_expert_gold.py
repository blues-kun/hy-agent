"""Offline tests for the designated expert consensus gold snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluator.expert_gold import (
    DEFAULT_MANIFEST_PATH,
    ExpertGoldAuditError,
    audit_expert_gold,
    load_expert_gold_records,
    selected_gold_fields,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, source: Path, *, count: int = 1) -> Path:
    manifest = {
        "schema_version": "mitoevidence.expert_consensus_gold.v1",
        "designation": "expert_consensus_gold",
        "confirmed_at": "2026-08-30",
        "confirmation_source": "project_owner",
        "rater_structure": {
            "inter_expert_agreement_computable": False,
            "reason": "single consolidated result",
        },
        "datasets": [
            {
                "name": "claim_reviews",
                "path": str(source.relative_to(root)),
                "sha256": _sha256(source),
                "record_count": count,
                "id_field": "review_id",
                "gold_fields": ["ai_decision"],
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_repository_snapshot_is_exactly_127_designated_gold_records():
    report = audit_expert_gold()
    assert report["ok"], report["errors"]
    assert report["designation"] == "expert_consensus_gold"
    assert report["total_records"] == 127
    assert {
        name: dataset["record_count"] for name, dataset in report["datasets"].items()
    } == {
        "pilot_questions": 5,
        "claim_reviews": 50,
        "terminology_rules": 60,
        "review_pool": 12,
    }
    assert report["inter_expert_agreement"]["computable"] is False


def test_repository_snapshot_reports_known_completeness_without_imputation():
    report = audit_expert_gold()
    pilot = report["datasets"]["pilot_questions"]
    assert pilot["summary"]["required_claims"] == 30
    assert pilot["summary"]["required_claims_core_true"] == 23
    assert pilot["summary"]["required_claims_core_false"] == 7
    assert pilot["summary"]["evidence_papers"] == 0
    assert pilot["summary"]["evidence_spans"] == 0
    assert pilot["fields"]["evidence_spans"] == {
        "present": 5,
        "non_null": 5,
        "non_empty": 0,
    }

    claims = report["datasets"]["claim_reviews"]
    assert claims["summary"]["decision"] == {
        "accept": 8,
        "accept_with_edits": 25,
        "reject": 14,
        "uncertain": 3,
    }
    assert claims["summary"]["usable_for_beta_cell_evidence"] == {
        "<null>": 3,
        "false": 25,
        "true": 22,
    }
    assert len(claims["summary"]["empty_recorded_conditions"]) == 12

    terminology = report["datasets"]["terminology_rules"]
    assert terminology["summary"]["explicit_approval_decision_present"] is False
    assert len(terminology["summary"]["missing_local_corpus_observation"]) == 22
    assert len(terminology["summary"]["empty_unresolved_notes"]) == 20

    review_pool = report["datasets"]["review_pool"]
    assert review_pool["summary"]["fulltext_sha256_present"] == 7
    assert review_pool["summary"]["bibliography_complete"]["pmcid"] == 9
    assert review_pool["summary"]["total_reference_count"] == 2043


def test_loader_preserves_legacy_names_and_null_labels():
    records = load_expert_gold_records()
    assert len(records["claim_reviews"]) == 50
    clm_26 = next(row for row in records["claim_reviews"] if row["review_id"] == "CLM-26")
    assert clm_26["usable_for_beta_cell_evidence"] is None
    selected = selected_gold_fields("claim_reviews", clm_26)
    assert selected["ai_decision"] == clm_26["ai_decision"]
    assert selected["usable_for_beta_cell_evidence"] is None
    assert "expert_decision" not in selected


def test_hash_drift_is_an_error(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"review_id":"C1","ai_decision":"accept"}\n', encoding="utf-8")
    manifest = _write_manifest(tmp_path, source)
    source.write_text('{"review_id":"C1","ai_decision":"reject"}\n', encoding="utf-8")
    report = audit_expert_gold(manifest, repo_root=tmp_path)
    assert not report["ok"]
    assert any("SHA-256 漂移" in error for error in report["errors"])
    with pytest.raises(ExpertGoldAuditError, match="审计失败"):
        load_expert_gold_records(manifest, repo_root=tmp_path)


def test_duplicate_ids_and_missing_gold_fields_are_errors(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        '{"review_id":"C1","ai_decision":"accept"}\n'
        '{"review_id":"C1"}\n',
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, source, count=2)
    report = audit_expert_gold(manifest, repo_root=tmp_path)
    assert not report["ok"]
    assert any("review_id 重复" in error for error in report["errors"])
    assert any("指定金标字段缺失" in error for error in report["errors"])


def test_manifest_path_is_the_repository_designation_file():
    assert DEFAULT_MANIFEST_PATH == (
        REPO_ROOT / "annotation_prelabel" / "expert_gold_manifest.json"
    )
