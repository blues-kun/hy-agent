"""Regression tests for dual-schema A/B/C/D grid auditing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ablation import (
    ABLATION_ARTIFACT_VERSION_V2,
    ARM_DEFINITIONS,
    AblationCellRecord,
    CellOutcome,
    InputSnapshot,
    PilotAblationSuiteState,
    PilotArm,
    SuiteStatus,
    audit_pilot_ablation_grid,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64


def _record(arm: PilotArm, *, failed: bool = False) -> AblationCellRecord:
    common = {
        "question_id": "q1",
        "replicate": 1,
        "arm": arm,
        "cell_dir": f"q1/replicate-01/{arm.value}",
    }
    if failed:
        return AblationCellRecord(
            **common,
            outcome=CellOutcome.FAILED,
            failure_type="SchemaError",
            failure_reason="fixture structured output failure",
        )
    return AblationCellRecord(
        **common,
        outcome=CellOutcome.SUCCEEDED,
        cell_manifest_sha256=ZERO_HASH,
    )


def _runtime_state(*, complete: bool) -> PilotAblationSuiteState:
    records = (
        [
            _record(PilotArm.A),
            _record(PilotArm.B),
            _record(PilotArm.C),
            _record(PilotArm.D, failed=True),
        ]
        if complete
        else [_record(PilotArm.A)]
    )
    return PilotAblationSuiteState(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        suite_id="runtime-grid-fixture",
        status=SuiteStatus.COMPLETED if complete else SuiteStatus.RUNNING,
        created_at_utc="2026-08-30T00:00:00+00:00",
        completed_at_utc="2026-08-30T00:01:00+00:00" if complete else None,
        input_snapshot=InputSnapshot(
            path="pilot.jsonl",
            sha256=ONE_HASH,
            question_ids=["q1"],
        ),
        evidence_manifest_path="eval/data/evidence_pool_manifest.json",
        evidence_manifest_sha256=ZERO_HASH,
        arm_definitions=list(ARM_DEFINITIONS),
        replicates=1,
        top_k=12,
        judge_k=1,
        expected_grid_cells=4,
        records=records,
    )


def _run_cli(source: Path, target: Path, *, require_complete: bool = False):
    command = [
        sys.executable,
        "scripts/audit_ablation_grid.py",
        "--input",
        str(source),
        "--output",
        str(target),
    ]
    if require_complete:
        command.append("--require-complete")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_runtime_grid_cli_accepts_suite_state_and_retains_failed_cell(tmp_path: Path):
    state = _runtime_state(complete=True)
    direct = audit_pilot_ablation_grid(state)
    assert direct["grid_complete"] is True
    assert direct["outcomes"] == {"failed": 1, "succeeded": 3}

    source = tmp_path / "suite_state.json"
    target = tmp_path / "audit.json"
    source.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    completed = _run_cli(source, target, require_complete=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["input_schema"] == "mitoevidence.pilot-ablation.v2"
    assert result["schema_version"] == "mitoevidence.pilot-ablation-grid-audit.v1"
    assert result["expected_grid_cells"] == 4
    assert result["declared_expected_grid_cells"] == 4
    assert result["recorded_grid_cells"] == 4
    assert result["missing_grid_cells"] == 0
    assert result["grid_complete"] is True
    assert result["outcomes"] == {"failed": 1, "succeeded": 3}
    assert result["by_arm"]["D"]["failed"] == 1
    assert result["checks"]["declared_expected_matches_cartesian_product"] is True
    assert result["checks"]["explicit_failed_cells_retained_in_denominator"] is True
    d_c_check = result["artifact_level_checks"]["d_parent_c_artifact_hash_binding"]
    assert d_c_check["checked"] is False
    assert "artifact" in d_c_check["reason"]


def test_runtime_running_grid_returns_two_only_for_missing_cartesian_cells(tmp_path: Path):
    state = _runtime_state(complete=False)
    source = tmp_path / "suite_state.json"
    target = tmp_path / "audit.json"
    source.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    completed = _run_cli(source, target, require_complete=True)
    assert completed.returncode == 2, completed.stderr
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["input_schema"] == "mitoevidence.pilot-ablation.v2"
    assert result["recorded_grid_cells"] == 1
    assert result["missing_grid_cells"] == 3
    assert result["missing"] == [
        {"question_id": "q1", "arm": arm, "replicate": 1}
        for arm in ("B", "C", "D")
    ]


def test_runtime_grid_cli_explicitly_preserves_legacy_v1_schema(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload["schema_version"] = "mitoevidence.pilot-ablation.v1"
    source = tmp_path / "legacy-v1-state.json"
    target = tmp_path / "legacy-v1-audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target, require_complete=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["input_schema"] == "mitoevidence.pilot-ablation.v1"


def test_runtime_full_grid_still_requires_finalized_status(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload["status"] = "running"
    payload["completed_at_utc"] = None
    source = tmp_path / "running-full.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target, require_complete=True)
    assert completed.returncode == 2, completed.stderr
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["grid_complete"] is True
    assert result["suite_finalized"] is False
    assert result["runtime_complete"] is False


def test_runtime_grid_cli_rejects_declared_expected_count_drift(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload["expected_grid_cells"] = 5
    source = tmp_path / "bad-state.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target)
    assert completed.returncode == 1
    assert "expected_grid_cells 应为 4" in completed.stderr
    assert not target.exists()


def test_cli_rejects_mixed_schema_instead_of_dropping_extra_fields(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload["protocol_id"] = "must-not-be-silently-dropped"
    source = tmp_path / "mixed.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target)
    assert completed.returncode == 1
    assert "protocol_id" in completed.stderr
    assert "Extra inputs are not permitted" in completed.stderr
    assert not target.exists()


@pytest.mark.parametrize(
    ("schema_version", "expected_error"),
    [
        ([], "schema_version 必须是字符串"),
        ({}, "schema_version 必须是字符串"),
        ("unknown.ablation.v9", "不支持的 schema_version"),
    ],
)
def test_cli_rejects_invalid_schema_version_without_traceback(
    tmp_path: Path,
    schema_version,
    expected_error: str,
):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload["schema_version"] = schema_version
    source = tmp_path / "bad-version.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target)
    assert completed.returncode == 1
    assert expected_error in completed.stderr
    assert "Traceback" not in completed.stderr


def test_runtime_without_explicit_schema_version_is_rejected(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload.pop("schema_version")
    source = tmp_path / "runtime-without-version.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target)
    assert completed.returncode == 1
    assert "runtime 输入必须显式声明" in completed.stderr


def test_unversioned_mixed_identity_is_rejected_as_ambiguous(tmp_path: Path):
    payload = _runtime_state(complete=True).model_dump(mode="json")
    payload.pop("schema_version")
    payload["protocol_id"] = "legacy-and-runtime-mixed"
    source = tmp_path / "ambiguous.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target)
    assert completed.returncode == 1
    assert "输入 Schema 混合" in completed.stderr


def test_legacy_complete_grid_with_failed_cell_is_complete(tmp_path: Path):
    records = []
    for arm in ("A", "B", "C", "D"):
        if arm == "D":
            records.append(
                {
                    "question_id": "q1",
                    "arm": arm,
                    "replicate": 1,
                    "outcome": "failed",
                    "failure_type": "Timeout",
                    "failure_reason": "fixture timeout",
                }
            )
        else:
            records.append(
                {
                    "question_id": "q1",
                    "arm": arm,
                    "replicate": 1,
                    "outcome": "succeeded",
                    "run_manifest_sha256": ZERO_HASH,
                    "final_score": 70,
                    "d2_support_precision": 0.8,
                    "d3_evidence_recall": 0.7,
                }
            )
    payload = {
        # Deliberately omit schema_version to exercise backward compatibility.
        "protocol_id": "legacy-complete",
        "question_ids": ["q1"],
        "replicates_per_arm": 1,
        "model_config_sha256": ZERO_HASH,
        "evidence_manifest_sha256": ONE_HASH,
        "rubric_config_sha256": ZERO_HASH,
        "records": records,
    }
    source = tmp_path / "legacy.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run_cli(source, target, require_complete=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["input_schema"] == "mitoevidence.ablation.v1"
    assert result["grid_complete"] is True
    assert result["outcomes"] == {"failed": 1, "succeeded": 3}


def test_print_input_schema_lists_both_strict_contracts(tmp_path: Path):
    target = tmp_path / "schemas.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_ablation_grid.py",
            "--print-input-schema",
            "--output",
            str(target),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rendered = json.loads(target.read_text(encoding="utf-8"))
    schemas = rendered["supported_input_schemas"]
    assert set(schemas) == {
        "mitoevidence.ablation.v1",
            "mitoevidence.pilot-ablation.v1",
            "mitoevidence.pilot-ablation.v2",
            "mitoevidence.pilot-ablation.v3",
        }
    assert schemas["mitoevidence.ablation.v1"]["additionalProperties"] is False
    assert schemas["mitoevidence.pilot-ablation.v1"]["additionalProperties"] is False
    assert schemas["mitoevidence.pilot-ablation.v2"]["additionalProperties"] is False
