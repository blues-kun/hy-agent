"""Offline tests for fail-closed formal experiment orchestration."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ablation import (
    ARM_DEFINITIONS,
    GeneratorProvenance,
    InputSnapshot,
    JudgeProvenanceIdentity,
    PilotAblationSuiteState,
    SuiteStatus,
)
from evaluator.experiment_protocol import (
    ABLATION_RUNTIME_SCHEMA_VERSIONS,
    ABLATION_SCHEMA_VERSION,
    AblationInput,
    AblationRunRecord,
    ExpertConcordanceInput,
    ExperimentStage,
    ReadinessStatus,
    ablation_grid_audit,
    analyze_expert_concordance,
    build_experiment_preflight,
    build_pilot_answerability_concordance,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    pilot = root / "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
    pilot.parent.mkdir(parents=True)
    pilot.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"PILOT-{index:02d}",
                    "question": f"question {index}",
                    "answerability": "answerable",
                }
            )
            + "\n"
            for index in range(1, 6)
        ),
        encoding="utf-8",
    )
    xml = root / "eval/data/corpus_raw/PMC1.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text("<article><body><p>evidence</p></body></article>", encoding="utf-8")
    digest = hashlib.sha256(xml.read_bytes()).hexdigest()
    manifest = root / "eval/data/evidence_pool_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fulltext": [
                    {
                        "pmid": "1",
                        "path": "eval/data/corpus_raw/PMC1.xml",
                        "sha256": digest,
                    },
                    {"pmid": "2", "error": "no lawful full text"},
                ]
            }
        ),
        encoding="utf-8",
    )
    for relative in (
        "scripts/run_pilot_suite.py",
        "app/hy3_review.py",
        "scripts/run_pilot_ablation.py",
        "app/ablation.py",
        "app/experiment_retrieval.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return root


def _stages(report):
    return {stage.stage: stage for stage in report.stages}


def test_preflight_can_ready_the_real_pilot_but_refuses_to_invent_other_inputs(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    secret = "must-never-appear-in-report"
    report = build_experiment_preflight(
        root,
        environment={"HY3_API_KEY": secret},
        expert_reference_paths=[
            "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
        ],
    )
    stages = _stages(report)
    assert stages[ExperimentStage.REAL_HY3_PILOT].status is ReadinessStatus.READY
    assert (
        stages[ExperimentStage.EXPERT_REFERENCE_CONCORDANCE].status
        is ReadinessStatus.BLOCKED
    )
    assert stages[ExperimentStage.DISCRIMINATION].status is ReadinessStatus.BLOCKED
    assert stages[ExperimentStage.STABILITY].status is ReadinessStatus.BLOCKED
    assert stages[ExperimentStage.ADVERSARIAL].status is ReadinessStatus.BLOCKED
    assert stages[ExperimentStage.ABLATION_ABCD].status is ReadinessStatus.READY
    rendered = report.model_dump_json()
    assert secret not in rendered
    assert report.safety.model_dump() == {
        "network_calls_performed": False,
        "contains_api_key": False,
        "mutates_experiment_inputs": False,
    }


def test_preflight_blocks_real_run_on_missing_key_and_corpus_drift(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    manifest_path = root / "eval/data/evidence_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fulltext"][0]["sha256"] = ZERO_HASH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = build_experiment_preflight(
        root,
        environment={},
        expert_reference_paths=[
            "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
        ],
    )
    pilot = _stages(report)[ExperimentStage.REAL_HY3_PILOT]
    assert pilot.status is ReadinessStatus.BLOCKED
    failures = {check.code for check in pilot.checks if not check.passed}
    assert failures == {"HY3_API_KEY_PRESENT", "FROZEN_CORPUS_INTEGRITY"}
    assert pilot.next_command is None


@pytest.mark.parametrize("schema_version", ABLATION_RUNTIME_SCHEMA_VERSIONS)
def test_preflight_accepts_runtime_ablation_state_without_legacy_projection(
    tmp_path: Path,
    schema_version: str,
):
    root = _fixture_repo(tmp_path)
    v3_provenance = (
        {
            "generator_provenance": GeneratorProvenance(
                execution_kind="test_fixture",
                provider="test-fixture",
                model="fake-hy3",
                endpoint_origin="https://fixture.invalid",
                endpoint_url="https://fixture.invalid/v1/chat/completions",
                config_sha256=ZERO_HASH,
                base_seed=101,
                cache_namespace="mitoevidence-fixture-v3",
            ),
            "judge_provenance_identity": JudgeProvenanceIdentity(
                execution_kind="test_fixture",
                provider="test-fixture",
                model="fake-judge",
                endpoint_origin="https://fixture.invalid",
                endpoint_url="https://fixture.invalid/v1/chat/completions",
                config_sha256=ZERO_HASH,
                config_hash_scope="source_file_bytes",
                schema_sha256=ZERO_HASH,
                prompt_template_sha256=ZERO_HASH,
                structured_output_channel="function_calling",
                k=1,
                temperature=0.7,
                base_seed=202,
                min_agreement_votes=1,
                escalate_on_refuted=True,
            ),
        }
        if schema_version == "mitoevidence.pilot-ablation.v3"
        else {}
    )
    state = PilotAblationSuiteState(
        schema_version=schema_version,
        suite_id="runtime-preflight",
        status=SuiteStatus.RUNNING,
        created_at_utc="2026-08-31T00:00:00+00:00",
        input_snapshot=InputSnapshot(
            path="pilot.jsonl",
            sha256=ZERO_HASH,
            question_ids=["q1"],
        ),
        evidence_manifest_path="eval/data/evidence_pool_manifest.json",
        evidence_manifest_sha256=hashlib.sha256(
            (root / "eval/data/evidence_pool_manifest.json").read_bytes()
        ).hexdigest(),
        arm_definitions=list(ARM_DEFINITIONS),
        replicates=1,
        top_k=2,
        judge_k=1,
        expected_grid_cells=4,
        records=[],
        **v3_provenance,
    )
    source = tmp_path / f"suite-state-{schema_version.rsplit('.', 1)[-1]}.json"
    source.write_text(state.model_dump_json(), encoding="utf-8")
    report = build_experiment_preflight(
        root,
        environment={"HY3_API_KEY": "present-but-never-rendered"},
        expert_reference_paths=[
            "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
        ],
        ablation_input=source,
    )
    checks = {
        check.code: check
        for check in _stages(report)[ExperimentStage.ABLATION_ABCD].checks
    }
    assert checks["ABLATION_GRID_INPUT"].passed is True
    assert schema_version in checks["ABLATION_GRID_INPUT"].detail
    assert checks["ABLATION_GRID_COMPLETE"].passed is False
    assert f"input_schema={schema_version}" in checks["ABLATION_GRID_COMPLETE"].detail


def test_single_expert_concordance_is_role_explicit_and_hand_checkable():
    payload = {
        "reference_authority": "user_declared_expert",
        "reference_manifest_sha256": ZERO_HASH,
        "rubric_config_sha256": ONE_HASH,
        "nominal": [
            {
                "item_id": "n1",
                "task": "answerability",
                "expert_label": "partial",
                "automatic_label": "partial",
            },
            {
                "item_id": "n2",
                "task": "answerability",
                "expert_label": "answerable",
                "automatic_label": None,
                "automatic_error": "SchemaError: no valid output",
            },
        ],
        "ordinal_0_4": [
            {"item_id": "x1", "dimension": "D2", "expert_score": 0, "automatic_score": 0},
            {"item_id": "x2", "dimension": "D2", "expert_score": 2, "automatic_score": 1},
            {"item_id": "x3", "dimension": "D2", "expert_score": 4, "automatic_score": 4},
        ],
        "total_scores": [
            {"item_id": "x1", "expert_score": 10, "automatic_score": 12},
            {"item_id": "x2", "expert_score": 50, "automatic_score": 48},
            {"item_id": "x3", "expert_score": 90, "automatic_score": 88},
        ],
    }
    result = analyze_expert_concordance(ExpertConcordanceInput.model_validate(payload))
    assert result["nominal_tasks"]["answerability"]["n_input"] == 2
    assert result["nominal_tasks"]["answerability"]["n_complete"] == 1
    assert result["nominal_tasks"]["answerability"]["automatic_errors"] == {
        "n2": "SchemaError: no valid output"
    }
    assert result["dimensions"]["D2"]["raw_agreement"] == pytest.approx(2 / 3)
    assert result["dimensions"]["D2"]["role_a"] == "expert_reference"
    assert result["dimensions"]["D2"]["role_b"] == "automatic_evaluator"
    assert result["total_scores"]["spearman_rho"] == pytest.approx(1.0)
    assert result["total_scores"]["mean_absolute_error"] == pytest.approx(2.0)
    assert "not expert reliability" in result["total_scores"]["icc_interpretation"]
    assert "does not estimate inter-expert reliability" in result["interpretation"]


def test_expert_concordance_rejects_duplicate_dimension_item():
    row = {"item_id": "x", "dimension": "D2", "expert_score": 2, "automatic_score": 2}
    with pytest.raises(ValidationError, match="必须唯一"):
        ExpertConcordanceInput.model_validate(
            {
                "reference_authority": "expert",
                "reference_manifest_sha256": ZERO_HASH,
                "rubric_config_sha256": ONE_HASH,
                "ordinal_0_4": [row, row],
                "total_scores": [],
            }
        )


def test_real_pilot_suite_can_be_bound_to_all_five_expert_answerability_labels(tmp_path: Path):
    suite = tmp_path / "pilot-real"
    suite.mkdir()
    expert_rows = [
        json.loads(line)
        for line in (
            REPO_ROOT / "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    for expert in expert_rows:
        item_id = expert["question_id"]
        run_dir = suite / item_id
        run_dir.mkdir()
        answerability = expert["answerability"]
        claims = []
        if answerability in {"answerable", "partial"}:
            claims = [{"claim_id": "C1", "text": "fixture claim"}]
        review = {
            "answerability": answerability,
            "answer": "fixture answer",
            "claims": claims,
            "limitations": [],
        }
        review_path = run_dir / "review.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_kind": "hy3",
                    "question_id": item_id,
                    "files": {"review.json": {"sha256": review_sha}},
                }
            ),
            encoding="utf-8",
        )
        records.append({"pilot_id": item_id, "ok": True, "run_dir": item_id})
    summary = {
        "suite_id": "fixture-real",
        "run_kind": "hy3",
        "records": records,
    }
    summary_path = suite / "suite_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    data = build_pilot_answerability_concordance(REPO_ROOT, suite)
    assert len(data.nominal) == 5
    assert all(row.automatic_error is None for row in data.nominal)
    assert all(row.expert_label == row.automatic_label for row in data.nominal)
    assert data.automatic_system_role == "hy3_application_answerability"
    result = analyze_expert_concordance(data)
    assert result["nominal_tasks"]["pilot_answerability"]["raw_agreement"] == 1.0

    summary["run_kind"] = "offline_smoke"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="offline_smoke"):
        build_pilot_answerability_concordance(REPO_ROOT, suite)


def _ablation_record(question: str, arm: str, replicate: int, *, failed: bool = False):
    if failed:
        return {
            "question_id": question,
            "arm": arm,
            "replicate": replicate,
            "outcome": "failed",
            "failure_type": "SchemaError",
            "failure_reason": "model output failed local validation",
        }
    return {
        "question_id": question,
        "arm": arm,
        "replicate": replicate,
        "outcome": "succeeded",
        "run_manifest_sha256": ZERO_HASH,
        "final_score": 70 + replicate,
        "d2_support_precision": 0.8,
        "d3_evidence_recall": 0.7,
    }


def test_ablation_grid_keeps_failures_in_denominator_and_exposes_missing_cells():
    records = [
        _ablation_record(question, arm, replicate, failed=(question, arm, replicate) == ("q1", "C", 2))
        for question in ("q1", "q2")
        for arm in ("A", "B", "C", "D")
        for replicate in (1, 2, 3)
    ]
    data = AblationInput.model_validate(
        {
            "protocol_id": "formal-v1",
            "question_ids": ["q1", "q2"],
            "model_config_sha256": ZERO_HASH,
            "evidence_manifest_sha256": ONE_HASH,
            "rubric_config_sha256": ZERO_HASH,
            "records": records,
        }
    )
    result = ablation_grid_audit(data)
    assert result["expected_grid_cells"] == 24
    assert result["recorded_grid_cells"] == 24
    assert result["grid_complete"] is True
    assert result["outcomes"] == {"failed": 1, "succeeded": 23}
    assert result["by_arm"]["C"]["expected"] == 6
    assert result["by_arm"]["C"]["failed"] == 1
    assert result["by_arm"]["C"]["score_denominator"] == 5

    incomplete = data.model_copy(update={"records": data.records[:-1]})
    audit = ablation_grid_audit(incomplete)
    assert audit["grid_complete"] is False
    assert audit["missing_grid_cells"] == 1
    assert audit["missing"] == [{"question_id": "q2", "arm": "D", "replicate": 3}]


def test_failed_ablation_record_cannot_carry_success_metrics():
    with pytest.raises(ValidationError, match="不得伪造成功指标"):
        AblationRunRecord.model_validate(
            {
                "question_id": "q1",
                "arm": "A",
                "replicate": 1,
                "outcome": "failed",
                "final_score": 100,
                "failure_type": "Timeout",
                "failure_reason": "request timed out",
            }
        )


def test_preflight_cli_prints_all_strict_input_schemas():
    completed = subprocess.run(
        [sys.executable, "scripts/preflight_experiments.py", "--print-input-schemas"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    schemas = json.loads(completed.stdout)
    assert set(schemas) == {"expert_concordance", "validation", "ablation"}
    assert schemas["expert_concordance"]["additionalProperties"] is False
    supported = schemas["ablation"]["supported_input_schemas"]
    assert set(supported) == {
        ABLATION_SCHEMA_VERSION,
        *ABLATION_RUNTIME_SCHEMA_VERSIONS,
    }
    assert all(schema["additionalProperties"] is False for schema in supported.values())


def test_expert_concordance_cli_writes_role_explicit_result(tmp_path: Path):
    payload = {
        "reference_authority": "expert",
        "reference_manifest_sha256": ZERO_HASH,
        "rubric_config_sha256": ONE_HASH,
        "ordinal_0_4": [],
        "total_scores": [],
    }
    source = tmp_path / "input.json"
    target = tmp_path / "output.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_expert_concordance.py",
            "--input",
            str(source),
            "--output",
            str(target),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["schema_version"] == "mitoevidence.expert-concordance.v1"
    assert "does not estimate inter-expert reliability" in result["interpretation"]


def test_ablation_grid_cli_returns_two_only_for_missing_cells(tmp_path: Path):
    payload = {
        "protocol_id": "grid-cli",
        "question_ids": ["q1"],
        "replicates_per_arm": 1,
        "model_config_sha256": ZERO_HASH,
        "evidence_manifest_sha256": ONE_HASH,
        "rubric_config_sha256": ZERO_HASH,
        "records": [_ablation_record("q1", "A", 1)],
    }
    source = tmp_path / "grid.json"
    target = tmp_path / "audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_ablation_grid.py",
            "--input",
            str(source),
            "--output",
            str(target),
            "--require-complete",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["input_schema"] == "mitoevidence.ablation.v1"
    assert result["recorded_grid_cells"] == 1
    assert result["expected_grid_cells"] == 4
    assert result["missing_grid_cells"] == 3
