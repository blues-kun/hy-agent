"""Filesystem and cross-arm tests for the independent ablation artifact audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ablation import (
    ARM_DEFINITION_BY_ID,
    ARM_DEFINITIONS,
    ABLATION_ARTIFACT_VERSION_V2,
    AblationCellArtifact,
    ClaimGateAudit,
    GeneratorProvenance,
    InputSnapshot,
    JudgeClaimProvenance,
    JudgeProvenance,
    JudgeProvenanceIdentity,
    JudgeSampleBinding,
    PilotAblationRunner,
    PilotAblationSuiteState,
    PilotArm,
    RetrievalAudit,
    SuiteStatus,
    derive_d_review_and_warnings,
)
from app.experiment_retrieval import FROZEN_GRAPH_METHOD, SPARSE_TFIDF_METHOD
from app.schemas import (
    CorpusPassage,
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    SearchPlan,
)
from evaluator.ablation_artifacts import (
    _cross_arm_checks,
    _generator_identity_checks,
    audit_pilot_ablation_artifacts as _audit_pilot_ablation_artifacts,
)
from evaluator.artifact_security import ArtifactSecurityError
from evaluator.judge import JudgeAggregate, JudgeSample, aggregate_samples
from evaluator.schemas import Answerability, JudgeVerdict, SupportVerdict


REPO_ROOT = Path(__file__).resolve().parent.parent
ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64


def audit_pilot_ablation_artifacts(suite: Path) -> dict:
    return _audit_pilot_ablation_artifacts(suite, allow_test_fixture=True)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"


def _model_sha(value) -> str:
    return hashlib.sha256(_json_bytes(value.model_dump(mode="json"))).hexdigest()


def _retrieval(arm: PilotArm, passage: CorpusPassage | None) -> RetrievalAudit:
    if arm is PilotArm.A:
        return RetrievalAudit(
            method="none",
            construction_source="none",
            queries=[],
            source_pmids=[],
            top_k=0,
            candidate_count=0,
            seed_passage_ids=[],
            selected_passage_ids=[],
            expanded_candidate_count=0,
            graph_node_count=0,
            graph_edge_count=0,
            graph_adjacency_edge_count=0,
            graph_lexical_edge_count=0,
        )
    return RetrievalAudit(
        method=SPARSE_TFIDF_METHOD if arm is PilotArm.B else FROZEN_GRAPH_METHOD,
        construction_source=(
            "frozen_corpus_fulltext_only"
            if arm is PilotArm.B
            else "frozen_corpus_text_and_metadata_only"
        ),
        queries=["mitochondrial calcium"],
        source_pmids=[],
        top_k=1,
        candidate_count=1,
        seed_passage_ids=[passage.passage_id],
        selected_passage_ids=[passage.passage_id],
        expanded_candidate_count=1 if arm is not PilotArm.B else 0,
        graph_node_count=1 if arm is not PilotArm.B else 0,
        graph_edge_count=0,
        graph_adjacency_edge_count=0,
        graph_lexical_edge_count=0,
    )


def _suite(
    tmp_path: Path,
    *,
    failed_d: bool = False,
    failure_exc: BaseException | None = None,
) -> tuple[Path, PilotAblationSuiteState]:
    suite = tmp_path / "suite"
    suite.mkdir()
    request = ReviewRequest(question_id="q1", question="What is supported?")
    plan = SearchPlan(
        queries=["mitochondrial calcium"],
        rationale="one frozen plan",
        answerability_hint=Answerability.ANSWERABLE,
    )
    plan_hash = _model_sha(plan)
    identity = {
        "provider": "test-fixture",
        "model": "deterministic-generator",
        "endpoint_origin": "test://local",
        "config_sha256": "9" * 64,
    }
    direct_audit = ModelCallAudit(stage="ablation_A_direct", **identity)
    plan_audit = ModelCallAudit(stage="plan", **identity)
    synthesis_audit = ModelCallAudit(stage="synthesis", **identity)
    passage = CorpusPassage(
        passage_id="p1",
        paper_id="PMID:1",
        pmid="1",
        pmcid="PMC1",
        text="A sufficiently detailed frozen evidence passage.",
        source_path="PMC1.xml",
        source_sha256=ZERO_HASH,
    )
    grounded_review = GeneratedReview(
        answerability=Answerability.ANSWERABLE,
        answer="Supported answer.",
        claims=[
            GeneratedClaim(
                claim_id="C1",
                text="Supported claim.",
                evidence_passage_ids=[passage.passage_id],
            )
        ],
    )
    a = AblationCellArtifact(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        arm=PilotArm.A,
        arm_definition=ARM_DEFINITION_BY_ID[PilotArm.A],
        question_id="q1",
        replicate=1,
        request=request,
        evidence_manifest_path="evidence.json",
        evidence_manifest_sha256=ONE_HASH,
        passages=[],
        retrieval=_retrieval(PilotArm.A, None),
        review=GeneratedReview(
            answerability=Answerability.ANSWERABLE,
            answer="Direct answer.",
            claims=[GeneratedClaim(claim_id="A1", text="Direct ungrounded claim.")],
        ),
        model_calls=[direct_audit],
    )
    b = AblationCellArtifact(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        arm=PilotArm.B,
        arm_definition=ARM_DEFINITION_BY_ID[PilotArm.B],
        question_id="q1",
        replicate=1,
        request=request,
        shared_plan=plan,
        shared_plan_sha256=plan_hash,
        evidence_manifest_path="evidence.json",
        evidence_manifest_sha256=ONE_HASH,
        passages=[passage],
        retrieval=_retrieval(PilotArm.B, passage),
        review=grounded_review,
        model_calls=[plan_audit, synthesis_audit],
    )
    c = AblationCellArtifact(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        arm=PilotArm.C,
        arm_definition=ARM_DEFINITION_BY_ID[PilotArm.C],
        question_id="q1",
        replicate=1,
        request=request,
        shared_plan=plan,
        shared_plan_sha256=plan_hash,
        evidence_manifest_path="evidence.json",
        evidence_manifest_sha256=ONE_HASH,
        passages=[passage],
        retrieval=_retrieval(PilotArm.C, passage),
        review=grounded_review,
        model_calls=[plan_audit, synthesis_audit],
    )
    verdict = JudgeVerdict(
        claim_id="C1",
        verdict=SupportVerdict.FULLY_SUPPORTED,
        confidence=1.0,
        reason="fixture supported",
        evidence_span_refs=[passage.passage_id],
    )
    judge_sample = JudgeSample(
        index=0,
        ok=True,
        verdict=verdict,
        parse_source="test_fixture",
        response_sha256="2" * 64,
        temperature=0.7,
        seed=100,
    )
    aggregate = aggregate_samples(
        "C1",
        [judge_sample],
        k=1,
        min_agreement_votes=1,
        escalate_on_refuted=True,
    )
    sample_binding = JudgeSampleBinding.from_sample(aggregate.samples[0])
    gates = [ClaimGateAudit(claim_id="C1", passed=True, aggregate=aggregate)]
    d_review, d_warnings = derive_d_review_and_warnings(c, gates, judge_k=1)
    d = AblationCellArtifact(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        arm=PilotArm.D,
        arm_definition=ARM_DEFINITION_BY_ID[PilotArm.D],
        question_id="q1",
        replicate=1,
        request=request,
        shared_plan=plan,
        shared_plan_sha256=plan_hash,
        evidence_manifest_path="evidence.json",
        evidence_manifest_sha256=ONE_HASH,
        passages=c.passages,
        retrieval=c.retrieval,
        review=d_review,
        model_calls=c.model_calls,
        claim_gates=gates,
        judge_provenance=JudgeProvenance(
            execution_kind="test_fixture",
            provider="test-fixture",
            model="deterministic-test-gate",
            endpoint_origin="test://local",
            endpoint_url="test://local/chat/completions",
            config_sha256="3" * 64,
            config_hash_scope="source_file_bytes",
            schema_sha256="4" * 64,
            prompt_template_sha256="5" * 64,
            structured_output_channel="function_calling",
            k=1,
            temperature=0.7,
            base_seed=100,
            min_agreement_votes=1,
            escalate_on_refuted=True,
            execution_status="test_fixture_invoked",
            calls=[
                JudgeClaimProvenance(
                    claim_id="C1",
                    prompt_sha256="6" * 64,
                    samples=[sample_binding],
                )
            ],
        ),
        parent_c_artifact_sha256=_model_sha(c),
        warnings=d_warnings,
    )

    records = [PilotAblationRunner._write_cell(suite, artifact) for artifact in (a, b, c)]
    if failed_d:
        records.append(
            PilotAblationRunner._write_failure(
                suite,
                question_id="q1",
                replicate=1,
                arm=PilotArm.D,
                exc=failure_exc or RuntimeError("fixture D failure"),
                schema_version=ABLATION_ARTIFACT_VERSION_V2,
            )
        )
    else:
        records.append(PilotAblationRunner._write_cell(suite, d))
    state = PilotAblationSuiteState(
        schema_version=ABLATION_ARTIFACT_VERSION_V2,
        suite_id="fixture-suite",
        status=SuiteStatus.COMPLETED,
        created_at_utc="2026-08-30T00:00:00+00:00",
        completed_at_utc="2026-08-30T00:01:00+00:00",
        input_snapshot=InputSnapshot(
            path="pilot.jsonl",
            sha256=ZERO_HASH,
            question_ids=["q1"],
        ),
        evidence_manifest_path="evidence.json",
        evidence_manifest_sha256=ONE_HASH,
        arm_definitions=list(ARM_DEFINITIONS),
        replicates=1,
        top_k=1,
        judge_k=1,
        expected_grid_cells=4,
        records=records,
    )
    state_bytes = _json_bytes(state.model_dump(mode="json"))
    (suite / "suite_state.json").write_bytes(state_bytes)
    (suite / "suite_summary.json").write_bytes(state_bytes)
    return suite, state


def _write_state(suite: Path, payload: dict) -> None:
    (suite / "suite_state.json").write_bytes(_json_bytes(payload))


def _rewrite_artifact(suite: Path, state: PilotAblationSuiteState, arm: str, mutate) -> None:
    record = next(record for record in state.records if record.arm.value == arm)
    cell = suite / record.cell_dir
    artifact_path = cell / "artifact.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(payload)
    artifact_data = _json_bytes(payload)
    artifact_path.write_bytes(artifact_data)
    manifest_path = cell / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["artifact.json"] = {
        "bytes": len(artifact_data),
        "sha256": hashlib.sha256(artifact_data).hexdigest(),
    }
    manifest_data = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_data)
    state_payload = state.model_dump(mode="json")
    for row in state_payload["records"]:
        if row["arm"] == arm:
            row["cell_manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()
    _write_state(suite, state_payload)


def test_valid_success_suite_passes_file_and_cross_arm_audit(tmp_path: Path):
    suite, _ = _suite(tmp_path)
    result = audit_pilot_ablation_artifacts(suite)
    assert result["ok"] is False
    assert result["artifact_integrity_ok"] is True
    assert result["production_ready"] is False
    assert result["non_production"] is True
    assert result["test_fixture_audit_ok"] is True
    assert result["records"] == {
        "total": 4,
        "succeeded": 4,
        "failed": 0,
        "audited_cells": 4,
    }
    assert all(cell["ok"] for cell in result["cell_results"])
    assert all(check["ok"] for check in result["cross_arm_checks"])


def test_explicit_failed_cell_is_valid_when_failure_json_matches(tmp_path: Path):
    suite, _ = _suite(tmp_path, failed_d=True)
    result = audit_pilot_ablation_artifacts(suite)
    assert result["test_fixture_audit_ok"] is True
    assert result["records"]["failed"] == 1
    failed = next(cell for cell in result["cell_results"] if cell["outcome"] == "failed")
    assert failed["ok"] is True


def test_write_failure_redacts_secrets_before_state_and_artifact_audit(tmp_path: Path):
    api_key = "sk-" + "abcdefghijklmnopqrstuvwx"
    bearer = "bearer-secret-abcdefghijklmnopqrstuvwxyz"
    opaque = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl"
    leaked_url = "https://alice:password@example.invalid/v1/run?token=secret#private"
    exc = RuntimeError(
        f"api_key={api_key} Authorization: Bearer {bearer} "
        f"url={leaked_url} opaque={opaque} reasoning_content: tiny private chain"
    )
    suite, state = _suite(
        tmp_path,
        failed_d=True,
        failure_exc=exc,
    )
    record = next(record for record in state.records if record.arm is PilotArm.D)
    payload = json.loads(
        (suite / record.cell_dir / "failure.json").read_text(encoding="utf-8")
    )
    rendered = json.dumps(payload, ensure_ascii=False)
    for secret in (
        api_key,
        bearer,
        opaque,
        "alice",
        "password",
        "token=secret",
        "private",
        "tiny private chain",
    ):
        assert secret not in rendered
        assert secret not in record.failure_reason
    assert "[REDACTED]" in payload["failure_reason"]
    assert payload["security"] == {
        "contains_api_key": False,
        "contains_reasoning_content": False,
        "failure_text_sanitized": True,
        "redaction_policy": "mitoevidence.failure-redaction.v1",
    }
    result = audit_pilot_ablation_artifacts(suite)
    assert result["test_fixture_audit_ok"] is True
    assert not any(
        error["code"] == "FAILURE_SECRET_PATTERN_DETECTED"
        for error in result["errors"]
    )


def test_path_traversal_is_rejected_without_reading_outside_suite(tmp_path: Path):
    suite, state = _suite(tmp_path)
    payload = state.model_dump(mode="json")
    payload["records"][0]["cell_dir"] = "../outside"
    _write_state(suite, payload)
    result = audit_pilot_ablation_artifacts(suite)
    assert result["ok"] is False
    assert any(error["code"] == "CELL_PATH_TRAVERSAL" for error in result["errors"])


def test_manifest_size_hash_and_suite_record_hash_are_all_checked(tmp_path: Path):
    suite, state = _suite(tmp_path)
    record = state.records[0]
    artifact = suite / record.cell_dir / "artifact.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    manifest = suite / record.cell_dir / "manifest.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["files"]["review.json"]["bytes"] += 1
    manifest.write_bytes(_json_bytes(manifest_payload))
    result = audit_pilot_ablation_artifacts(suite)
    codes = {error["code"] for error in result["errors"]}
    assert "SUITE_RECORD_MANIFEST_HASH_MISMATCH" in codes
    assert "ARTIFACT_FILE_HASH_MISMATCH" in codes
    assert "ARTIFACT_FILE_SIZE_MISMATCH" in codes


def test_failure_json_must_match_failed_suite_record(tmp_path: Path):
    suite, state = _suite(tmp_path, failed_d=True)
    record = next(record for record in state.records if record.arm is PilotArm.D)
    path = suite / record.cell_dir / "failure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["failure_reason"] = "silently changed"
    path.write_bytes(_json_bytes(payload))
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "FAILURE_ARTIFACT_RECORD_MISMATCH"
        for error in result["errors"]
    )


def test_failure_auditor_does_not_trust_contains_api_key_declaration(tmp_path: Path):
    suite, state = _suite(tmp_path, failed_d=True)
    record = next(record for record in state.records if record.arm is PilotArm.D)
    path = suite / record.cell_dir / "failure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["failure_reason"] = (
        "Authorization: " + "Bearer " + "raw-secret-abcdefghijklmnop"
    )
    payload["security"]["contains_api_key"] = False
    path.write_bytes(_json_bytes(payload))
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "FAILURE_SECRET_PATTERN_DETECTED"
        for error in result["errors"]
    )

    payload["failure_reason"] = "reasoning_content: short private thought"
    payload["security"]["contains_reasoning_content"] = False
    path.write_bytes(_json_bytes(payload))
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "FAILURE_SECRET_PATTERN_DETECTED"
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    ("arm", "mutate"),
    [
        (
            "A",
            lambda payload: payload["review"].update(
                answer="Authorization: "
                + "Bearer "
                + "fixture-secret-abcdefghijklmnop"
            ),
        ),
        (
            "A",
            lambda payload: payload["request"].update(
                scope="https://alice:password@example.invalid/review?token=secret"
            ),
        ),
        (
            "B",
            lambda payload: payload["passages"][0].update(
                text="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            ),
        ),
        (
            "A",
            lambda payload: payload["warnings"].append(
                "reasoning_content: short private thought"
            ),
        ),
    ],
)
def test_success_writer_fails_closed_before_persisting_sensitive_payload(
    tmp_path: Path,
    arm: str,
    mutate,
):
    suite, state = _suite(tmp_path)
    record = next(row for row in state.records if row.arm.value == arm)
    payload = json.loads(
        (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
    )
    mutate(payload)
    artifact = AblationCellArtifact.model_validate(payload)
    destination = tmp_path / "sensitive-success"
    destination.mkdir()

    with pytest.raises(ArtifactSecurityError):
        PilotAblationRunner._write_cell(destination, artifact)

    assert not (destination / artifact.question_id).exists()


def test_success_auditor_scans_actual_files_instead_of_trusting_manifest_flag(
    tmp_path: Path,
):
    suite, state = _suite(tmp_path)

    def inject_sensitive_success_text(payload):
        payload["warnings"].append(
            "Authorization: "
            + "Bearer "
            + "abcdefghijklmnopqrstuvwxyz; "
            "https://alice:password@example.invalid/run?token=secret; "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghij; "
            "reasoning_content: short private thought"
        )

    _rewrite_artifact(suite, state, "A", inject_sensitive_success_text)
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "SUCCESS_FILE_SENSITIVE_MATERIAL_DETECTED"
        for error in result["errors"]
    )
    cell = next(row for row in result["cell_results"] if row["cell"].endswith(":A"))
    assert cell["security_scans"]["artifact.json"]["ok"] is False


def test_formal_status_is_schema_frozen_and_audit_reports_runtime_constant(
    tmp_path: Path,
):
    suite, state = _suite(tmp_path)
    state_payload = state.model_dump(mode="json")
    state_payload["formal_status"] = "formally_passed"
    with pytest.raises(ValueError, match="pilot_ablation_generation_unscored"):
        PilotAblationSuiteState.model_validate(state_payload)

    record = next(row for row in state.records if row.arm is PilotArm.A)
    artifact_payload = json.loads(
        (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
    )
    artifact_payload["formal_status"] = "formally_passed"
    with pytest.raises(ValueError, match="pilot_ablation_generation_unscored"):
        AblationCellArtifact.model_validate(artifact_payload)

    result = audit_pilot_ablation_artifacts(suite)
    assert result["formal_status"] == "pilot_ablation_generation_unscored"
    assert result["formal_status_source"] == "runtime_constant_by_input_schema"


@pytest.mark.parametrize(
    ("model", "endpoint_origin", "endpoint_url"),
    [
        (
            "not-hy3",
            "https://tokenhub.tencentmaas.com",
            "https://tokenhub.tencentmaas.com/v1/chat/completions",
        ),
        (
            "hy3",
            "https://evil.example",
            "https://evil.example/v1/chat/completions",
        ),
    ],
)
def test_generator_production_identity_uses_shared_model_and_host_allowlist(
    tmp_path: Path,
    model: str,
    endpoint_origin: str,
    endpoint_url: str,
):
    suite, state = _suite(tmp_path)
    artifacts = {}
    for record in state.records:
        if record.arm not in {PilotArm.A, PilotArm.B, PilotArm.C}:
            continue
        payload = json.loads(
            (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
        )
        for call in payload["model_calls"]:
            call.update(
                provider="tencent-tokenhub",
                model=model,
                endpoint_origin=endpoint_origin,
                endpoint_url=endpoint_url,
            )
        artifacts[(record.question_id, record.replicate, record.arm)] = (
            AblationCellArtifact.model_validate(payload)
        )

    v3_state = state.model_copy(
        update={
            "schema_version": "mitoevidence.pilot-ablation.v3",
            "generator_provenance": GeneratorProvenance(
                execution_kind="remote_hy3",
                provider="tencent-tokenhub",
                model=model,
                endpoint_origin=endpoint_origin,
                endpoint_url=endpoint_url,
                config_sha256="9" * 64,
                base_seed=20260831,
                cache_namespace="mitoevidence-identity-negative",
            ),
        }
    )
    checks, _identity, _nonformal = _generator_identity_checks(
        v3_state,
        artifacts,
        allow_test_fixture=False,
    )
    identity_check = next(
        row
        for row in checks
        if row["check"] == "GENERATOR_IDENTITY_CONSISTENT_ACROSS_A_B_C"
    )
    assert identity_check["formal_hy3_identity_allowlisted"] is False
    assert identity_check["ok"] is False


@pytest.mark.parametrize(
    ("model", "endpoint_origin", "endpoint_url"),
    [
        (
            "not-hy3",
            "https://tokenhub.tencentmaas.com",
            "https://tokenhub.tencentmaas.com/v1/chat/completions",
        ),
        (
            "hy3",
            "https://evil.example",
            "https://evil.example/v1/chat/completions",
        ),
    ],
)
def test_judge_production_identity_uses_shared_model_and_host_allowlist(
    tmp_path: Path,
    model: str,
    endpoint_origin: str,
    endpoint_url: str,
):
    suite, state = _suite(tmp_path)
    artifacts = {}
    artifact_hashes = {}
    for record in state.records:
        payload = json.loads(
            (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
        )
        if record.arm is PilotArm.D:
            payload["judge_provenance"].update(
                execution_kind="remote_hy3",
                provider="tencent-tokenhub",
                model=model,
                endpoint_origin=endpoint_origin,
                endpoint_url=endpoint_url,
                execution_status="remote_invoked",
            )
        artifact = AblationCellArtifact.model_validate(payload)
        key = (record.question_id, record.replicate, record.arm)
        artifacts[key] = artifact
        artifact_hashes[key] = hashlib.sha256(
            (suite / record.cell_dir / "artifact.json").read_bytes()
        ).hexdigest()

    d_artifact = artifacts[("q1", 1, PilotArm.D)]
    assert d_artifact.judge_provenance is not None
    identity_fields = JudgeProvenanceIdentity.model_fields
    v3_state = state.model_copy(
        update={
            "schema_version": "mitoevidence.pilot-ablation.v3",
            "judge_provenance_identity": JudgeProvenanceIdentity.model_validate(
                {
                    field: getattr(d_artifact.judge_provenance, field)
                    for field in identity_fields
                }
            ),
        }
    )

    checks = _cross_arm_checks(
        v3_state,
        artifacts,
        artifact_hashes,
        allow_test_fixture=False,
    )
    judge_check = next(
        row for row in checks if row["check"] == "D_JUDGE_PROVENANCE_BINDING"
    )
    assert judge_check["formal_hy3_identity_allowlisted"] is False
    assert judge_check["ok"] is False

def test_d_must_bind_exact_c_snapshot_and_one_gate_per_c_claim(tmp_path: Path):
    suite, state = _suite(tmp_path)

    def mutate(payload):
        payload["parent_c_artifact_sha256"] = "f" * 64
        payload["passages"][0]["text"] = "changed D-only passage"
        payload["retrieval"]["candidate_count"] = 99
        payload["claim_gates"] = []
        payload["judge_provenance"]["calls"] = []
        payload["judge_provenance"]["execution_status"] = "no_claims_no_request"
        payload["review"]["answerability"] = "insufficient"
        payload["review"]["answer"] = (
            "C 草稿中的主张均未通过自动 Claim—Evidence Judge 门控，"
            "因此不保留科学结论。"
        )
        payload["review"]["claims"] = []
        payload["review"]["limitations"][-1] = "门控保留 0/0 条主张。"

    _rewrite_artifact(suite, state, "D", mutate)
    result = audit_pilot_ablation_artifacts(suite)
    check = next(
        item
        for item in result["cross_arm_checks"]
        if item["check"] == "D_EXACT_C_ARTIFACT_BINDING"
    )
    assert check["ok"] is False
    assert check["parent_c_canonical_sha256_matches"] is None
    assert check["parent_c_manifest_file_sha256_matches"] is False
    assert check["passages_identical"] is False
    assert check["retrieval_identical"] is False
    assert check["gate_count_equals_c_claim_count"] is False


def test_d_parent_must_equal_c_manifest_raw_artifact_hash(tmp_path: Path):
    suite, state = _suite(tmp_path)
    c_record = next(record for record in state.records if record.arm is PilotArm.C)
    cell = suite / c_record.cell_dir
    artifact_path = cell / "artifact.json"
    # Reformat only: legacy v2 is intentionally audited through the exact raw
    # bytes authenticated by C's manifest, never a model reserialization.
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode() + b"\n"
    artifact_path.write_bytes(artifact_data)
    manifest_path = cell / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["artifact.json"] = {
        "bytes": len(artifact_data),
        "sha256": hashlib.sha256(artifact_data).hexdigest(),
    }
    manifest_data = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_data)
    state_payload = state.model_dump(mode="json")
    for row in state_payload["records"]:
        if row["arm"] == "C":
            row["cell_manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()
    _write_state(suite, state_payload)

    result = audit_pilot_ablation_artifacts(suite)
    check = next(
        item
        for item in result["cross_arm_checks"]
        if item["check"] == "D_EXACT_C_ARTIFACT_BINDING"
    )
    assert check["parent_c_canonical_sha256_matches"] is None
    assert check["parent_c_manifest_file_sha256_matches"] is False
    assert check["ok"] is False


def test_d_schema_rejects_review_claims_not_equal_to_passed_gate_ids(tmp_path: Path):
    suite, state = _suite(tmp_path)
    record = next(record for record in state.records if record.arm is PilotArm.D)
    payload = json.loads(
        (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
    )
    payload["review"]["claims"] = []
    payload["review"]["answerability"] = "insufficient"
    payload["review"]["answer"] = (
        "C 草稿中的主张均未通过自动 Claim—Evidence Judge 门控，"
        "因此不保留科学结论。"
    )
    with pytest.raises(ValueError, match="passed gates"):
        AblationCellArtifact.model_validate(payload)


def test_d_cross_audit_rejects_coordinated_identity_and_derivation_drift(
    tmp_path: Path,
):
    suite, state = _suite(tmp_path)

    def mutate(payload):
        payload["request"]["scope"] = "D-only request drift"
        payload["shared_plan"]["rationale"] = "D-only plan drift"
        payload["shared_plan_sha256"] = _model_sha(
            SearchPlan.model_validate(payload["shared_plan"])
        )
        payload["model_calls"][0]["model"] = "D-only-model"
        payload["evidence_manifest_path"] = "D-only-evidence.json"
        payload["evidence_manifest_sha256"] = "f" * 64
        payload["review"]["claims"][0]["text"] = "Forged D-only claim text."
        payload["review"]["answer"] = (
            "经自动 Claim—Evidence Judge 门控保留的主张：\n"
            "- Forged D-only claim text."
        )
        payload["review"]["limitations"].insert(0, "D-only limitation drift")
        payload["warnings"].insert(0, "D-only warning drift")

    _rewrite_artifact(suite, state, "D", mutate)
    result = audit_pilot_ablation_artifacts(suite)
    check = next(
        item
        for item in result["cross_arm_checks"]
        if item["check"] == "D_EXACT_C_ARTIFACT_BINDING"
    )
    assert check["ok"] is False
    assert check["request_identical"] is False
    assert check["shared_plan_identical"] is False
    assert check["shared_plan_sha256_identical"] is False
    assert check["model_calls_identical"] is False
    assert check["evidence_snapshot_identical"] is False
    assert check["review_claims_exact_passed_c_subset"] is False
    assert check["answer_deterministic"] is False
    assert check["limitations_deterministic"] is False
    assert check["warnings_deterministic"] is False


def test_b_c_plan_hash_and_top_k_must_match_frozen_suite_policy(tmp_path: Path):
    suite, state = _suite(tmp_path)

    def mutate(payload):
        payload["shared_plan"]["rationale"] = "drifted plan"
        plan = SearchPlan.model_validate(payload["shared_plan"])
        payload["shared_plan_sha256"] = _model_sha(plan)
        payload["retrieval"]["top_k"] = 2

    _rewrite_artifact(suite, state, "B", mutate)
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        not check["ok"] and check["check"] == "B_C_D_SHARED_PLAN_HASH_ACROSS_REPLICATES"
        for check in result["cross_arm_checks"]
    )
    assert any(
        not check["ok"]
        and check["check"] == "B_C_TOP_K_BUDGET"
        and check["scope"].endswith(":B")
        for check in result["cross_arm_checks"]
    )


def test_d_requires_judge_provenance_and_matches_suite_judge_k(tmp_path: Path):
    suite, state = _suite(tmp_path)

    def remove_provenance(payload):
        payload.pop("judge_provenance")

    _rewrite_artifact(suite, state, "D", remove_provenance)
    missing = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "D_JUDGE_PROVENANCE_MISSING"
        for error in missing["errors"]
    )

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_suite, second_state = _suite(second_root)
    state_payload = second_state.model_dump(mode="json")
    state_payload["judge_k"] = 2
    _write_state(second_suite, state_payload)
    mismatch = audit_pilot_ablation_artifacts(second_suite)
    assert any(
        error["code"] == "D_JUDGE_K_SUITE_MISMATCH"
        for error in mismatch["errors"]
    )


def test_provenance_requirement_is_explicitly_versioned_v2(tmp_path: Path):
    suite, state = _suite(tmp_path)
    record = next(record for record in state.records if record.arm is PilotArm.D)
    payload = json.loads(
        (suite / record.cell_dir / "artifact.json").read_text(encoding="utf-8")
    )
    payload.pop("judge_provenance")
    payload["schema_version"] = "mitoevidence.pilot-ablation.v1"
    legacy = AblationCellArtifact.model_validate(payload)
    assert legacy.judge_provenance is None

    payload["schema_version"] = "mitoevidence.pilot-ablation.v2"
    with pytest.raises(ValueError, match="Judge provenance"):
        AblationCellArtifact.model_validate(payload)


def test_judge_provenance_rejects_forged_aggregate_even_when_hashes_are_rewritten(
    tmp_path: Path,
):
    suite, state = _suite(tmp_path)

    def forge_aggregate(payload):
        payload["claim_gates"][0]["aggregate"]["n_valid"] = 999
        payload["claim_gates"][0]["aggregate"]["votes"] = {
            "fully_supported": 999
        }

    _rewrite_artifact(suite, state, "D", forge_aggregate)
    result = audit_pilot_ablation_artifacts(suite)
    assert any(
        error["code"] == "CELL_ARTIFACT_SCHEMA_INVALID"
        and "重算" in error["detail"]
        for error in result["errors"]
    )


def test_out_of_scope_claims_are_common_schema_and_artifact_safety_failure(tmp_path: Path):
    with pytest.raises(ValueError, match="claims 必须为空"):
        GeneratedReview(
            answerability=Answerability.OUT_OF_SCOPE,
            answer="refusal",
            claims=[GeneratedClaim(claim_id="C1", text="unsafe residual claim")],
        )

    suite, state = _suite(tmp_path)

    def mutate(payload):
        payload["review"]["answerability"] = "out_of_scope"
        payload["review"]["answer"] = "refusal with unsafe residual claim"

    _rewrite_artifact(suite, state, "A", mutate)
    result = audit_pilot_ablation_artifacts(suite)
    assert result["safety"]["review_boundary_ok"] is False
    assert any(
        error["code"] == "OUT_OF_SCOPE_CLAIMS_NONEMPTY"
        for error in result["errors"]
    )


def test_artifact_audit_cli_writes_report_and_returns_two_on_failure(tmp_path: Path):
    suite, state = _suite(tmp_path)
    record = state.records[0]
    artifact = suite / record.cell_dir / "artifact.json"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_pilot_ablation_artifacts.py",
            "--suite-dir",
            str(suite),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "mitoevidence.pilot-ablation-artifact-audit.v3"
    assert report["audit_kind"] == "artifact_level_filesystem_and_cross_arm"
    assert report["ok"] is False


def test_cli_requires_explicit_test_fixture_mode_and_never_marks_it_production(
    tmp_path: Path,
):
    suite, _ = _suite(tmp_path)
    default_output = tmp_path / "default.json"
    denied = subprocess.run(
        [
            sys.executable,
            "scripts/audit_pilot_ablation_artifacts.py",
            "--suite-dir",
            str(suite),
            "--output",
            str(default_output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert denied.returncode == 2

    fixture_output = tmp_path / "fixture.json"
    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_pilot_ablation_artifacts.py",
            "--suite-dir",
            str(suite),
            "--output",
            str(fixture_output),
            "--allow-test-fixture",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert allowed.returncode == 0, allowed.stderr
    report = json.loads(fixture_output.read_text(encoding="utf-8"))
    assert report["audit_mode"] == "allow_test_fixture"
    assert report["test_fixture_audit_ok"] is True
    assert report["non_production"] is True
    assert report["production_ready"] is False
    assert report["ok"] is False
