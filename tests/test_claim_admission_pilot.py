from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from evaluator.pilot_identity import PilotExecutionIdentity

from app.claim_admission_pilot import (
    CLAIM_CACHE_NAMESPACE,
    MODEL_EXPOSED_FIELDS,
    AdmissionDecision,
    BlindClaimInput,
    CellOutcome,
    FORMAL_STATUS,
    ClaimAdmissionSuiteState,
    ClaimCallArtifact,
    ClaimAdmissionPilotRunner,
    ClaimAdmissionVerdict,
    Hy3ClaimAdmissionModel,
    OUTPUT_SCHEMA_SHA256,
    PROMPT_TEMPLATE_SHA256,
    SuiteStatus,
    _binary_keep_candidate_metrics,
    _claim_base_prompt_sha256,
    _redact_failure,
    analyze_claim_admission_pilot,
    select_claim_records,
)
from app.hy3_review import (
    GENERATOR_OUTPUT_HASH_SCOPE,
    GENERATOR_PROMPT_HASH_SCOPE,
    StructuredResult,
    Hy3ReviewModel,
    _json_sha256 as _hy3_json_sha256,
)
from app.schemas import ModelCallAudit
from evaluator.expert_gold import audit_expert_gold, load_expert_gold_records
from evaluator.judge.config import default_judge_config


REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeModel:
    model = "fixture-hy3"
    config_sha256 = "1" * 64
    prompt_template_sha256 = PROMPT_TEMPLATE_SHA256
    output_schema_sha256 = OUTPUT_SCHEMA_SHA256

    def __init__(self, *, decision_by_triple=None, fail_on=None, interrupt_on=None):
        self.decision_by_triple = decision_by_triple or {}
        self.fail_on = set(fail_on or set())
        self.interrupt_on = interrupt_on
        self.calls = []

    def classify(self, item, *, temperature, seed):
        index = len(self.calls)
        self.calls.append((item, temperature, seed))
        if self.interrupt_on is not None and len(self.calls) == self.interrupt_on:
            raise KeyboardInterrupt()
        if index in self.fail_on:
            raise RuntimeError("fixture failure")
        decision = self.decision_by_triple.get(item.triple, AdmissionDecision.ACCEPT)
        verdict = ClaimAdmissionVerdict(
            decision=decision, concise_reason="fixture reason"
        )
        return (
            verdict,
            ModelCallAudit(
                stage="claim_admission_blind",
                provider="offline-fixture",
                model=self.model,
                config_sha256=self.config_sha256,
                schema_sha256=self.output_schema_sha256,
                prompt_sha256="a" * 64,
                base_prompt_sha256=_claim_base_prompt_sha256(item),
                prompt_hash_scope=GENERATOR_PROMPT_HASH_SCOPE,
                structured_output_sha256=_hy3_json_sha256(
                    verdict.model_dump(mode="json")
                ),
                structured_output_hash_scope=GENERATOR_OUTPUT_HASH_SCOPE,
                temperature=temperature,
                requested_seed=seed,
                cache_namespace=CLAIM_CACHE_NAMESPACE,
            ),
        )


def _run(
    tmp_path: Path,
    model: _FakeModel,
    *,
    suite_id: str = "claim-fixture",
    limit: int = 3,
    repeats: int = 1,
    resume: bool = False,
):
    return ClaimAdmissionPilotRunner(model=model).run_suite(
        repo_root=REPO_ROOT,
        out_root=tmp_path / "results",
        suite_id=suite_id,
        limit=limit,
        repeats=repeats,
        selection_seed="fixture-selection",
        temperature=0.2,
        base_seed=123,
        resume=resume,
    )


def _gold_decision_by_triple():
    rows = load_expert_gold_records()["claim_reviews"]
    return {row["triple"]: AdmissionDecision(row["ai_decision"]) for row in rows}


def test_expert_gold_and_hash_fixed_selection_are_valid():
    audit = audit_expert_gold()
    assert audit["ok"] is True
    assert audit["datasets"]["claim_reviews"]["record_count"] == 50
    rows = load_expert_gold_records()["claim_reviews"]
    selected_a = select_claim_records(rows, limit=11, selection_seed="fixed")
    selected_b = select_claim_records(rows, limit=11, selection_seed="fixed")
    assert [row["review_id"] for row in selected_a] == [row["review_id"] for row in selected_b]


def test_runner_exposes_only_whitelisted_blind_fields_and_full_grid(tmp_path: Path):
    model = _FakeModel()
    suite_dir, state = _run(tmp_path, model, limit=4, repeats=2)
    assert state.formal_status == "offline_fixture_nonformal_claim_admission_pilot"
    assert state.execution_identity.execution_kind.value == "offline_fixture"
    assert state.provider == "offline-fixture"
    assert state.status is SuiteStatus.COMPLETED
    assert len(state.records) == state.expected_calls == 8
    assert all(record.outcome is CellOutcome.SUCCEEDED for record in state.records)
    assert len(model.calls) == 8
    for item, _, _ in model.calls:
        assert list(item.model_dump()) == MODEL_EXPOSED_FIELDS
        dumped = json.dumps(item.model_dump(), ensure_ascii=False)
        assert "ai_prelabel_pending_human" not in dumped
        assert "claude-fable" not in dumped
    assert state.safety.expert_decision_available_to_model is False
    assert state.input_snapshot.model_exposed_fields == MODEL_EXPOSED_FIELDS
    assert (suite_dir / "suite_summary.json").read_bytes() == (
        suite_dir / "suite_state.json"
    ).read_bytes()


def test_execution_identity_closes_formal_provider_and_endpoint_contract():
    with pytest.raises(ValidationError):
        PilotExecutionIdentity(
            execution_kind="remote_hy3", provider="offline-fixture",
            model="fixture-hy3", endpoint_origin="", endpoint_url="",
        )


def test_all_failure_fixture_cannot_be_relabelled_as_formal_remote(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakeModel(fail_on={0}), limit=1, repeats=1)
    remote = {
        "execution_kind": "remote_hy3", "provider": "tencent-tokenhub",
        "model": "hy3", "endpoint_origin": "https://tokenhub.tencentmaas.com",
        "endpoint_url": "https://tokenhub.tencentmaas.com/chat/completions",
    }
    for name in ("suite_state.json", "suite_summary.json"):
        path = suite / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(execution_identity=remote, provider="tencent-tokenhub", model="hy3", formal_status=FORMAL_STATUS)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure cell_request|provenance"):
        analyze_claim_admission_pilot(suite, repo_root=REPO_ROOT)
    with pytest.raises(ValidationError):
        PilotExecutionIdentity(
            execution_kind="remote_hy3", provider="tencent-tokenhub", model="hy3",
            endpoint_origin="https://tokenhub.tencentmaas.com",
            endpoint_url="https://user:secret@tokenhub.tencentmaas.com/chat/completions?key=x",
        )


def test_analysis_computes_four_class_system_reference_metrics(tmp_path: Path):
    suite_dir, _ = _run(
        tmp_path,
        _FakeModel(decision_by_triple=_gold_decision_by_triple()),
        limit=50,
        repeats=1,
    )
    result = analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)
    metrics = result["classification"]
    assert metrics["n"] == 50
    assert metrics["raw_accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["gold_distribution"] == {
        "accept": 8,
        "accept_with_edits": 25,
        "reject": 14,
        "uncertain": 3,
    }
    assert all(value["f1"] == 1.0 for value in metrics["per_class"].values())
    baseline = result["baselines"]["four_class_majority"]
    assert baseline["majority_labels"] == ["accept_with_edits"]
    assert baseline["correct"] == 25
    assert baseline["total"] == 50
    assert baseline["accuracy"] == 0.5
    binary = result["binary_keep_candidate"]
    assert binary["counts"] == {
        "true_positive": 33,
        "false_negative": 0,
        "false_positive": 0,
        "true_negative": 17,
    }
    assert binary["metrics"] == {
        "accuracy": 1.0,
        "sensitivity_recall": 1.0,
        "specificity": 1.0,
        "precision": 1.0,
        "f1": 1.0,
    }
    assert "cannot replace" in binary["interpretation_boundary"]
    assert (
        result["comparison_design"]["inter_expert_reliability"]
        == "not_computable_no_independent_rater_A_B_labels"
    )


def test_failures_remain_in_denominator_and_are_hash_audited(tmp_path: Path):
    suite_dir, state = _run(tmp_path, _FakeModel(fail_on={1}), limit=3, repeats=1)
    assert len(state.records) == 3
    assert sum(row.outcome is CellOutcome.FAILED for row in state.records) == 1
    failed = next(row for row in state.records if row.outcome is CellOutcome.FAILED)
    assert (suite_dir / failed.cell_dir / "failure.json").is_file()
    result = analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)
    assert result["denominators"]["expected_calls"] == 3
    assert result["denominators"]["succeeded_calls"] == 2
    assert result["denominators"]["failed_calls"] == 1
    assert result["abstention_and_failure"]["call_failure_rate_all_planned"] == 1 / 3


def test_uncertain_is_reported_as_abstention_and_remains_a_class(tmp_path: Path):
    rows = select_claim_records(
        load_expert_gold_records()["claim_reviews"],
        limit=1,
        selection_seed="fixture-selection",
    )
    model = _FakeModel(
        decision_by_triple={rows[0]["triple"]: AdmissionDecision.UNCERTAIN}
    )
    suite_dir, _ = _run(tmp_path, model, limit=1)
    result = analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)
    assert result["abstention_and_failure"]["predicted_uncertain_calls"] == 1
    assert result["abstention_and_failure"]["predicted_uncertain_rate_among_succeeded"] == 1.0
    assert "uncertain" in result["classification"]["per_class"]


def test_binary_keep_candidate_metrics_use_registered_mapping_and_2x2():
    result = _binary_keep_candidate_metrics(
        [
            AdmissionDecision.ACCEPT,
            AdmissionDecision.ACCEPT_WITH_EDITS,
            AdmissionDecision.REJECT,
            AdmissionDecision.UNCERTAIN,
        ],
        [
            AdmissionDecision.ACCEPT_WITH_EDITS,
            AdmissionDecision.REJECT,
            AdmissionDecision.ACCEPT,
            AdmissionDecision.UNCERTAIN,
        ],
    )
    assert result["mapping"] == {
        "positive_keep_candidate": ["accept", "accept_with_edits"],
        "negative_block_candidate": ["reject", "uncertain"],
    }
    assert result["confusion_matrix_rows_gold_columns_system"] == {
        "positive_keep_candidate": {
            "positive_keep_candidate": 1,
            "negative_block_candidate": 1,
        },
        "negative_block_candidate": {
            "positive_keep_candidate": 1,
            "negative_block_candidate": 1,
        },
    }
    assert result["counts"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 1,
    }
    assert result["metrics"] == {
        "accuracy": 0.5,
        "sensitivity_recall": 0.5,
        "specificity": 0.5,
        "precision": 0.5,
        "f1": 0.5,
    }


def test_resume_skips_hash_audited_cells_after_interrupt(tmp_path: Path):
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _FakeModel(interrupt_on=3), limit=3, repeats=1)
    partial = json.loads(
        (tmp_path / "results/claim-fixture/suite_state.json").read_text(encoding="utf-8")
    )
    assert partial["status"] == "running"
    assert len(partial["records"]) == 2

    resumed = _FakeModel()
    suite_dir, state = _run(tmp_path, resumed, limit=3, repeats=1, resume=True)
    assert state.status is SuiteStatus.COMPLETED
    assert len(resumed.calls) == 1
    assert (suite_dir / "suite_summary.json").is_file()


def test_resume_rejects_parameter_drift(tmp_path: Path):
    _run(tmp_path, _FakeModel(), limit=2, repeats=1)
    with pytest.raises(ValueError, match="resume 配置/输入不一致"):
        _run(tmp_path, _FakeModel(), limit=3, repeats=1, resume=True)


def test_resume_and_analysis_reject_tampered_artifact(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakeModel(), limit=1)
    artifact = next((suite_dir / "cells").glob("*/replicate-01/artifact.json"))
    artifact.write_text(artifact.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash 不匹配"):
        _run(tmp_path, _FakeModel(), limit=1, resume=True)
    with pytest.raises(ValueError, match="artifact.json hash 不匹配"):
        analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)


def test_analysis_recomputes_all_frozen_blind_input_hashes(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakeModel(fail_on={0}), limit=1)
    state_path = suite_dir / "suite_state.json"
    summary_path = suite_dir / "suite_summary.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    review_id = payload["input_snapshot"]["selected_review_ids"][0]
    payload["input_snapshot"]["blind_input_sha256_by_review_id"][review_id] = "0" * 64
    tampered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    state_path.write_text(tampered, encoding="utf-8")
    summary_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="冻结盲输入 hash"):
        analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)


def test_failure_redaction_removes_configured_key_and_bearer(monkeypatch):
    monkeypatch.setenv("HY3_API_KEY", "secret-fixture-key")
    value = _redact_failure(
        "request failed key=secret-fixture-key Authorization: "
        + "Bearer "
        + "abc.def-123"
    )
    assert "secret-fixture-key" not in value
    assert "abc.def-123" not in value
    assert "<REDACTED_SECRET>" in value


def test_claim_failure_artifact_and_state_never_persist_secret_or_reasoning(
    tmp_path: Path, monkeypatch
):
    current_key = "claim-current-key-fixture"
    long_old_key = "ClaimOldToken_ABCDEF0123456789_abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("HY3_API_KEY", current_key)

    class _SensitiveFailure(_FakeModel):
        def classify(self, item, *, temperature, seed):
            self.calls.append((item, temperature, seed))
            raise RuntimeError(
                f"x-api-key={current_key} old={long_old_key} "
                "https://alice:password@example.org/v1?token=query-secret "
                'reasoning_content="private model reasoning"'
            )

    suite_dir, state = _run(tmp_path, _SensitiveFailure(), limit=1, repeats=1)
    assert state.records[0].outcome is CellOutcome.FAILED
    persisted = "\n".join(
        path.read_text(encoding="utf-8") for path in suite_dir.rglob("*.json")
    )
    for secret in (
        current_key,
        long_old_key,
        "alice",
        "password",
        "query-secret",
        "private model reasoning",
    ):
        assert secret not in persisted
    assert "<REDACTED_SECRET>" in persisted
    assert "<REDACTED_LONG_TOKEN>" in persisted
    assert "REDACTED_QUERY" in persisted
    assert "<REDACTED_MODEL_REASONING>" in persisted


def _make_claim_orphan(tmp_path: Path) -> tuple[Path, Path]:
    runner = ClaimAdmissionPilotRunner(model=_FakeModel())
    original_write_state = runner._write_state
    writes = 0

    def crash_after_atomic_cell(suite_dir, state):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise KeyboardInterrupt("crash after cell rename before state journal")
        original_write_state(suite_dir, state)

    runner._write_state = crash_after_atomic_cell  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        runner.run_suite(
            repo_root=REPO_ROOT,
            out_root=tmp_path / "results",
            suite_id="claim-fixture",
            limit=1,
            repeats=1,
            selection_seed="fixture-selection",
            temperature=0.2,
            base_seed=123,
        )
    suite = tmp_path / "results/claim-fixture"
    artifact = next((suite / "cells").glob("*/replicate-01/artifact.json"))
    state = json.loads((suite / "suite_state.json").read_text(encoding="utf-8"))
    assert state["records"] == []
    return suite, artifact


def _rewrite_claim_orphan_artifact(artifact_path: Path, mutate) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(payload)
    artifact_data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    artifact_path.write_bytes(artifact_data)
    manifest_path = artifact_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["artifact.json"] = {
        "bytes": len(artifact_data),
        "sha256": hashlib.sha256(artifact_data).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("requested_seed", 7),
        lambda value: value["model_call"].__setitem__("model", "tampered-model"),
        lambda value: (
            value.__setitem__("model_config_sha256", "4" * 64),
            value["model_call"].__setitem__("config_sha256", "4" * 64),
        ),
        lambda value: value.__setitem__("prompt_template_sha256", "5" * 64),
        lambda value: (
            value.__setitem__("output_schema_sha256", "6" * 64),
            value["model_call"].__setitem__("schema_sha256", "6" * 64),
        ),
    ],
    ids=["seed", "model", "config", "prompt", "schema"],
)
def test_claim_orphan_recovery_rejects_provenance_drift(tmp_path: Path, mutate):
    _, artifact = _make_claim_orphan(tmp_path)
    _rewrite_claim_orphan_artifact(artifact, mutate)
    with pytest.raises(
        ValueError, match="model/config/prompt/schema/seed|canonical base prompt"
    ):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)


def test_claim_resume_rejects_symlinked_cell(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    record = json.loads((suite_dir / "suite_state.json").read_text(encoding="utf-8"))["records"][0]
    cell = suite_dir / record["cell_dir"]
    target = suite_dir / "real-cell-target"
    shutil.move(str(cell), target)
    cell.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)


@pytest.mark.parametrize(
    ("file_name", "failed_cell"),
    [("manifest.json", False), ("artifact.json", False), ("failure.json", True)],
)
def test_claim_resume_and_analysis_reject_file_level_symlink(
    tmp_path: Path, file_name: str, failed_cell: bool
):
    model = _FakeModel(fail_on={0} if failed_cell else set())
    suite_dir, state = _run(tmp_path, model, limit=1, repeats=1)
    path = suite_dir / state.records[0].cell_dir / file_name
    target = suite_dir / f"real-{file_name}"
    shutil.copy2(path, target)
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="symlink"):
        analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)


def test_claim_resume_and_analysis_reject_extra_cell_file(tmp_path: Path):
    suite_dir, state = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    cell = suite_dir / state.records[0].cell_dir
    (cell / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="文件集合"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="文件集合"):
        analyze_claim_admission_pilot(suite_dir, repo_root=REPO_ROOT)


def test_claim_resume_repairs_completed_state_summary_crash_window(
    tmp_path: Path, monkeypatch
):
    import app.claim_admission_pilot as pilot

    original = pilot._write_atomic
    crashed = False

    def crash_once_on_summary(path, data):
        nonlocal crashed
        if Path(path).name == "suite_summary.json" and not crashed:
            crashed = True
            raise KeyboardInterrupt("crash after completed state before summary")
        original(path, data)

    monkeypatch.setattr(pilot, "_write_atomic", crash_once_on_summary)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    suite = tmp_path / "results/claim-fixture"
    assert json.loads((suite / "suite_state.json").read_text())["status"] == "completed"
    assert not (suite / "suite_summary.json").exists()

    resumed = _FakeModel()
    _run(tmp_path, resumed, limit=1, repeats=1, resume=True)
    assert resumed.calls == []
    assert (suite / "suite_summary.json").read_bytes() == (
        suite / "suite_state.json"
    ).read_bytes()


def test_claim_completed_resume_rejects_divergent_existing_summary(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    (suite / "suite_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completed suite_state"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)


def _make_claim_failure_orphan(tmp_path: Path) -> tuple[Path, Path]:
    runner = ClaimAdmissionPilotRunner(model=_FakeModel(fail_on={0}))
    original_write_state = runner._write_state
    writes = 0

    def crash_after_failure_cell(suite_dir, state):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise KeyboardInterrupt("crash after failure cell before state journal")
        original_write_state(suite_dir, state)

    runner._write_state = crash_after_failure_cell  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        runner.run_suite(
            repo_root=REPO_ROOT,
            out_root=tmp_path / "results",
            suite_id="claim-fixture",
            limit=1,
            repeats=1,
            selection_seed="fixture-selection",
            temperature=0.2,
            base_seed=123,
        )
    suite = tmp_path / "results/claim-fixture"
    failure = next((suite / "cells").glob("*/replicate-01/failure.json"))
    assert json.loads((suite / "suite_state.json").read_text())["records"] == []
    return suite, failure


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request.__setitem__("suite_id", "other-suite"),
        lambda request: request.__setitem__("blind_input_sha256", "9" * 64),
        lambda request: request.__setitem__("requested_seed", 7),
        lambda request: request.__setitem__("model", "other-model"),
        lambda request: request.__setitem__("model_config_sha256", "4" * 64),
        lambda request: request.__setitem__("prompt_template_sha256", "5" * 64),
        lambda request: request.__setitem__("output_schema_sha256", "6" * 64),
    ],
    ids=["suite", "input", "seed", "model", "config", "prompt", "schema"],
)
def test_claim_orphan_failure_rejects_recommitted_wrong_request(
    tmp_path: Path, mutate
):
    _, failure_path = _make_claim_failure_orphan(tmp_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    mutate(failure["cell_request"])
    failure["cell_request_sha256"] = hashlib.sha256(
        json.dumps(
            failure["cell_request"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    failure_data = (
        json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    failure_path.write_bytes(failure_data)
    manifest_path = failure_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["failure.json"] = {
        "bytes": len(failure_data),
        "sha256": hashlib.sha256(failure_data).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cell_request/provenance"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)


def test_claim_v2_models_require_seed_and_all_provenance_hashes(tmp_path: Path):
    suite, state = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    artifact_path = suite / state.records[0].cell_dir / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for field, value in (
        ("requested_seed", None),
        ("model_config_sha256", ""),
        ("prompt_template_sha256", ""),
        ("output_schema_sha256", ""),
    ):
        candidate = dict(artifact)
        candidate[field] = value
        with pytest.raises(ValueError, match="v2 artifact"):
            ClaimCallArtifact.model_validate(candidate)
    for field, value in (
        ("base_prompt_sha256", "7" * 64),
        ("structured_output_sha256", "8" * 64),
        ("cache_namespace", "wrong-cache"),
        ("prompt_hash_scope", "wrong-scope"),
    ):
        candidate = json.loads(json.dumps(artifact))
        candidate["model_call"][field] = value
        with pytest.raises(ValueError, match="canonical base prompt"):
            ClaimCallArtifact.model_validate(candidate)
    state_payload = state.model_dump(mode="json")
    state_payload["base_seed"] = None
    with pytest.raises(ValueError, match="base_seed"):
        ClaimAdmissionSuiteState.model_validate(state_payload)


def test_claim_legacy_v1_is_readable_but_explicitly_nonformal(tmp_path: Path):
    suite, state = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    state_path = suite / "suite_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "mitoevidence.claim-admission-pilot.v1"
    record = payload["records"][0]
    cell = suite / record["cell_dir"]
    artifact_path = cell / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema_version"] = "mitoevidence.claim-admission-pilot.v1"
    artifact.pop("model_config_sha256", None)
    artifact.pop("prompt_template_sha256", None)
    artifact.pop("output_schema_sha256", None)
    artifact_data = (
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    artifact_path.write_bytes(artifact_data)
    manifest_path = cell / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "mitoevidence.claim-admission-pilot.v1"
    manifest["files"]["artifact.json"] = {
        "bytes": len(artifact_data),
        "sha256": hashlib.sha256(artifact_data).hexdigest(),
    }
    manifest_data = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_data)
    record["cell_manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()
    state_data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    state_path.write_text(state_data, encoding="utf-8")
    (suite / "suite_summary.json").write_text(state_data, encoding="utf-8")

    result = analyze_claim_admission_pilot(suite, repo_root=REPO_ROOT)
    assert result["formal_status"] == "legacy_v1_nonformal_limited_cell_provenance"
    assert result["provenance_assurance"]["per_cell_requested_seed_verified"] is False
    assert result["provenance_assurance"]["warning"]


def test_claim_resume_and_analysis_reject_symlinked_suite_root(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    target = suite.parent / "claim-real-suite"
    shutil.move(str(suite), target)
    suite.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="suite_dir.*symlink"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="suite_dir.*symlink"):
        analyze_claim_admission_pilot(suite, repo_root=REPO_ROOT)


@pytest.mark.parametrize("journal_name", ["suite_state.json", "suite_summary.json"])
def test_claim_resume_and_analysis_reject_symlinked_top_journal(
    tmp_path: Path, journal_name: str
):
    suite, _ = _run(tmp_path, _FakeModel(), limit=1, repeats=1)
    journal = suite / journal_name
    target = suite.parent / f"claim-real-{journal_name}"
    shutil.copy2(journal, target)
    journal.unlink()
    journal.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakeModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="symlink"):
        analyze_claim_admission_pilot(suite, repo_root=REPO_ROOT)


@pytest.mark.parametrize("source_kind", ["manifest", "claim_source"])
def test_claim_runner_rejects_symlinked_gold_source(
    tmp_path: Path, source_kind: str
):
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "annotation_prelabel", repo / "annotation_prelabel")
    runner = ClaimAdmissionPilotRunner(model=_FakeModel())
    existing_suite, _ = runner.run_suite(
        repo_root=repo,
        out_root=tmp_path / "source-results",
        suite_id="source-analyzer",
        limit=1,
        repeats=1,
        selection_seed="fixture-selection",
        temperature=0.2,
        base_seed=123,
    )
    if source_kind == "manifest":
        source = repo / "annotation_prelabel/expert_gold_manifest.json"
    else:
        source = repo / "annotation_prelabel/claim_review_sample/claim_review_sample.jsonl"
    target = repo / f"real-{source.name}"
    shutil.copy2(source, target)
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(ValueError, match="source snapshot.*symlink"):
        runner.run_suite(
            repo_root=repo,
            out_root=tmp_path / "source-results",
            suite_id="source-symlink",
            limit=1,
            repeats=1,
            selection_seed="fixture-selection",
            temperature=0.2,
            base_seed=123,
        )
    with pytest.raises(ValueError, match="source snapshot.*symlink"):
        analyze_claim_admission_pilot(existing_suite, repo_root=repo)


class _CaptureStructuredClient:
    model = "capture-hy3"

    def __init__(self):
        self.kwargs = None

    def _call(self, **kwargs):
        self.kwargs = kwargs
        return StructuredResult(
            value=ClaimAdmissionVerdict(
                decision=AdmissionDecision.ACCEPT_WITH_EDITS,
                concise_reason="condition should be retained",
            ),
            audit=ModelCallAudit(
                stage="claim_admission_blind",
                provider="offline-fixture",
                model=self.model,
                config_sha256=default_judge_config().sha256,
                schema_sha256=OUTPUT_SCHEMA_SHA256,
            ),
        )


def test_claim_wrapper_reads_real_hy3_audit_identity():
    class _Transport:
        base_url = "https://tokenhub.tencentmaas.com/v1"

    structured = Hy3ReviewModel(
        config=default_judge_config(), model="hy3", transport=_Transport()
    )
    model = Hy3ClaimAdmissionModel(
        config=default_judge_config(), transport=structured.transport,
        structured_client=structured,
    )
    identity = model.execution_identity
    assert identity.execution_kind.value == "remote_hy3"
    assert identity.provider == "tencent-tokenhub"
    assert identity.endpoint_origin == "https://tokenhub.tencentmaas.com"
    assert identity.endpoint_url == "https://tokenhub.tencentmaas.com/v1/chat/completions"

    class _EvilTransport:
        base_url = "https://evil.example/v1"

    custom_endpoint = Hy3ReviewModel(
        config=default_judge_config(), model="hy3", transport=_EvilTransport()
    )
    custom_model = Hy3ReviewModel(
        config=default_judge_config(), model="qwen", transport=_Transport()
    )
    for structured_client in (custom_endpoint, custom_model):
        downgraded = Hy3ClaimAdmissionModel(
            config=default_judge_config(), transport=structured_client.transport,
            structured_client=structured_client,
        ).execution_identity
        assert downgraded.execution_kind.value == "offline_fixture"
        assert downgraded.provider == "offline-fixture"
        assert downgraded.endpoint_url == ""


def test_real_wrapper_uses_strict_schema_and_no_gold_or_reviewer_fields():
    client = _CaptureStructuredClient()
    model = Hy3ClaimAdmissionModel(
        config=default_judge_config(), transport=object(), structured_client=client
    )
    assert model.execution_identity.execution_kind.value == "offline_fixture"
    item = BlindClaimInput(
        triple="A --promotes--> B",
        evidence_text="A increased B under treatment.",
        recorded_conditions={"species": "mouse"},
        paper_id="paper-1",
        paper_short="Paper 1",
        source_type="primary",
        section="results",
    )
    verdict, audit = model.classify(item, temperature=0.2, seed=7)
    assert verdict.decision is AdmissionDecision.ACCEPT_WITH_EDITS
    assert audit.stage == "claim_admission_blind"
    kwargs = client.kwargs
    assert kwargs["schema"] == {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["accept", "accept_with_edits", "reject", "uncertain"],
            },
            "concise_reason": {"type": "string", "minLength": 1, "maxLength": 800},
        },
        "required": ["decision", "concise_reason"],
        "additionalProperties": False,
    }
    assert kwargs["seed"] == 7
    prompt = kwargs["user"]
    for field in MODEL_EXPOSED_FIELDS:
        assert field in prompt
    for forbidden in (
        "ai_decision",
        "ai_reasoning",
        "defect_codes",
        "suggested_edits",
        "usable_for_beta_cell_evidence",
        "needs_human_verification",
        "annotator",
        "review_status",
    ):
        assert forbidden not in prompt
