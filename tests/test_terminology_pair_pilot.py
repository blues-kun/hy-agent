"""Offline contracts for the real-Hy3 terminology pair Pilot."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from app.hy3_review import (
    GENERATOR_OUTPUT_HASH_SCOPE,
    GENERATOR_PROMPT_HASH_SCOPE,
    Hy3ReviewModel,
    _json_sha256 as _hy3_json_sha256,
)
from app.schemas import ModelCallAudit
from app.terminology_pair_pilot import (
    BlindTerminologyPair,
    CellOutcome,
    FORMAL_STATUS,
    Hy3TerminologyPairModel,
    ORDER_ALGORITHM_V1,
    PairCallArtifact,
    PairSide,
    PAIR_PROMPT_TEMPLATE_SHA256,
    PAIR_SCHEMA_SHA256,
    PAIR_CACHE_NAMESPACE,
    SuiteStatus,
    TerminologyPairPilotRunner,
    TerminologyPairSuiteState,
    TerminologyPairVerdict,
    analyze_terminology_pair_pilot,
    build_pair_order,
    _pair_base_prompt_sha256,
    reconstruct_pair_order_from_gold,
    select_terminology_records,
)
from evaluator.expert_gold import audit_expert_gold, load_expert_gold_records
from evaluator.judge.config import default_judge_config


REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakePairModel:
    model = "fake-hy3"
    config_sha256 = "1" * 64
    prompt_template_sha256 = PAIR_PROMPT_TEMPLATE_SHA256
    output_schema_sha256 = PAIR_SCHEMA_SHA256

    def __init__(self, *, interrupt_on: int | None = None, fail_on: set[int] | None = None):
        self.calls: list[tuple[BlindTerminologyPair, float, int | None]] = []
        self.interrupt_on = interrupt_on
        self.fail_on = fail_on or set()

    def choose(self, pair, *, temperature, seed):
        self.calls.append((pair, temperature, seed))
        call_number = len(self.calls)
        if call_number == self.interrupt_on:
            raise KeyboardInterrupt("fixture interruption")
        if call_number in self.fail_on:
            raise RuntimeError("fixture API failure")
        verdict = TerminologyPairVerdict(
            preferred_side=PairSide.LEFT,
            confidence=0.8,
            concise_reason="left is the safer fixture choice",
        )
        return (
            verdict,
            ModelCallAudit(
                stage="terminology_pair_discrimination",
                provider="offline-fixture",
                model=self.model,
                config_sha256=self.config_sha256,
                schema_sha256=self.output_schema_sha256,
                prompt_sha256="a" * 64,
                base_prompt_sha256=_pair_base_prompt_sha256(pair),
                prompt_hash_scope=GENERATOR_PROMPT_HASH_SCOPE,
                structured_output_sha256=_hy3_json_sha256(
                    verdict.model_dump(mode="json")
                ),
                structured_output_hash_scope=GENERATOR_OUTPUT_HASH_SCOPE,
                temperature=temperature,
                requested_seed=seed,
                cache_namespace=PAIR_CACHE_NAMESPACE,
            ),
        )


def _run(
    tmp_path: Path,
    model: _FakePairModel,
    *,
    suite_id: str = "term-fixture",
    limit: int = 2,
    repeats: int = 2,
    resume: bool = False,
):
    return TerminologyPairPilotRunner(model=model).run_suite(
        repo_root=REPO_ROOT,
        out_root=tmp_path / "results",
        suite_id=suite_id,
        limit=limit,
        repeats=repeats,
        selection_seed="fixture-selection",
        order_seed="fixture-order",
        temperature=0.4,
        base_seed=100,
        resume=resume,
    )


def test_gold_selection_and_pair_order_are_hash_fixed():
    audit = audit_expert_gold()
    assert audit["ok"] is True
    assert audit["datasets"]["terminology_rules"]["record_count"] == 60
    records = load_expert_gold_records()["terminology_rules"]
    selected_a = select_terminology_records(records, limit=7, selection_seed="fixed")
    selected_b = select_terminology_records(records, limit=7, selection_seed="fixed")
    assert [row["term_id"] for row in selected_a] == [row["term_id"] for row in selected_b]

    row = selected_a[0]
    order_a = build_pair_order(row, order_seed="fixed-order")
    order_b = build_pair_order(row, order_seed="fixed-order")
    assert order_a == order_b
    assert {order_a.pair.left_text, order_a.pair.right_text} == {
        row["wrong"],
        row["correct"],
    }
    assert {order_a.correct_side, order_a.wrong_side} == {
        PairSide.LEFT,
        PairSide.RIGHT,
    }
    changed = dict(row)
    changed["correct"] = str(row["correct"]) + " [content changed]"
    changed_order = build_pair_order(changed, order_seed="fixed-order")
    assert changed_order.order_sha256 != order_a.order_sha256
    with pytest.raises(ValueError, match="expert source"):
        reconstruct_pair_order_from_gold(
            changed,
            order_a,
            order_seed_sha256=hashlib.sha256(b"fixed-order").hexdigest(),
        )


def test_runner_uses_only_blind_pair_and_keeps_complete_grid(tmp_path: Path):
    model = _FakePairModel()
    suite_dir, state = _run(tmp_path, model, limit=3, repeats=2)
    assert state.formal_status == "offline_fixture_nonformal_terminology_pair_pilot"
    assert state.execution_identity.execution_kind.value == "offline_fixture"
    assert state.provider == "offline-fixture"
    assert state.status is SuiteStatus.COMPLETED
    assert state.expected_calls == 6
    assert len(state.records) == 6
    assert all(record.outcome is CellOutcome.SUCCEEDED for record in state.records)
    assert len(model.calls) == 6
    assert all(set(pair.model_dump()) == {"left_text", "right_text"} for pair, _, _ in model.calls)
    assert state.input_snapshot.prompt_exposed_fields == ["left_text", "right_text"]
    assert state.input_snapshot.label_derivation == "field_role_wrong_correct"
    assert len(state.input_snapshot.scoring_pair_orders) == 3
    assert state.safety.gold_side_available_to_model is False
    assert (suite_dir / "suite_summary.json").is_file()


def test_all_failure_fixture_cannot_be_relabelled_as_formal_remote(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakePairModel(fail_on={0}), limit=1, repeats=1)
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
    with pytest.raises(ValueError, match="audit 与 suite|failure request"):
        analyze_terminology_pair_pilot(suite, repo_root=REPO_ROOT)


def test_failed_call_is_retained_in_denominator_and_analysis(tmp_path: Path):
    suite_dir, state = _run(
        tmp_path,
        _FakePairModel(fail_on={1}),
        limit=2,
        repeats=2,
    )
    assert len(state.records) == 4
    assert sum(record.outcome is CellOutcome.FAILED for record in state.records) == 1
    failed = next(record for record in state.records if record.outcome is CellOutcome.FAILED)
    assert (suite_dir / failed.cell_dir / "failure.json").is_file()

    result = analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)
    assert result["denominators"]["expected_calls"] == 4
    assert result["denominators"]["succeeded_calls"] == 3
    assert result["denominators"]["failed_calls"] == 1
    assert result["metrics"]["call_failure_rate"] == 0.25
    assert result["metrics"]["pair_accuracy_all_schema_valid_calls"] is not None
    assert "not full-text" in result["interpretation"]
    assert "length_only_baseline_accuracy_non_tied_available_pairs" in result["metrics"]


def test_failure_writer_redacts_keys_urls_long_tokens_and_reasoning(tmp_path: Path, monkeypatch):
    current_key = "current-key-fixture"
    old_long_token = "OldOpaqueToken_ABCDEF0123456789_abcdefghijklmnopqrstuvwxyz"
    bearer = "BearerFixture_ABCDEF0123456789_abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("HY3_API_KEY", current_key)

    class _SensitiveFailure(_FakePairModel):
        def choose(self, pair, *, temperature, seed):
            self.calls.append((pair, temperature, seed))
            raise RuntimeError(
                f"x-api-key={current_key} Authorization: Bearer {bearer} "
                f"old={old_long_token} "
                "https://alice:password@example.org/v1/chat?api_key=query-secret&x=1 "
                'reasoning_content="private chain of thought"'
            )

    suite_dir, state = _run(tmp_path, _SensitiveFailure(), limit=1, repeats=1)
    assert state.records[0].outcome is CellOutcome.FAILED
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in suite_dir.rglob("*.json")
    )
    for secret in (
        current_key,
        bearer,
        old_long_token,
        "alice",
        "password",
        "query-secret",
        "private chain of thought",
    ):
        assert secret not in persisted
    assert "<REDACTED_SECRET>" in persisted
    assert "<REDACTED_LONG_TOKEN>" in persisted
    assert "REDACTED_QUERY" in persisted
    assert "<REDACTED_MODEL_REASONING>" in persisted
    failure = json.loads(
        (suite_dir / state.records[0].cell_dir / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["security"]["contains_reasoning_content"] is False


def test_success_artifact_is_writer_sanitized_before_no_secret_manifest_claim(
    tmp_path: Path,
):
    long_token = "OutputToken_ABCDEF0123456789_abcdefghijklmnopqrstuvwxyz"

    class _SensitiveSuccess(_FakePairModel):
        def choose(self, pair, *, temperature, seed):
            self.calls.append((pair, temperature, seed))
            verdict = TerminologyPairVerdict(
                preferred_side=PairSide.LEFT,
                confidence=0.8,
                concise_reason=f"Bearer {long_token}",
            )
            return (
                verdict,
                ModelCallAudit(
                    stage="terminology_pair_discrimination",
                    provider="offline-fixture",
                    model=self.model,
                    endpoint_origin=(
                        "https://alice:password@example.org/v1?api_key=query-secret"
                    ),
                    config_sha256=self.config_sha256,
                    schema_sha256=self.output_schema_sha256,
                    prompt_sha256="a" * 64,
                    base_prompt_sha256=_pair_base_prompt_sha256(pair),
                    prompt_hash_scope=GENERATOR_PROMPT_HASH_SCOPE,
                    structured_output_sha256=_hy3_json_sha256(
                        verdict.model_dump(mode="json")
                    ),
                    structured_output_hash_scope=GENERATOR_OUTPUT_HASH_SCOPE,
                    temperature=temperature,
                    requested_seed=seed,
                    cache_namespace=PAIR_CACHE_NAMESPACE,
                ),
            )

    suite_dir, state = _run(tmp_path, _SensitiveSuccess(), limit=1, repeats=1)
    # An offline fixture that claims a remote/credential-bearing endpoint is
    # rejected as a failed cell; it cannot be sanitized into a formal success.
    assert state.records[0].outcome is CellOutcome.FAILED
    artifact_path = suite_dir / state.records[0].cell_dir / "failure.json"
    persisted = artifact_path.read_text(encoding="utf-8")
    for secret in (long_token, "alice", "password", "query-secret"):
        assert secret not in persisted
    analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)


def test_resume_skips_audited_cells_after_keyboard_interrupt(tmp_path: Path):
    interrupted = _FakePairModel(interrupt_on=3)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, interrupted, limit=2, repeats=2)
    state_path = tmp_path / "results/term-fixture/suite_state.json"
    partial = json.loads(state_path.read_text(encoding="utf-8"))
    assert partial["status"] == "running"
    assert len(partial["records"]) == 2

    resumed_model = _FakePairModel()
    suite_dir, completed = _run(
        tmp_path,
        resumed_model,
        limit=2,
        repeats=2,
        resume=True,
    )
    assert completed.status is SuiteStatus.COMPLETED
    assert len(completed.records) == 4
    assert len(resumed_model.calls) == 2  # first two successful cells were not called again
    assert (suite_dir / "suite_summary.json").is_file()


def _make_terminology_orphan(tmp_path: Path) -> tuple[Path, Path]:
    runner = TerminologyPairPilotRunner(model=_FakePairModel())
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
            suite_id="term-fixture",
            limit=1,
            repeats=1,
            selection_seed="fixture-selection",
            order_seed="fixture-order",
            temperature=0.4,
            base_seed=100,
        )
    suite = tmp_path / "results/term-fixture"
    artifact = next((suite / "cells").glob("*/replicate-01/artifact.json"))
    state = json.loads((suite / "suite_state.json").read_text(encoding="utf-8"))
    assert state["records"] == []
    return suite, artifact


def _rewrite_orphan_artifact(artifact_path: Path, mutate) -> None:
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
def test_orphan_recovery_rejects_semantic_provenance_drift(tmp_path: Path, mutate):
    _, artifact = _make_terminology_orphan(tmp_path)
    _rewrite_orphan_artifact(artifact, mutate)
    with pytest.raises(
        ValueError, match="model/config/prompt/schema/seed|canonical base prompt"
    ):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)


def test_resume_rejects_symlinked_cell_even_when_target_is_inside_suite(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    record = json.loads((suite_dir / "suite_state.json").read_text(encoding="utf-8"))["records"][0]
    cell = suite_dir / record["cell_dir"]
    target = suite_dir / "real-cell-target"
    shutil.move(str(cell), target)
    cell.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)


@pytest.mark.parametrize(
    ("file_name", "failed_cell"),
    [("manifest.json", False), ("artifact.json", False), ("failure.json", True)],
)
def test_term_resume_and_analysis_reject_file_level_symlink(
    tmp_path: Path, file_name: str, failed_cell: bool
):
    model = _FakePairModel(fail_on={1} if failed_cell else set())
    suite_dir, state = _run(tmp_path, model, limit=1, repeats=1)
    path = suite_dir / state.records[0].cell_dir / file_name
    target = suite_dir / f"real-{file_name}"
    shutil.copy2(path, target)
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="symlink"):
        analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)


def test_term_resume_and_analysis_reject_extra_cell_file(tmp_path: Path):
    suite_dir, state = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    cell = suite_dir / state.records[0].cell_dir
    (cell / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="文件集合"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="文件集合"):
        analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)


def test_term_resume_repairs_completed_state_summary_crash_window(
    tmp_path: Path, monkeypatch
):
    import app.terminology_pair_pilot as pilot

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
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    suite = tmp_path / "results/term-fixture"
    assert json.loads((suite / "suite_state.json").read_text())["status"] == "completed"
    assert not (suite / "suite_summary.json").exists()

    resumed = _FakePairModel()
    _run(tmp_path, resumed, limit=1, repeats=1, resume=True)
    assert resumed.calls == []
    assert (suite / "suite_summary.json").read_bytes() == (
        suite / "suite_state.json"
    ).read_bytes()


def test_term_completed_resume_rejects_divergent_existing_summary(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    (suite / "suite_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completed suite_state"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)


def _make_term_failure_orphan(tmp_path: Path) -> tuple[Path, Path]:
    runner = TerminologyPairPilotRunner(model=_FakePairModel(fail_on={1}))
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
            suite_id="term-fixture",
            limit=1,
            repeats=1,
            selection_seed="fixture-selection",
            order_seed="fixture-order",
            temperature=0.4,
            base_seed=100,
        )
    suite = tmp_path / "results/term-fixture"
    failure = next((suite / "cells").glob("*/replicate-01/failure.json"))
    assert json.loads((suite / "suite_state.json").read_text())["records"] == []
    return suite, failure


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request.__setitem__("suite_id", "other-suite"),
        lambda request: request.__setitem__("pair_order_sha256", "8" * 64),
        lambda request: request.__setitem__("pair_sha256", "9" * 64),
        lambda request: request.__setitem__("requested_seed", 7),
        lambda request: request.__setitem__("model", "other-model"),
        lambda request: request.__setitem__("model_config_sha256", "4" * 64),
        lambda request: request.__setitem__("prompt_template_sha256", "5" * 64),
        lambda request: request.__setitem__("output_schema_sha256", "6" * 64),
    ],
    ids=["suite", "order", "pair", "seed", "model", "config", "prompt", "schema"],
)
def test_term_orphan_failure_rejects_recommitted_wrong_request(
    tmp_path: Path, mutate
):
    _, failure_path = _make_term_failure_orphan(tmp_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    mutate(failure["cell_request"])
    request_data = (
        json.dumps(
            failure["cell_request"], ensure_ascii=False, sort_keys=True, indent=2
        )
        + "\n"
    ).encode("utf-8")
    failure["cell_request_sha256"] = hashlib.sha256(request_data).hexdigest()
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
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)


def test_term_v2_requires_seed_hashes_and_failure_hash_size(tmp_path: Path):
    suite, state = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
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
            PairCallArtifact.model_validate(candidate)
    for field, value in (
        ("base_prompt_sha256", "7" * 64),
        ("structured_output_sha256", "8" * 64),
        ("cache_namespace", "wrong-cache"),
        ("prompt_hash_scope", "wrong-scope"),
    ):
        candidate = json.loads(json.dumps(artifact))
        candidate["model_call"][field] = value
        with pytest.raises(ValueError, match="canonical base prompt"):
            PairCallArtifact.model_validate(candidate)
    state_payload = state.model_dump(mode="json")
    state_payload["base_seed"] = None
    with pytest.raises(ValueError, match="base_seed"):
        TerminologyPairSuiteState.model_validate(state_payload)

    _, failed_state = _run(
        tmp_path,
        _FakePairModel(fail_on={1}),
        suite_id="term-failed",
        limit=1,
        repeats=1,
    )
    failed_payload = failed_state.model_dump(mode="json")
    failed_payload["records"][0]["failure_sha256"] = None
    failed_payload["records"][0]["failure_bytes"] = None
    with pytest.raises(ValueError, match="failure record"):
        TerminologyPairSuiteState.model_validate(failed_payload)


def test_resume_rejects_parameter_drift(tmp_path: Path):
    _run(tmp_path, _FakePairModel(), limit=2, repeats=1)
    with pytest.raises(ValueError, match="resume 配置/输入不一致"):
        _run(
            tmp_path,
            _FakePairModel(),
            limit=3,
            repeats=1,
            resume=True,
        )


def test_completed_resume_reaudits_existing_cell_hashes(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    artifact = next((suite_dir / "cells").glob("*/replicate-01/artifact.json"))
    artifact.write_text(artifact.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash 不匹配"):
        _run(
            tmp_path,
            _FakePairModel(),
            limit=1,
            repeats=1,
            resume=True,
        )


def test_analyzer_rebuilds_pair_text_and_sides_from_expert_source(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    state_path = suite_dir / "suite_state.json"
    summary_path = suite_dir / "suite_summary.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["input_snapshot"]["scoring_pair_orders"][0]["pair"]["left_text"] = (
        "tampered pair text"
    )
    tampered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    state_path.write_text(tampered, encoding="utf-8")
    summary_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="expert source"):
        analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)


def test_legacy_v1_suite_remains_strictly_readable(tmp_path: Path):
    suite_dir, _ = _run(tmp_path, _FakePairModel(), limit=2, repeats=1)
    state_path = suite_dir / "suite_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "mitoevidence.terminology-pair-pilot.v1"
    payload["input_snapshot"]["order_algorithm"] = ORDER_ALGORITHM_V1
    legacy_order_by_id = {}
    for order in payload["input_snapshot"]["scoring_pair_orders"]:
        digest = hashlib.sha256(
            f"fixture-order\0{order['term_id']}".encode("utf-8")
        ).hexdigest()
        order["order_sha256"] = digest
        legacy_order_by_id[order["term_id"]] = order

    for record in payload["records"]:
        cell = suite_dir / record["cell_dir"]
        artifact_path = cell / "artifact.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["schema_version"] = "mitoevidence.terminology-pair-pilot.v1"
        artifact["order"] = legacy_order_by_id[record["term_id"]]
        artifact.pop("model_config_sha256", None)
        artifact.pop("prompt_template_sha256", None)
        artifact.pop("output_schema_sha256", None)
        artifact_data = (
            json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        artifact_path.write_bytes(artifact_data)
        manifest_path = cell / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = "mitoevidence.terminology-pair-pilot.v1"
        manifest["files"]["artifact.json"] = {
            "bytes": len(artifact_data),
            "sha256": hashlib.sha256(artifact_data).hexdigest(),
        }
        manifest_data = (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_data)
        record["artifact_manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()

    legacy_state = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    state_path.write_text(legacy_state, encoding="utf-8")
    (suite_dir / "suite_summary.json").write_text(legacy_state, encoding="utf-8")
    result = analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)
    assert result["denominators"]["expected_calls"] == 2
    assert result["denominators"]["succeeded_calls"] == 2
    assert result["formal_status"] == "legacy_v1_nonformal_limited_cell_provenance"
    assert result["provenance_assurance"]["per_cell_requested_seed_verified"] is False


def test_term_resume_and_analysis_reject_symlinked_suite_root(tmp_path: Path):
    suite, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    target = suite.parent / "term-real-suite"
    shutil.move(str(suite), target)
    suite.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="suite_dir.*symlink"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="suite_dir.*symlink"):
        analyze_terminology_pair_pilot(suite, repo_root=REPO_ROOT)


@pytest.mark.parametrize("journal_name", ["suite_state.json", "suite_summary.json"])
def test_term_resume_and_analysis_reject_symlinked_top_journal(
    tmp_path: Path, journal_name: str
):
    suite, _ = _run(tmp_path, _FakePairModel(), limit=1, repeats=1)
    journal = suite / journal_name
    target = suite.parent / f"term-real-{journal_name}"
    shutil.copy2(journal, target)
    journal.unlink()
    journal.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path, _FakePairModel(), limit=1, repeats=1, resume=True)
    with pytest.raises(ValueError, match="symlink"):
        analyze_terminology_pair_pilot(suite, repo_root=REPO_ROOT)


@pytest.mark.parametrize("source_kind", ["manifest", "term_source"])
def test_term_runner_rejects_symlinked_gold_source(
    tmp_path: Path, source_kind: str
):
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "annotation_prelabel", repo / "annotation_prelabel")
    runner = TerminologyPairPilotRunner(model=_FakePairModel())
    existing_suite, _ = runner.run_suite(
        repo_root=repo,
        out_root=tmp_path / "source-results",
        suite_id="source-analyzer",
        limit=1,
        repeats=1,
        selection_seed="fixture-selection",
        order_seed="fixture-order",
        temperature=0.4,
        base_seed=100,
    )
    if source_kind == "manifest":
        source = repo / "annotation_prelabel/expert_gold_manifest.json"
    else:
        source = repo / "annotation_prelabel/terminology_blacklist/terminology_blacklist.jsonl"
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
            order_seed="fixture-order",
            temperature=0.4,
            base_seed=100,
        )
    with pytest.raises(ValueError, match="source snapshot.*symlink"):
        analyze_terminology_pair_pilot(existing_suite, repo_root=repo)


def test_full_wrong_correct_pairs_disclose_strong_length_only_baseline(tmp_path: Path):
    suite_dir, _ = _run(
        tmp_path,
        _FakePairModel(),
        suite_id="all-terminology",
        limit=60,
        repeats=1,
    )
    result = analyze_terminology_pair_pilot(suite_dir, repo_root=REPO_ROOT)
    assert result["counts"]["length_only_baseline_correct"] == 54
    assert result["counts"]["length_only_baseline_wrong"] == 6
    assert result["counts"]["length_only_baseline_tied"] == 0
    assert result["metrics"]["length_only_baseline_accuracy_non_tied_available_pairs"] == 0.9
    assert sum(result["counts"]["registered_gold_correct_side_pairs"].values()) == 60
    assert result["bias_baselines"]["length_only"]["unit"] == "unicode_codepoint_count"


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class _Session:
    def __init__(self, body):
        self.headers = {}
        self.body = body
        self.calls = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append({"json": json, "headers": headers})
        return _Response(self.body)


def test_real_client_payload_is_structured_seeded_and_contains_only_pair_data():
    body = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "must not persist",
                    "tool_calls": [
                        {
                            "id": "call-pair",
                            "type": "function",
                            "function": {
                                "name": "emit_pair_verdict",
                                "arguments": json.dumps(
                                    {
                                        "preferred_side": "right",
                                        "confidence": 0.9,
                                        "concise_reason": "right preserves the experimental condition",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    session = _Session(body)
    structured = Hy3ReviewModel(
        api_key="dummy",
        base_url="https://tokenhub.tencentmaas.com/v1",
        model="hy3",
        session=session,
        sleep_fn=lambda _: None,
    )
    model = Hy3TerminologyPairModel(
        config=default_judge_config(),
        transport=structured.transport,
        structured_client=structured,
    )
    assert model.execution_identity.execution_kind.value == "remote_hy3"
    assert model.execution_identity.provider == "tencent-tokenhub"
    assert model.execution_identity.endpoint_origin == "https://tokenhub.tencentmaas.com"
    assert model.execution_identity.endpoint_url == "https://tokenhub.tencentmaas.com/v1/chat/completions"
    pair = BlindTerminologyPair(left_text="unsafe candidate", right_text="safer candidate")
    verdict, audit = model.choose(pair, temperature=0.4, seed=123)
    assert verdict.preferred_side is PairSide.RIGHT
    payload = session.calls[0]["json"]
    assert payload["seed"] == 123
    assert payload["prompt_cache_key"].startswith(f"{PAIR_CACHE_NAMESPACE}-")
    assert session.calls[0]["headers"]["X-Session-ID"].startswith(
        f"{PAIR_CACHE_NAMESPACE}-"
    )
    assert payload["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    prompt = json.dumps(payload["messages"], ensure_ascii=False)
    assert "unsafe candidate" in prompt and "safer candidate" in prompt
    assert "TERM-" not in prompt
    assert "expert_consensus_gold" not in prompt
    assert "correct_side" not in prompt and "wrong_side" not in prompt
    assert audit.stage == "terminology_pair_discrimination"
    assert "reasoning_content" not in audit.model_dump()


def test_cli_has_no_offline_mode_and_requires_key_before_output(tmp_path: Path, monkeypatch):
    from scripts.run_terminology_pair_pilot import build_parser, main

    help_text = build_parser().format_help()
    assert "offline" not in help_text.lower()
    monkeypatch.delenv("HY3_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="缺少 HY3_API_KEY"):
        main(
            [
                "--limit",
                "1",
                "--repeats",
                "1",
                "--out-root",
                str(tmp_path / "should-not-exist"),
            ]
        )
    assert not (tmp_path / "should-not-exist").exists()
