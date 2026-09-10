"""Synthetic contract fixtures only: never evidence of actual model success."""
from copy import deepcopy
from pathlib import Path

import pytest

from rp_grpo import gate_validation as gate
from rp_grpo.train_policy import canonical_json, canonical_hash, sha256_file


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    expected = {"base_snapshot_sha256": "base", "initial_adapter_sha256": "adapter"}
    put(tmp_path / "rp_grpo/audit_sft.py", {"test": "fake auditor file for contract-only test"})
    put(tmp_path / "run/episodes.jsonl", {"test_only": True})
    source = {"schema": "mito.rp-grpo.policy-run.v1", "status": "completed", "mode": "evaluate",
              "record_source": "actual_hf_model", "arm": "B1", "split": "dev", "bindings": expected,
              "config": {"test": True}, "config_sha256": canonical_hash({"test": True}),
              "episodes_sha256": sha256_file(tmp_path / "run/episodes.jsonl")}
    report = {"schema": "mito.sft-readiness.rp-evaluation.v1", "passed": True, "status": "PASS",
              "actual_model_evaluation_verified": True, "bindings": expected,
              "checks": [{"id": key, "status": "PASS"} for key in sorted(gate.CHECKS)],
              "multipath_environment_gate": {"status": "PASS", "tasks_with_two_successful_policy_paths": ["test-task"]},
              "episodes": [{"task_id": "test-task", "success": True, "trace_sha256": "timing-dependent"}],
              "source_evaluation_manifest": {"path": "run/run_manifest.json"},
              "source_episodes_sha256": source["episodes_sha256"],
              "auditor_sha256": sha256_file(tmp_path / "rp_grpo/audit_sft.py")}
    admitted = {"schema": "mito.rp-grpo.sft-gate.v1", "passed": True, "bindings": dict(expected),
                "evaluation_report_path": "readiness.json"}
    actual = deepcopy(report)
    monkeypatch.setattr(gate, "_reaudit", lambda *_: actual)
    def save():
        put(tmp_path / "run/run_manifest.json", source)
        report["source_evaluation_manifest"]["sha256"] = sha256_file(tmp_path / "run/run_manifest.json")
        put(tmp_path / "readiness.json", report)
        admitted["evaluation_report_sha256"] = sha256_file(tmp_path / "readiness.json")
        admitted["bindings"]["evaluation_report_sha256"] = admitted["evaluation_report_sha256"]
        put(tmp_path / "SFT_GATE.json", admitted)
    save()
    return tmp_path, expected, source, report, admitted, actual, save


def check(a):
    root, expected, *rest = a
    return gate.validate_sft_gate(root / "SFT_GATE.json", expected, bundle=root)


def test_valid_full_contract_and_timing_only_difference(artifacts):
    artifacts[5]["episodes"][0]["trace_sha256"] = "replayed-different-time"
    assert check(artifacts)["independent_cpu_reaudit"] is True


@pytest.mark.parametrize("field,value", [("schema", "stub"), ("passed", False), ("status", "BLOCK"),
    ("actual_model_evaluation_verified", False), ("episodes", [])])
def test_outer_hash_cannot_approve_invalid_report(artifacts, field, value):
    artifacts[3][field] = value
    artifacts[-1]()
    with pytest.raises(ValueError):
        check(artifacts)


@pytest.mark.parametrize("mutation", ["duplicate", "failed", "no_paths", "auditor", "binding"])
def test_actual_checks_multipath_and_bindings_required(artifacts, mutation):
    report = artifacts[3]
    if mutation == "duplicate": report["checks"][0] = report["checks"][1]
    elif mutation == "failed": report["checks"][0]["status"] = "BLOCK"
    elif mutation == "no_paths": report["multipath_environment_gate"]["tasks_with_two_successful_policy_paths"] = []
    elif mutation == "auditor": report["auditor_sha256"] = "wrong"
    else: report["bindings"] = {"base_snapshot_sha256": "wrong"}
    artifacts[-1]()
    with pytest.raises(ValueError): check(artifacts)


@pytest.mark.parametrize("field,value", [("mode", "train"), ("split", "train"), ("arm", "B0"),
    ("record_source", "reference"), ("status", "incomplete"), ("config_sha256", "wrong")])
def test_actual_unassisted_complete_dev_only(artifacts, field, value):
    artifacts[2][field] = value
    artifacts[-1]()
    with pytest.raises(ValueError): check(artifacts)


def test_reaudited_failure_overrides_all_pass_flags(artifacts):
    artifacts[5]["passed"] = False
    with pytest.raises(ValueError): check(artifacts)


def test_edited_summary_rejected_even_if_hashes_updated(artifacts):
    artifacts[3]["episodes"][0]["success"] = False
    artifacts[-1]()
    with pytest.raises(ValueError, match="summaries"):
        check(artifacts)


def test_episode_file_tampering_rejected(artifacts):
    put(artifacts[0] / "run/episodes.jsonl", {"test_only": "changed"})
    with pytest.raises(ValueError, match="episodes hash"):
        check(artifacts)


def test_smoke_is_never_formal_admission():
    assert gate.validate_sft_gate(Path("does-not-exist"), {}, engineering_smoke=True)["formal_training"] is False
