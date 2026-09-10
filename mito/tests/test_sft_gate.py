"""CPU gate regressions; synthetic fixtures are never policy evidence."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "rp_grpo/audit_sft.py"
spec = importlib.util.spec_from_file_location("test_sft_gate_module", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_no_evaluation_is_blocked():
    report = gate.blocked_gate("No actual model rollout")
    assert report["passed"] is False
    assert report["training_authorized_by_this_gate"] is False
    assert len(report["checks"]) == 4
    assert {item["status"] for item in report["checks"]} == {"BLOCK"}
    assert report["multipath_environment_gate"]["status"] == "BLOCK"


def test_historical_success_cannot_unlock_rp():
    report = gate.blocked_gate("New environment untested", legacy={"success_rate": 1.0})
    assert report["passed"] is False


@pytest.mark.parametrize("text", ['{"success":true,"success":false}', '{"reward":NaN}', '{"reward":Infinity}', '{"reward":1e999}'])
def test_bad_json_refused(text):
    with pytest.raises(ValueError):
        gate.strict_json(text)


def test_no_existing_audit_overwrite(tmp_path):
    path = tmp_path / "report.json"
    gate.write_new(path, {"passed": False})
    with pytest.raises(ValueError, match="overwrite"):
        gate.write_new(path, {"passed": True})
    assert gate.read_json(path) == {"passed": False}


def test_symlink_output_refused(tmp_path, make_symlink):
    original = tmp_path / "original"
    original.write_text("untouched")
    target = tmp_path / "report.json"
    make_symlink(target, original)
    with pytest.raises(ValueError):
        gate.write_new(target, {})
    assert original.read_text() == "untouched"


def test_only_timing_ignored():
    assert gate.stable({"latency_ms": 1, "success": False}) == {"success": False}
    assert gate.stable({"success": False}) != gate.stable({"success": True})


def test_rp_replay_ignores_only_nondeterministic_timing():
    trace = {"trace_sha256": "original", "steps": [{"action": "read", "cost_proxy": 1}],
             "telemetry": {"wall_time_s": 1, "per_call_seconds": [1], "call_count": 1, "cost_basis": "proxy"}}
    replay = deepcopy(trace)
    replay["trace_sha256"] = "new"
    replay["telemetry"].update(wall_time_s=9, per_call_seconds=[9])
    assert gate.replay_content(trace) == gate.replay_content(replay)
    assert trace["telemetry"]["wall_time_s"] == 1
    replay["telemetry"]["call_count"] = 2
    assert gate.replay_content(trace) != gate.replay_content(replay)
    replay["telemetry"]["call_count"] = 1
    replay["steps"][0]["cost_proxy"] = 9
    assert gate.replay_content(trace) != gate.replay_content(replay)


class TinyTokenizer:
    """Token-contract test double; never a claimed model generation."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids if value != self.eos_token_id)


def token_step(completion="{}", ended=True):
    ids = [ord(char) for char in completion] + ([0] if ended else [])
    return {"prompt_ids": [120], "completion_ids": ids, "policy_tokens_sha256": gate.canonical_hash(ids),
            "prompt_tokens": 1, "completion_tokens": len(ids), "raw_text": completion, "terminated": ended}


TOKEN_CONFIG = {"max_new_tokens": 4, "max_context_tokens": 8}


def test_flat_actual_token_contract_and_truncated_record():
    gate.validate_policy_tokens(token_step(), TinyTokenizer(), "x", {}, TOKEN_CONFIG)
    gate.validate_policy_tokens(token_step("{", ended=False), TinyTokenizer(), "x",
                                {"termination_reason": "generation_truncated"}, TOKEN_CONFIG)


@pytest.mark.parametrize("field,value", [
    ("prompt_ids", [121]), ("completion_ids", [True]), ("policy_tokens_sha256", "wrong"),
    ("prompt_tokens", 9), ("completion_tokens", 9), ("raw_text", "invented"), ("terminated", False),
])
def test_mismatched_actual_tokens_refused(field, value):
    step = token_step()
    step[field] = value
    with pytest.raises(ValueError):
        gate.validate_policy_tokens(step, TinyTokenizer(), "x", {}, TOKEN_CONFIG)


def test_actual_empty_context_stop_is_failure_record_not_audit_crash():
    step = token_step("", ended=False)
    budget = {"max_new_tokens": 8, "max_context_tokens": 8}
    gate.validate_policy_tokens(step, TinyTokenizer(), "x", {"termination_reason": "context_budget"}, budget)
    for config, episode in [(TOKEN_CONFIG, {"termination_reason": "context_budget"}), (budget, {})]:
        with pytest.raises(ValueError, match="context-budget"):
            gate.validate_policy_tokens(step, TinyTokenizer(), "x", episode, config)
    step["action"] = {"tool": "stop", "arguments": {}}
    with pytest.raises(ValueError, match="context-budget"):
        gate.validate_policy_tokens(step, TinyTokenizer(), "x", {"termination_reason": "context_budget"}, budget)


def test_interval_records_uncertainty():
    lower, upper = gate.interval(70, 80)
    assert lower < .875 < upper < 1
    assert gate.interval(0, 0) is None
    assert gate.grouped_interval([{"family": "one", "success": True}], "family")["ci95"] is None


def test_split_audit_groups_and_hashes(tmp_path):
    manifest = {"files": {}}
    for split, name in (("train", "train.jsonl"), ("development", "development.jsonl"), ("test", "test.jsonl")):
        row = {"split": split, "task_id": split, "pair_id": split,
               "provenance": {"paper_ids": [split]}}
        path = tmp_path / name
        path.write_text(json.dumps(row) + "\n")
        manifest["files"][name] = {"sha256": gate.file_hash(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = gate.verify_split(tmp_path)
    assert result["status"] == "PASS"
    assert not result["test_usable_for_new_model_selection"]
    path = tmp_path / "test.jsonl"
    row = json.loads(path.read_text())
    row["provenance"]["paper_ids"] = ["train"]
    path.write_text(json.dumps(row) + "\n")
    manifest["files"]["test.jsonl"]["sha256"] = gate.file_hash(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        gate.verify_split(tmp_path)


def test_model_index_directory_escape_refused(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../escape"}}))
    with pytest.raises(ValueError, match="basenames"):
        gate.model_snapshot(tmp_path)


def episode(task="t1", side="available", **updates):
    value = {"task_id": task, "pair_id": "p1", "side": side, "group_id": "g1",
        "family": "text", "success": True, "relevant": True, "calls": 3,
        "schema_valid_calls": 3, "repair_opportunity": False, "repair_success": False,
        "initial_sha256": "initial", "observation_hashes": [side, "final"],
        "action_hashes": ["same_search", side], "path": ["search_first", "text"]}
    value.update(updates)
    return value


def test_perfect_single_path_and_no_error_test_is_blocked():
    rows = [episode(), episode("t2", "view_without_target"),
            episode("t3", pair_id="p2", group_id="g2", family="metric"),
            episode("t4", "view_without_target", pair_id="p2", group_id="g2", family="metric")]
    report = gate.decision_checks(rows)
    assert not report["passed"]
    assert report["checks"][3]["status"] == "PASS"
    assert report["checks"][2]["status"] == "BLOCK"
    assert report["multipath_environment_gate"]["status"] == "BLOCK"


def test_changed_actions_but_failed_task_does_not_pass_feedback():
    rows = [episode(), episode("t2", "view_without_target", success=False),
            episode("t3", pair_id="p2", group_id="g2", family="metric"),
            episode("t4", "view_without_target", pair_id="p2", group_id="g2", family="metric", success=False)]
    assert gate.decision_checks(rows)["checks"][2]["paired_success"] == 0


def test_different_stochastic_first_actions_are_not_feedback_adaptation():
    a, b = episode(), episode("t2", "view_without_target")
    assert gate.feedback_continuation_changed(a, b)
    b["action_hashes"][0] = "unmatched_first_guess"
    assert not gate.feedback_continuation_changed(a, b)
    b["action_hashes"] = ["same_search"]
    assert not gate.feedback_continuation_changed(a, b)


def test_different_task_paths_do_not_establish_multipath():
    rows = [episode(), episode("t2", path=["catalog_first", "text"])]
    assert gate.decision_checks(rows)["multipath_environment_gate"]["status"] == "BLOCK"


def test_duplicate_repair_draws_are_not_distinct_error_test_cases():
    rows = [episode(repair_opportunity=True, repair_success=True)] * 3
    check = gate.decision_checks(rows)["checks"][2]
    assert check["repair_opportunities"] == 3
    assert check["distinct_repair_tasks"] == 1
    assert check["status"] == "BLOCK"


def test_complete_synthetic_gate_logic_can_pass_but_is_not_release_evidence():
    rows = [episode(repair_opportunity=True, repair_success=True),
            episode("t2", "view_without_target", repair_opportunity=True, repair_success=True),
            episode("t3", pair_id="p2", group_id="g2", family="metric"),
            episode("t4", "view_without_target", pair_id="p2", group_id="g2", family="metric"),
            episode(path=["catalog_first", "text"])]
    assert gate.decision_checks(rows)["passed"]


@pytest.mark.parametrize("kwargs", [{"min_success": 0}, {"min_schema": 0}, {"min_pairs": 0}, {"min_repair": 0}, {"min_families": 0}])
def test_thresholds_cannot_vacuously_disable_checks(kwargs):
    with pytest.raises(ValueError):
        gate.decision_checks([episode()], **kwargs)
