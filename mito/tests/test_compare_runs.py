"""CPU-only comparison contract fixtures; never claimed as model evaluations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from rp_grpo import compare_runs as compare
from rp_grpo import environment, train_policy
from rp_grpo.audit_sft import canonical_hash, file_hash

pytestmark = pytest.mark.requires_data


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fixture_run(tmp_path, *, arm="B1", omit_scope=False):
    """Synthetic serialization around a genuinely executed CPU tool result."""
    folder = tmp_path / arm
    folder.mkdir()
    data = environment.DEFAULT_DATA
    selected = environment.load_tasks("dev")[0]
    env = environment.ResearchEnvironment(max_steps=6)
    initial = env.reset(selected)
    messages = train_policy.initial_messages(initial)
    action = {"tool": "stop", "arguments": {"scope": "authorized_snapshot_only", "reason": "当前授权范围，停止。"}}
    if omit_scope:
        del action["arguments"]["scope"]
    raw = json.dumps(action, ensure_ascii=False)
    step = {"messages": deepcopy(messages), "raw_text": raw, "terminated": True,
            "prompt_tokens": 9, "completion_tokens": 5}
    if arm == "B0":
        guard = train_policy.guard_action(action, messages)
        step.update(model_original_action=deepcopy(action), rule_intervention=guard)
        action = guard["action"]
    result = env.step(action)
    step.update(action=action, result=result)
    episode = {"task_id": selected["id"], "seed": 42, "initial_observation": initial,
               "initial_prompt_sha256": canonical_hash(messages), "steps": [step],
               "done": result["done"], "reward": result["reward"], "environment_trace": env.export_trace(),
               "termination_reason": "environment_terminal" if result["done"] else "turn_budget"}
    ep = folder / "episodes.jsonl"
    ep.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = {"snapshot_sha256": "a" * 64, "files_sha256": {"tokenizer.json": "f" * 64}}
    config = {"max_turns": 6, "max_new_tokens": 512, "max_context_tokens": 8192, "max_seconds": 900,
              "eval_sample": True, "eval_rollouts": 1, "max_eval_tasks": 1, "seed": 42,
              "mode": "evaluate", "arm": arm}
    bindings = {"base_snapshot_sha256": snapshot["snapshot_sha256"], "initial_adapter_sha256": "b" * 64,
        "training_data_sha256": file_hash(data / "tasks.jsonl"), "scorers_sha256": file_hash(data / "scorers.jsonl"),
        "data_manifest_sha256": file_hash(data / "MANIFEST.json"), "environment_sha256": file_hash(Path(environment.__file__)),
        "prompt_contract_sha256": train_policy.prompt_contract(snapshot)["sha256"]}
    manifest = {"schema": "mito.rp-grpo.policy-run.v1", "mode": "evaluate", "status": "completed",
        "split": "dev", "record_source": "actual_hf_model", "arm": arm, "config": config,
        "config_sha256": canonical_hash(config), "bindings": bindings, "model_snapshot": snapshot,
        "episodes_sha256": file_hash(ep), "elapsed_seconds": 1.2,
        "generation_pad_token_id": 151645, "generation_eos_token_id": 151645,
        "rollout_sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}}
    path = folder / "run_manifest.json"
    save(path, manifest)
    return path


def change_episode(path, edit):
    ep = path.parent / "episodes.jsonl"
    rows = [json.loads(line) for line in ep.read_text().splitlines()]
    edit(rows)
    ep.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads(path.read_text())
    manifest["episodes_sha256"] = file_hash(ep)
    save(path, manifest)


def test_independent_execution_keeps_failed_task_in_denominator(tmp_path):
    path = fixture_run(tmp_path)
    result = compare.compare([("test_fixture_only", path)], environment.DEFAULT_DATA)
    assert result["runs"]["test_fixture_only"]["summary"] == {
        "episodes": 1, "unique_tasks": 1, "successes": 0, "success_rate": 0}
    assert not result["is_sft_admission_gate"]
    assert not result["algorithm_improvement_proven"]


@pytest.mark.parametrize("field,value", [("status", "running"), ("record_source", "reference_policy"),
                                        ("mode", "train"), ("split", "train")])
def test_non_actual_or_unfinished_manifest_refused(tmp_path, field, value):
    path = fixture_run(tmp_path)
    manifest = json.loads(path.read_text())
    manifest[field] = value
    save(path, manifest)
    with pytest.raises(ValueError, match="completed actual-HF"):
        compare.inspect_run(path, environment.DEFAULT_DATA)


def test_duplicate_draws_and_changed_tool_returns_refused(tmp_path):
    path = fixture_run(tmp_path)
    change_episode(path, lambda rows: rows.append(deepcopy(rows[0])))
    with pytest.raises(ValueError, match="duplicate"):
        compare.inspect_run(path, environment.DEFAULT_DATA)
    change_episode(path, lambda rows: rows.pop())
    change_episode(path, lambda rows: rows[0]["steps"][0]["result"].update(reward=1))
    with pytest.raises(ValueError, match="return/reward"):
        compare.inspect_run(path, environment.DEFAULT_DATA)


def test_b0_rule_intervention_recomputed_from_visible_input(tmp_path):
    path = fixture_run(tmp_path, arm="B0", omit_scope=True)
    report = compare.inspect_run(path, environment.DEFAULT_DATA)
    assert report["cost"]["rule_interventions"] == 1
    change_episode(path, lambda rows: rows[0]["steps"][0]["rule_intervention"].update(intervened=False))
    with pytest.raises(ValueError, match="B0 intervention"):
        compare.inspect_run(path, environment.DEFAULT_DATA)


def test_nonrule_original_action_must_match_raw_text(tmp_path):
    path = fixture_run(tmp_path)
    change_episode(path, lambda rows: rows[0]["steps"][0].update(raw_text='{"tool":"list_documents","arguments":{}}'))
    with pytest.raises(ValueError, match="actual action"):
        compare.inspect_run(path, environment.DEFAULT_DATA)


def test_budget_mismatch_and_output_overwrite_refused(tmp_path):
    first, second = fixture_run(tmp_path), fixture_run(tmp_path, arm="B0")
    manifest = json.loads(second.read_text())
    manifest["config"]["max_new_tokens"] = 1024
    manifest["config_sha256"] = canonical_hash(manifest["config"])
    save(second, manifest)
    with pytest.raises(ValueError, match="budgets"):
        compare.compare([("B1", first), ("B0", second)], environment.DEFAULT_DATA)
    output = tmp_path / "existing.json"
    output.write_text("keep")
    with pytest.raises(ValueError, match="overwrite"):
        compare.write_new(output, {})
    assert output.read_text() == "keep"


def test_different_adapters_allowed_when_every_other_contract_matches(tmp_path):
    first, second = fixture_run(tmp_path), fixture_run(tmp_path, arm="B0")
    manifest = json.loads(second.read_text())
    manifest["bindings"]["initial_adapter_sha256"] = "c" * 64
    save(second, manifest)
    report = compare.compare([("B1", first), ("B0", second)], environment.DEFAULT_DATA)
    assert len(report["runs"]) == 2


def test_reserved_arm_label_cannot_disguise_actual_policy(tmp_path):
    path = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="reserved arm"):
        compare.compare([("B0", path)], environment.DEFAULT_DATA)
