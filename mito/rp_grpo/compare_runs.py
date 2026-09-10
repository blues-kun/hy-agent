"""Read-only comparison of matched actual RP evaluations, NOT an SFT gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rp_grpo.audit_sft import canonical_hash, file_hash, read_json, read_jsonl, replay_content, write_new
from rp_grpo import environment, train_policy

COMMON_BINDINGS = ("base_snapshot_sha256", "training_data_sha256", "scorers_sha256",
                   "data_manifest_sha256", "environment_sha256", "prompt_contract_sha256")
COMMON_CONFIG = ("max_turns", "max_new_tokens", "max_context_tokens", "max_seconds", "eval_sample",
                 "eval_rollouts", "max_eval_tasks", "seed")


def finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def summarize(rows):
    return {"episodes": len(rows), "unique_tasks": len({row["task_id"] for row in rows}),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None}


def inspect_run(manifest_path: Path, data_dir: Path) -> dict:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = read_json(manifest_path)
    if (manifest.get("schema") != "mito.rp-grpo.policy-run.v1" or manifest.get("mode") != "evaluate"
            or manifest.get("status") != "completed" or manifest.get("split") != "dev"
            or manifest.get("record_source") != "actual_hf_model"):
        raise ValueError("only completed actual-HF development evaluations are comparable")
    config, bindings = manifest["config"], manifest["bindings"]
    if (manifest.get("arm") not in train_policy.ARMS or config.get("mode") != "evaluate"
            or config.get("arm") != manifest["arm"]):
        raise ValueError("arm/mode mismatch between execution and configuration")
    if canonical_hash(config) != manifest.get("config_sha256"):
        raise ValueError("configuration hash mismatch")
    if any(not isinstance(bindings.get(key), str) or len(bindings[key]) != 64 for key in COMMON_BINDINGS):
        raise ValueError("missing exact evaluation bindings")
    for key in COMMON_CONFIG:
        if key not in config:
            raise ValueError("missing evaluation budget: " + key)
    for key in set(COMMON_CONFIG) - {"eval_sample", "seed"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("invalid evaluation budget: " + key)
    if type(config["seed"]) is not int or type(config["eval_sample"]) is not bool:
        raise ValueError("invalid seed/sampling contract")
    current = {"training_data_sha256": file_hash(data_dir / "tasks.jsonl"),
               "scorers_sha256": file_hash(data_dir / "scorers.jsonl"),
               "data_manifest_sha256": file_hash(data_dir / "MANIFEST.json"),
               "environment_sha256": file_hash(Path(environment.__file__))}
    if any(bindings[key] != value for key, value in current.items()):
        raise ValueError("local environment/data differ from the evaluated snapshot")
    if train_policy.prompt_contract(manifest["model_snapshot"])["sha256"] != bindings["prompt_contract_sha256"]:
        raise ValueError("prompt contract differs from the current actor-visible harness")
    if manifest["model_snapshot"].get("snapshot_sha256") != bindings["base_snapshot_sha256"]:
        raise ValueError("base snapshot binding mismatch")
    episodes_path = manifest_path.parent / "episodes.jsonl"
    if file_hash(episodes_path) != manifest.get("episodes_sha256"):
        raise ValueError("episodes file hash mismatch")
    episodes = read_jsonl(episodes_path)
    tasks = environment.load_tasks("dev", data_dir=data_dir)
    selected = tasks[:config["max_eval_tasks"]]
    task_map = {task["id"]: task for task in selected}
    expected = {(task["id"], config["seed"] + index * 10000 + repeat * 100): repeat
                for index, task in enumerate(selected) for repeat in range(config["eval_rollouts"])}
    actual_keys = [(row.get("task_id"), row.get("seed")) for row in episodes]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != set(expected):
        raise ValueError("missing/duplicate/extra task-and-seed draws; no silent intersection")
    checked = []
    for episode in episodes:
        task = task_map[episode["task_id"]]
        trace = episode.get("environment_trace")
        if not isinstance(trace, dict) or trace.get("max_steps") != config["max_turns"]:
            raise ValueError("missing trace or changed tool-step budget")
        replay = environment.replay_trace(trace, data_dir=data_dir)
        if not replay["valid"]:
            raise ValueError("environment trace failed independent replay")
        env = environment.ResearchEnvironment(data_dir=data_dir, max_steps=config["max_turns"])
        initial = env.reset(task)
        messages = train_policy.initial_messages(initial)
        if initial != episode.get("initial_observation") or canonical_hash(messages) != episode.get("initial_prompt_sha256"):
            raise ValueError("initial actor state/prompt changed")
        calls, invalid, interventions, prompt_tokens, completion_tokens = 0, 0, 0, 0, 0
        processed = 0
        for step in episode["steps"]:
            processed += 1
            if messages != step.get("messages"):
                raise ValueError("model history differs from actual prior observations")
            for name in ("prompt_tokens", "completion_tokens"):
                if type(step.get(name)) is not int or step[name] < 0:
                    raise ValueError("invalid token counts")
            prompt_tokens += step["prompt_tokens"]
            completion_tokens += step["completion_tokens"]
            if type(step.get("terminated")) is not bool:
                raise ValueError("missing generation termination flag")
            if not step["terminated"]:
                if "action" in step or "result" in step:
                    raise ValueError("truncated model response was executed")
                break
            try:
                parsed = train_policy.parse_action(step["raw_text"])
            except (ValueError, TypeError):
                parsed = None
            if manifest["arm"] == "B0":
                guard = train_policy.guard_action(parsed, messages)
                if step.get("model_original_action") != parsed or step.get("rule_intervention") != guard:
                    raise ValueError("B0 intervention does not match visible original model output")
                action = guard["action"]
                interventions += int(guard["intervened"])
            else:
                if "rule_intervention" in step or "model_original_action" in step:
                    raise ValueError("rules injected into a non-B0 policy")
                if parsed is None:
                    if "action" in step or "result" in step:
                        raise ValueError("invalid model JSON was silently repaired")
                    break
                action = parsed
            if action != step.get("action"):
                raise ValueError("actual action differs from parsed model/declared rule output")
            result = env.step(action)
            calls += 1
            invalid += result["observation"].get("status") == "invalid_action"
            if result != step.get("result"):
                raise ValueError("actual tool return/reward differs from independent execution")
            if result["done"]:
                break
            train_policy.append_observation(messages, step["raw_text"], action, result)
        if processed != len(episode["steps"]):
            raise ValueError("extra model calls after termination")
        rebuilt = env.export_trace()
        if replay_content(rebuilt) != replay_content(trace):
            raise ValueError("model actions do not reproduce the supplied tool trace")
        if episode.get("done") != rebuilt["done"]:
            raise ValueError("episode completion does not match environment")
        if rebuilt["done"] and episode.get("reward") != rebuilt["reward"]:
            raise ValueError("episode reward does not match environment")
        terminal = rebuilt.get("terminal") or {}
        checked.append({"task_id": task["id"], "seed": episode["seed"], "pair_id": task["pair_id"],
            "repeat": expected[(task["id"], episode["seed"])], "side": task["side"],
            "kind": task["task_kind"], "variant": task["variant"], "repair_setup": bool(task.get("setup_action")),
            "success": bool(terminal.get("passed")), "calls": calls, "invalid_calls": invalid,
            "action_attempts": len(episode["steps"]), "rule_interventions": interventions,
            "reported_prompt_tokens": prompt_tokens, "reported_completion_tokens": completion_tokens,
            "recorded_tool_wall_seconds": finite_nonnegative(trace["telemetry"]["wall_time_s"], "tool wall time"),
            "termination_reason": episode["termination_reason"], "terminal_dimensions": terminal.get("dimensions", {})})
    groups = {"answerable": [r for r in checked if r["side"] == "available"],
              "missing_in_authorized_view": [r for r in checked if r["side"] == "view_without_target"],
              "repair_setup": [r for r in checked if r["repair_setup"]],
              "observed_tool_error": [r for r in checked if r["invalid_calls"]],
              "text": [r for r in checked if r["kind"] == "text_evidence"],
              "numeric": [r for r in checked if r["kind"] == "finite_metric"]}
    paired = defaultdict(list)
    for row in checked:
        paired[(row["pair_id"], row["repeat"])].append(row)
    complete = [members for members in paired.values() if {r["side"] for r in members} == {"available", "view_without_target"}]
    return {"manifest": manifest, "manifest_sha256": file_hash(manifest_path), "episodes_sha256": file_hash(episodes_path),
            "draw_keys": sorted(actual_keys), "summary": summarize(checked),
            "coverage": {"development_tasks": len(tasks), "evaluated_tasks": len(selected),
                         "full_development": len(selected) == len(tasks)},
            "subgroups": {key: summarize(values) for key, values in groups.items()},
            "paired_both": {"complete_pair_draws": len(complete), "successes": sum(all(r["success"] for r in p) for p in complete),
                            "unique_complete_pairs": len({p[0]["pair_id"] for p in complete}),
                            "unpaired_draws": len(paired) - len(complete)},
            "cost": {**{name: sum(r[name] for r in checked) for name in ("calls", "invalid_calls", "action_attempts", "rule_interventions", "reported_prompt_tokens", "reported_completion_tokens", "recorded_tool_wall_seconds")},
                     "run_elapsed_seconds": finite_nonnegative(manifest["elapsed_seconds"], "elapsed seconds")},
            "episodes": checked}


def compare(runs: list[tuple[str, Path]], data_dir: Path) -> dict:
    if not runs or len({name for name, _ in runs}) != len(runs):
        raise ValueError("at least one uniquely named run is required")
    reports = {name: inspect_run(path, data_dir) for name, path in runs}
    baseline = next(iter(reports.values()))
    first = baseline["manifest"]
    for report in reports.values():
        manifest = report["manifest"]
        if any(manifest["bindings"][key] != first["bindings"][key] for key in COMMON_BINDINGS):
            raise ValueError("cannot compare different base/tasks/environment/prompt snapshots")
        if any(manifest["config"][key] != first["config"][key] for key in COMMON_CONFIG):
            raise ValueError("cannot compare different generation/tool/wall-clock budgets or sampling")
        if report["draw_keys"] != baseline["draw_keys"]:
            raise ValueError("task-and-seed draws differ; no silent intersection")
        for key in ("generation_pad_token_id", "generation_eos_token_id", "rollout_sampling"):
            if key not in manifest or key not in first or manifest[key] != first[key]:
                raise ValueError("generation termination/sampling configuration differs")
    output = {}
    for name, report in reports.items():
        manifest = report.pop("manifest")
        if name in train_policy.ARMS and name != manifest["arm"]:
            raise ValueError("reserved arm label differs from actual evaluation arm")
        report.pop("draw_keys")
        output[name] = {**report, "arm": manifest["arm"], "initial_adapter_sha256": manifest["bindings"].get("initial_adapter_sha256"),
                        "execution_code_sha256": manifest.get("code_sha256"),
                        "code_files_sha256": manifest.get("code_files_sha256"),
                        "generation_pad_token_id": manifest.get("generation_pad_token_id")}
    return {"schema": "mito.rp-grpo.evaluation-comparison.v1", "status": "completed",
            "is_sft_admission_gate": False, "algorithm_improvement_proven": False,
            "common_bindings": {key: first["bindings"][key] for key in COMMON_BINDINGS},
            "common_budgets": {key: first["config"][key] for key in COMMON_CONFIG}, "runs": output,
            "limitations": ["Failures remain in denominators; no intersection filtering or inferred missing outcomes.",
                "Metrics are descriptive and do not establish algorithm superiority, causal feedback use, or scientific generalization.",
                "Token counts and timing are recorded costs; this summary does not recompute tokenization or attest GPU execution.",
                "B0 is model plus explicit current deterministic guard; non-B0 model actions are not rewritten.",
                "Adapters may differ; execution code hashes are retained, not asserted identical. Do not attribute differences to one algorithm without controlled training provenance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="Label=path/to/run_manifest.json; repeat for each run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "rp_grpo/data")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output must be a new JSON file")
    runs = []
    for value in args.run:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path:
            raise ValueError("--run requires Label=manifest.json")
        runs.append((name, Path(path)))
    result = compare(runs, args.data_dir.resolve(strict=True))
    write_new(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
