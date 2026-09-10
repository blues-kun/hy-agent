"""Supervised warm-up on executed tool demonstrations, NOT expert verdict SFT.

Default is validation only. Only the LAST assistant action is supervised;
historical assistant actions and all observations are context. Source annotations,
production models, and prior runs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


class LengthExceeded(ValueError):
    def __init__(self, identifier, tokens, limit):
        self.identifier, self.tokens, self.limit = identifier, tokens, limit
        super().__init__(f"{identifier} has {tokens} tokens > {limit}; evidence must not be silently truncated")


def verify_reference_files(train_path, dev_path, tasks_path):
    """Only consume the manifest-bound, actually executed demonstrations."""
    helper = verifier_helpers()
    directory = tasks_path.resolve(strict=True).parent
    from rp_grpo.environment import ResearchEnvironment
    ResearchEnvironment(data_dir=directory)  # validates task/corpus/scorer hashes
    manifest_path = directory / "REFERENCE_MANIFEST.json"
    manifest = helper.strict_json(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("all_actual_executions_passed") is not True
            or manifest.get("all_independent_replays_passed") is not True
            or manifest.get("dataset_manifest_sha256") != helper.sha256_file(directory / "MANIFEST.json")):
        raise ValueError("Reference demonstrations are not bound to this verified data snapshot")
    for path in (train_path, dev_path, directory / "reference_traces.jsonl"):
        if path.resolve(strict=True).parent != directory:
            raise ValueError("Demonstrations must be in the same manifest-bound directory")
        if helper.sha256_file(path) != manifest.get("files", {}).get(path.name):
            raise ValueError(f"Reference data hash mismatch: {path.name}")
    return helper.sha256_file(manifest_path)


def verifier_helpers():
    spec = importlib.util.spec_from_file_location("mito_package_verifier_helpers", ROOT / "code/train_verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_index(path: Path, tasks: dict, *, data_dir: Path) -> dict:
    """Re-execute references before trusting their claimed success or hashes.

    Keep only hashes of each exact actor-visible history. Scorer labels, reward,
    terminal metadata and timing must never be added to the model's messages.
    """
    from rp_grpo.environment import replay_trace
    from rp_grpo.train_policy import initial_messages, append_observation, canonical_json
    helper = verifier_helpers()
    references = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = helper.strict_json(line)
            if not isinstance(row, dict):
                raise ValueError(f"Malformed reference at {path.name}:{line_no}")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in references:
                raise ValueError("Missing/duplicate reference id")
            task = tasks.get(row.get("task_id"))
            trace = row.get("environment_trace")
            if (task is None or not isinstance(trace, dict)
                    or row.get("target_provenance") != "executed_rule_demonstration"
                    or row.get("model_generated") is not False
                    or row.get("route") not in {"search_read", "catalog_read"}):
                raise ValueError("Reference is not an identified executed-rule trajectory")
            for key in ("split", "pair_id"):
                if row.get(key) != task[key]:
                    raise ValueError("Reference metadata differs from its registered task")
            if (trace.get("task_id") != task["id"]
                    or trace.get("task_sha256") != helper.canonical_hash(task)
                    or trace.get("done") is not True
                    or (trace.get("terminal") or {}).get("passed") is not True):
                raise ValueError("Reference lacks a successful terminal bound to its registered task")
            replay = replay_trace(trace, data_dir=data_dir)
            if replay.get("valid") is not True or replay.get("passed") is not True:
                raise ValueError(f"Reference replay rejected {identifier}: {replay.get('errors')}")
            history = initial_messages(trace["initial_observation"])
            expected_steps = []
            steps = trace["steps"]
            for index, step in enumerate(steps):
                if step["index"] != index:
                    raise ValueError("Noncontiguous reference step index")
                action = step["action"]
                action_text = canonical_json(action)
                expected_steps.append({
                    "messages_sha256": helper.canonical_hash([*history, {"role": "assistant", "content": action_text}]),
                    "input_char_count": sum(len(m["content"]) for m in history),
                    "tool": action["tool"],
                    "is_terminal": index == len(steps) - 1,
                    "answerability": action["arguments"].get("answerability") if index == len(steps) - 1 else None,
                })
                append_observation(history, action_text, action, {"observation": step["observation"]})
            if not expected_steps or expected_steps[-1]["tool"] not in {"finalize", "stop"}:
                raise ValueError("Reference does not end with an explicit terminal action")
            references[identifier] = {
                "task_id": task["id"], "split": task["split"], "pair_id": task["pair_id"],
                "group_id": task["group_id"], "side": task["side"], "route": row["route"],
                "task_sha256": trace["task_sha256"], "trace_sha256": trace["trace_sha256"],
                "steps": expected_steps, "independent_replay_passed": True,
                "reference_max_steps": trace["max_steps"],
            }
    if not references:
        raise ValueError("Empty reference trace file")
    return references


def read_demonstrations(path: Path, split: str, tasks: dict, references: dict) -> list[dict]:
    helper = verifier_helpers()
    rows, seen, seen_steps = [], set(), set()
    # Stream the large JSONL rather than duplicating it as read_text/splitlines.
    with path.open(encoding="utf-8") as stream:
        lines = enumerate(stream, 1)
        for line_no, line in lines:
            if not line.strip():
                continue
            row = helper.strict_json(line)
            _validate_demonstration(row, path, line_no, split, tasks, references, seen, seen_steps, helper)
            rows.append(row)
    if not rows:
        raise ValueError("Empty demonstration data")
    expected = {(identifier, index) for identifier, ref in references.items() if ref["split"] == split
                for index in range(len(ref["steps"]))}
    if seen_steps != expected:
        raise ValueError("Demonstration file does not cover every bound reference step before length filtering")
    return rows


def _validate_demonstration(row, path, line_no, split, tasks, references, seen, seen_steps, helper):
    if not isinstance(row, dict):
        raise ValueError("Malformed demonstration")
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier or identifier in seen:
        raise ValueError(f"Missing/duplicate demonstration id at {path.name}:{line_no}")
    seen.add(identifier)
    task = tasks.get(row.get("task_id"))
    if task is None or task.get("split") != split or row.get("split") != split:
        raise ValueError("Demonstration is not bound to a task in the requested split")
    if row.get("target_provenance") != "executed_rule_demonstration":
        raise ValueError("Expected explicitly identified executed-rule demonstration")
    reference = references.get(row.get("reference_trace_id"))
    index = row.get("trace_step_index")
    if (reference is None or reference.get("independent_replay_passed") is not True
            or isinstance(index, bool) or not isinstance(index, int)
            or not 0 <= index < len(reference["steps"])):
        raise ValueError("Demonstration has no independently replayed reference step")
    for key in ("task_id", "split", "pair_id", "group_id", "route"):
        if row.get(key) != reference[key]:
            raise ValueError("Demonstration metadata differs from its reference step")
    if (row.get("reference_trace_sha256") != reference["trace_sha256"]
            or row.get("environment_task_sha256") != reference["task_sha256"]
            or row.get("expert_trajectory") is not False):
        raise ValueError("Demonstration reference/task hash or provenance mismatch")
    step_key = (row["reference_trace_id"], index)
    if step_key in seen_steps:
        raise ValueError("Duplicate demonstration for the same reference step")
    seen_steps.add(step_key)
    messages = row.get("messages")
    if (not isinstance(messages, list) or len(messages) < 3
            or not isinstance(messages[0], dict) or not isinstance(messages[-1], dict)
            or messages[0].get("role") != "system" or messages[-1].get("role") != "assistant"):
        raise ValueError("Expected exact runtime history followed by one assistant target")
    if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
           or not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Malformed or non-runtime role in demonstration")
    if any(m["role"] == "system" for m in messages[1:]):
        raise ValueError("Extra system message in demonstration")
    action = helper.strict_json(messages[-1]["content"])
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"} or not isinstance(action["tool"], str) or not isinstance(action["arguments"], dict):
        raise ValueError("Target must match runtime tool-action envelope")
    from rp_grpo.environment import TOOL_SCHEMAS, _validate
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}
    if action["tool"] not in schemas:
        raise ValueError("Demonstration targets a nonexistent tool")
    _validate(action["arguments"], schemas[action["tool"]])
    from rp_grpo.train_policy import build_system_prompt
    if messages[0]["content"] != build_system_prompt():
        raise ValueError("SFT prompt differs from the actual rollout contract")
    expected = reference["steps"][index]
    if (helper.canonical_hash(messages) != expected["messages_sha256"]
            or row.get("input_char_count") != expected["input_char_count"]):
        raise ValueError("Demonstration messages differ from actual replayed history/action")


def _template_token_ids(value):
    """Transformers versions return either token ids or a BatchEncoding.

    Normalize the container, never infer the completion boundary by character
    offsets or by searching for a repeated answer string in the history.
    """
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (not isinstance(value, (list, tuple)) or not value
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value)):
        raise ValueError("Expected one unpadded sequence of tokenizer input_ids")
    return list(value)


def encode_last_action(row: dict, tokenizer, max_length: int) -> dict:
    """Use the exact actor prompt, followed only by action JSON and one EOS.

    A full training chat template may insert a thinking prefix or normalize a
    historical assistant differently. Never use that to infer target offsets.
    """
    from rp_grpo.train_policy import format_prompt
    messages = row["messages"]
    prompt = format_prompt(tokenizer, messages[:-1])
    if not isinstance(prompt, str):
        raise ValueError("Runtime prompt formatter must return text")
    prefix = _template_token_ids(tokenizer(prompt, add_special_tokens=False))
    if tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt:
        raise ValueError("Tokenizer changes runtime prompt bytes")
    action_text = messages[-1]["content"]
    target = _template_token_ids(tokenizer(action_text, add_special_tokens=False))
    if tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False) != action_text:
        raise ValueError("Tokenizer changes target action bytes")
    eos = tokenizer.eos_token_id
    if isinstance(eos, bool) or not isinstance(eos, int) or eos < 0 or eos in target:
        raise ValueError("Target must contain exactly one terminal EOS, appended by the trainer")
    completion = [*target, eos]
    full = [*prefix, *completion]
    if len(full) > max_length:
        raise LengthExceeded(row["id"], len(full), max_length)
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + completion}


def encode_demonstrations(rows, tokenizer, max_length, *, skip_overlength=False, allow_empty=False):
    encoded, retained, excluded = [], [], []
    for row in rows:
        try:
            encoded.append(encode_last_action(row, tokenizer, max_length))
            retained.append(row)
        except LengthExceeded as exc:
            if not skip_overlength:
                raise
            excluded.append({"id": row["id"], "task_id": row["task_id"],
                             "reference_trace_id": row.get("reference_trace_id"),
                             "trace_step_index": row.get("trace_step_index"), "route": row.get("route"),
                             "tokens": exc.tokens, "limit": exc.limit, "reason": "overlength_not_truncated"})
    if not encoded and not allow_empty:
        raise ValueError("No complete, untruncated demonstration remains")
    return encoded, retained, excluded


def filter_reference_budget(rows, references, max_reference_steps):
    if isinstance(max_reference_steps, bool) or not isinstance(max_reference_steps, int) or max_reference_steps <= 0:
        raise ValueError("Reference actor-step budget must be a positive integer")
    rejected = {row["reference_trace_id"] for row in rows
                if len(references[row["reference_trace_id"]]["steps"]) > max_reference_steps}
    excluded = []
    for identifier in sorted(rejected):
        reference = references[identifier]
        excluded.append({"reference_trace_id": identifier, "task_id": reference["task_id"],
                         "route": reference["route"], "split": reference["split"],
                         "actor_steps": len(reference["steps"]), "actor_step_budget": max_reference_steps,
                         "excluded_step_records": sum(row["reference_trace_id"] == identifier for row in rows),
                         "reason": "whole_trajectory_exceeds_actor_step_budget"})
    return [row for row in rows if row["reference_trace_id"] not in rejected], excluded


def encode_complete_trajectories(rows, references, tokenizer, max_length, *, skip_overlength):
    """A single overlength step excludes its whole reference, never truncates."""
    supplied = defaultdict(set)
    for row in rows:
        supplied[row["reference_trace_id"]].add(row["trace_step_index"])
    for identifier, indices in supplied.items():
        if indices != set(range(len(references[identifier]["steps"]))):
            raise ValueError("Length filtering requires complete reference trajectories")
    encoded, retained, excluded = encode_demonstrations(rows, tokenizer, max_length,
        skip_overlength=skip_overlength, allow_empty=True)
    rejected = {row["reference_trace_id"] for row in excluded}
    output, output_rows = [], []
    for row, item in zip(retained, encoded):
        if row["reference_trace_id"] in rejected:
            excluded.append({"id": row["id"], "task_id": row["task_id"],
                             "reference_trace_id": row["reference_trace_id"],
                             "trace_step_index": row["trace_step_index"], "route": row["route"],
                             "tokens": len(item["input_ids"]), "limit": max_length,
                             "reason": "whole_trajectory_excluded_due_to_another_overlength_step"})
        else:
            output.append(item)
            output_rows.append(row)
    return output, output_rows, excluded


def coverage_report(rows, references, tasks, split):
    """Report usable terminal/trajectory supervision, not just surviving tasks.

    A later example still contains the real earlier history even when an earlier
    *training target* was excluded. It may teach that action, but does not mean
    that every action of the trajectory received supervision.
    """
    retained = defaultdict(set)
    action_counts = Counter()
    for row in rows:
        retained[row["reference_trace_id"]].add(row["trace_step_index"])
        step = references[row["reference_trace_id"]]["steps"][row["trace_step_index"]]
        action_counts[step["tool"]] += 1
    trace_rows = []
    routes = defaultdict(Counter)
    terminal_by_side, complete_by_side = Counter(), Counter()
    terminal_answerability = Counter()
    task_any, task_terminal, task_complete = set(), set(), set()
    for identifier, reference in sorted(references.items()):
        if reference["split"] != split:
            continue
        indices = retained[identifier]
        expected_indices = set(range(len(reference["steps"])))
        missing = expected_indices - indices
        terminal_index = len(reference["steps"]) - 1
        terminal_retained = terminal_index in indices
        complete = not missing
        detail = {"reference_trace_id": identifier, "task_id": reference["task_id"],
                  "pair_id": reference["pair_id"], "side": reference["side"], "route": reference["route"],
                  "expected_steps": len(expected_indices), "retained_steps": len(indices),
                  "missing_step_indices": sorted(missing), "terminal_step_index": terminal_index,
                  "terminal_retained": terminal_retained, "complete_trajectory_supervision": complete,
                  "reference_execution_budget": reference["reference_max_steps"]}
        trace_rows.append(detail)
        route = routes[reference["route"]]
        route["reference_traces"] += 1
        route["retained_steps"] += len(indices)
        route["traces_with_any_step"] += bool(indices)
        route["traces_with_terminal_step"] += terminal_retained
        route["complete_trajectories"] += complete
        route["fully_excluded_traces"] += not indices
        if indices:
            task_any.add(reference["task_id"])
        if terminal_retained:
            task_terminal.add(reference["task_id"])
            terminal_by_side[reference["side"]] += 1
            terminal_answerability[reference["steps"][terminal_index]["answerability"] or "stop"] += 1
        if complete:
            task_complete.add(reference["task_id"])
            complete_by_side[reference["side"]] += 1
    expected_tasks = {task["id"] for task in tasks.values() if task["split"] == split}
    pair_tasks = defaultdict(set)
    for task_id in expected_tasks:
        pair_tasks[tasks[task_id]["pair_id"]].add(task_id)
    # Pair coverage means ALL registered sides, not two arbitrary samples.
    pair_rows = []
    for pair_id, members in sorted(pair_tasks.items()):
        sides = {tasks[task_id]["side"] for task_id in members}
        has_two_sides = len(members) == len(sides) == 2
        pair_rows.append({"pair_id": pair_id, "registered_sides": sorted(sides),
                          "both_sides_with_terminal": has_two_sides and members <= task_terminal,
                          "both_sides_with_complete_trajectory": has_two_sides and members <= task_complete,
                          "missing_terminal_task_ids": sorted(members - task_terminal),
                          "missing_complete_trajectory_task_ids": sorted(members - task_complete)})
    return {"reference_traces": len(trace_rows), "retained_step_records": len(rows),
            "tasks_before_filter": len(expected_tasks), "tasks_with_any_step": len(task_any),
            "tasks_with_terminal_step": len(task_terminal), "tasks_with_complete_trajectory": len(task_complete),
            "terminal_step_records": sum(item["terminal_retained"] for item in trace_rows),
            "complete_trajectories": sum(item["complete_trajectory_supervision"] for item in trace_rows),
            "missing_terminal_task_ids": sorted(expected_tasks - task_terminal),
            "missing_complete_trajectory_task_ids": sorted(expected_tasks - task_complete),
            "retained_action_counts": dict(sorted(action_counts.items())),
            "terminal_records_by_side": dict(sorted(terminal_by_side.items())),
            "terminal_answerability_counts": dict(sorted(terminal_answerability.items())),
            "complete_trajectories_by_side": dict(sorted(complete_by_side.items())),
            "routes": {key: dict(value) for key, value in sorted(routes.items())},
            "pairs_before_filter": len(pair_rows),
            "pairs_with_both_sides_terminal": sum(item["both_sides_with_terminal"] for item in pair_rows),
            "pairs_with_both_sides_complete_trajectory": sum(item["both_sides_with_complete_trajectory"] for item in pair_rows),
            "pairs": pair_rows, "traces": trace_rows,
            "interpretation": "Step supervision coverage, not model success; partial surviving histories do not establish complete-trajectory learning. Reference execution budget is separate from the model rollout budget."}


def collate(batch, pad_token_id):
    import torch
    width = max(len(row["input_ids"]) for row in batch)
    return {key: torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in batch], dtype=torch.long)
            for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train"), default="validate")
    p.add_argument("--train-data", type=Path, default=ROOT / "rp_grpo/data/planner_sft.train.jsonl")
    p.add_argument("--dev-data", type=Path, default=ROOT / "rp_grpo/data/planner_sft.dev.jsonl")
    p.add_argument("--tasks", type=Path, default=ROOT / "rp_grpo/data/tasks.jsonl")
    p.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    p.add_argument("--initial-adapter", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--gpu", type=int)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--skip-overlength", action="store_true", help="Explicitly exclude and report long examples; never truncate")
    p.add_argument("--validation-output", type=Path)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-reference-steps", type=int, default=6,
                   help="Keep only complete reference trajectories within this actor-step budget; 64 is explicit opt-in")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260909)
    args = p.parse_args(argv)
    if any(x <= 0 for x in (args.max_length, args.max_reference_steps, args.max_steps, args.max_seconds, args.grad_accum, args.lora_r)) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Training bounds must be finite and positive")
    if args.mode == "train" and args.execute and args.gpu is None:
        raise ValueError("An explicit idle physical GPU is required")
    if args.validation_output and args.validation_output.exists():
        raise ValueError("Refusing to overwrite a prior validation report")
    helper = verifier_helpers()
    reference_hash = verify_reference_files(args.train_data, args.dev_data, args.tasks)
    tasks_list = [helper.strict_json(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    tasks = {row["id"]: row for row in tasks_list}
    if len(tasks) != len(tasks_list):
        raise ValueError("Duplicate task id")
    groups = {}
    for task in tasks_list:
        for key in ("group_id", "pair_id"):
            identifier = (key, task[key])
            if identifier in groups and groups[identifier] != task["split"]:
                raise ValueError("Task family or pair crossed a split")
            groups[identifier] = task["split"]
    reference_path = args.tasks.resolve(strict=True).parent / "reference_traces.jsonl"
    references = load_reference_index(reference_path, tasks, data_dir=reference_path.parent)
    if {item["task_id"] for item in references.values()} != set(tasks):
        raise ValueError("Reference file does not cover every registered task")
    train = read_demonstrations(args.train_data, "train", tasks, references)
    dev = read_demonstrations(args.dev_data, "dev", tasks, references)
    if set(row["id"] for row in train) & set(row["id"] for row in dev):
        raise ValueError("Demonstration id crossed splits")
    snapshot = helper.model_snapshot(args.model_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    raw_counts = {"train": len(train), "dev": len(dev)}
    train, train_budget_excluded = filter_reference_budget(train, references, args.max_reference_steps)
    dev, dev_budget_excluded = filter_reference_budget(dev, references, args.max_reference_steps)
    after_budget_counts = {"train": len(train), "dev": len(dev)}
    train_encoded, train, train_excluded = encode_complete_trajectories(train, references, tokenizer, args.max_length,
        skip_overlength=args.skip_overlength)
    dev_encoded, dev, dev_excluded = encode_complete_trajectories(dev, references, tokenizer, args.max_length,
        skip_overlength=args.skip_overlength)
    from rp_grpo.train_policy import build_system_prompt
    report = {"schema": "mito.rp-planner-sft.v1", "mode": args.mode,
              "status": "validated" if train and dev else "blocked_no_usable_demonstrations", "training_started": False,
              "target_provenance": "executed_rule_demonstration_not_expert_trajectory",
              "model": snapshot, "prompt_sha256": helper.canonical_hash(build_system_prompt()),
              "train_sha256": helper.sha256_file(args.train_data), "dev_sha256": helper.sha256_file(args.dev_data),
              "tasks_sha256": helper.sha256_file(args.tasks),
              "reference_manifest_sha256": reference_hash,
              "reference_traces_sha256": helper.sha256_file(reference_path),
              "independently_replayed_reference_traces": len(references),
              "all_reference_steps_bound_before_filter": True,
              "records_before_filters": raw_counts,
              "records_before_length_check": after_budget_counts,
              "reference_step_policy": {"max_actor_steps": args.max_reference_steps, "whole_trajectory_filter": True,
                                        "note": "Reference execution budget (64) is distinct from the default model actor budget (6)."},
              "excluded_reference_step_budget": {"train": train_budget_excluded, "dev": dev_budget_excluded},
              "excluded_reference_step_budget_counts": {"train": raw_counts["train"] - after_budget_counts["train"],
                                                        "dev": raw_counts["dev"] - after_budget_counts["dev"]},
              "excluded_overlength": {"train": train_excluded, "dev": dev_excluded},
              "excluded_counts": {"train": len(train_excluded), "dev": len(dev_excluded)},
              "train_records": len(train), "dev_records": len(dev),
              "retained_task_counts": {"train": len({r["task_id"] for r in train}), "dev": len({r["task_id"] for r in dev})},
              "retained_task_counts_meaning": "At least one surviving step, not complete trajectory or terminal coverage; see supervision_coverage",
              "supervision_coverage": {"train": coverage_report(train, references, tasks, "train"),
                                       "dev": coverage_report(dev, references, tasks, "dev")},
              "length_policy": {"max_length": args.max_length, "skip_overlength": args.skip_overlength,
                                "applied_to": ["train", "dev"], "truncation": False, "whole_trajectory_filter": True},
              "target_encoding": "exact_runtime_prompt_tokens + independently_encoded_action_JSON + exactly_one_EOS",
              "train_task_families": len({tasks[row["task_id"]]["group_id"] for row in train}),
              "max_tokens": max((len(row["input_ids"]) for row in train_encoded + dev_encoded), default=0),
              "supervised_train_tokens": sum(sum(x != -100 for x in row["labels"]) for row in train_encoded),
              "history_and_observation_tokens_supervised": False, "truncated_records": 0,
              "requires_actual_policy_gate_after_training": True, "versions": helper.installed_versions()}
    report["warnings"] = []
    for split, coverage in report["supervision_coverage"].items():
        if coverage["complete_trajectories"] < coverage["reference_traces"]:
            report["warnings"].append(f"{split}: budget/length filtering excludes whole reference trajectories; no partial training trajectory is retained")
        if coverage["tasks_with_terminal_step"] < coverage["tasks_before_filter"]:
            report["warnings"].append(f"{split}: some tasks have no retained terminal supervision; inspect missing_terminal_task_ids")
    if args.validation_output:
        if args.validation_output.exists():
            raise ValueError("Refusing to overwrite a prior validation report")
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        helper.write_json(args.validation_output, report)
    if not train or not dev:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise ValueError("No complete, untruncated demonstration remains in train or dev; training is blocked")
    if args.mode != "train" or not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    output = helper.output_path(args.output_dir)
    started = time.monotonic()
    with helper.gpu_lease(args.gpu) as gpu:
        output.mkdir(parents=True, exist_ok=False)
        report.update(status="running", training_started=True, gpu=gpu, seed=args.seed)
        helper.write_json(output / "run_manifest.json", report)
        try:
            import torch
            from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
            from peft import PeftModel, LoraConfig, get_peft_model
            set_seed(args.seed)
            base = AutoModelForCausalLM.from_pretrained(str(args.model_path), dtype=torch.bfloat16,
                local_files_only=True, trust_remote_code=False, attn_implementation="sdpa")
            if args.initial_adapter:
                config = helper.strict_json((args.initial_adapter / "adapter_config.json").read_text())
                if config.get("peft_type") != "LORA" or "Qwen3-4B-Instruct-2507" not in str(config.get("base_model_name_or_path", "")):
                    raise ValueError("Initial adapter is not bound to the required model family")
                model = PeftModel.from_pretrained(base, str(args.initial_adapter), is_trainable=True)
                report["initial_adapter_sha256"] = helper.sha256_file(args.initial_adapter / "adapter_model.safetensors")
            else:
                model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                    lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))
            model.config.use_cache = False
            model.enable_input_require_grads()

            class Callback(TrainerCallback):
                def on_step_end(self, args_, state, control, **kwargs):
                    if time.monotonic() - started >= args.max_seconds:
                        control.should_training_stop = control.should_save = True
                    return control

                def on_log(self, args_, state, control, logs=None, **kwargs):
                    values = {key: value if not isinstance(value, float) or math.isfinite(value) else None
                              for key, value in (logs or {}).items()}
                    helper.append_json(output / "metrics.jsonl", {"step": state.global_step,
                        "elapsed_seconds": time.monotonic() - started, **values})

            if time.monotonic() - started >= args.max_seconds:
                raise RuntimeError("Model loading exhausted the training budget")
            config = TrainingArguments(output_dir=str(output / "checkpoints"), max_steps=args.max_steps,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
                bf16=True, tf32=True, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                save_steps=25, eval_steps=25, eval_strategy="steps", save_total_limit=3,
                logging_steps=1, seed=args.seed, data_seed=args.seed, report_to=[],
                push_to_hub=False, remove_unused_columns=False, disable_tqdm=True)
            trainer = Trainer(model=model, args=config, train_dataset=train_encoded, eval_dataset=dev_encoded,
                data_collator=lambda batch: collate(batch, tokenizer.pad_token_id), callbacks=[Callback()])
            outcome = trainer.train()
            trainer.save_model(str(output / "adapter"))
            tokenizer.save_pretrained(output / "adapter")
            trainer.save_state()
            report.update(status="completed", steps=trainer.state.global_step,
                ended_by="wall_clock" if time.monotonic() - started >= args.max_seconds else "configured_steps",
                adapter_sha256=helper.sha256_file(output / "adapter/adapter_model.safetensors"),
                adapter_config_sha256=helper.sha256_file(output / "adapter/adapter_config.json"),
                metrics=outcome.metrics)
        except BaseException as exc:
            report.update(status="failed", error_type=type(exc).__name__, training_started=True)
            helper.write_json(output / "run_manifest.json", report)
            raise
        helper.write_json(output / "run_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
