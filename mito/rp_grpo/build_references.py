"""Execute two bounded rule routes, export only successful real tool histories.

These are automatic CPU rule demonstrations, NOT model or expert trajectories.
The policies below only inspect actor-visible observations, never hidden gold.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from collections import Counter

from .environment import ResearchEnvironment, SYSTEM_PROMPT, TOOL_SCHEMAS, DEFAULT_DATA, load_tasks, replay_trace
from .build_data import canonical_hash, file_hash, write_jsonl


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_system_prompt():
    return SYSTEM_PROMPT + "\n\nTool schema:\n" + canonical_json(TOOL_SCHEMAS) + '\nReturn exactly one JSON object with keys "tool" and "arguments". No markdown or hidden reasoning.'


def initial_messages(observation):
    return [{"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": canonical_json({"observation": observation})}]


def execute_reference(env, task, route):
    observation = env.reset(task)
    messages, samples = initial_messages(observation), []

    def invoke(tool, arguments):
        action = {"tool": tool, "arguments": arguments}
        target = {"role": "assistant", "content": canonical_json(action)}
        samples.append({"messages": copy.deepcopy([*messages, target]), "step_index": len(samples)})
        result = env.step(action)
        messages.append(target)
        if not result["done"]:
            messages.append({"role": "user", "content": canonical_json({"tool": tool, "observation": result["observation"]})})
        return result["observation"]

    claims, values = [], []
    if observation.get("measurement_request"):
        if route == "catalog_read":
            invoke("list_documents", {})
        request = {k: v for k, v in observation["measurement_request"].items() if k != "unit"}
        result = invoke("query_metrics", request)
        if result.get("status") == "ok":
            values = [{key: result[key] for key in ("result_id", "value", "unit")}]
    elif route == "search_read":
        result = invoke("search_evidence", {"query": observation["retrieval_query"]})
        for hit in result["hits"]:
            if not hit["exact_match"]:
                continue
            read = invoke("read_passage", {"evidence_id": hit["evidence_id"]})
            if read.get("status") == "ok" and observation["retrieval_query"].casefold() in read["text"].casefold():
                claims = [{"evidence_id": read["evidence_id"], "quote": read["text"]}]
                break
    elif route == "catalog_read":
        catalog = invoke("list_documents", {"limit": 100})
        tokens = set(re.findall(r"\w{3,}", observation["retrieval_query"].casefold()))
        # Ranking only uses actual visible catalog fields. A wrong candidate is
        # read and rejected based on its bytes; it is not a hidden gold lookup.
        candidates = sorted(catalog["passages"], key=lambda item: (-len(tokens & set(re.findall(r"\w{3,}", (item.get("section", "") + " " + item.get("title", "")).casefold()))), item["evidence_id"]))
        for candidate in candidates:
            read = invoke("read_passage", {"evidence_id": candidate["evidence_id"]})
            if read.get("status") == "ok" and observation["retrieval_query"].casefold() in read["text"].casefold():
                claims = [{"evidence_id": read["evidence_id"], "quote": read["text"]}]
                break
    else:
        raise ValueError("unknown reference route")
    invoke("finalize", {"answerability": "answerable" if claims or values else "insufficient",
                        "scope": "authorized_snapshot_only", "claims": claims, "measurements": values,
                        "reason": "返回已读取原文或实际计算结果。" if claims or values else "当前授权快照中未找到所需记录；不推断其他文献或原始实验的情况。"})
    trace = env.export_trace()
    if not trace["done"] or not trace["terminal"]["passed"]:
        raise ValueError(f"reference route failed: {task['id']} {route}: {trace.get('terminal')}")
    return trace, samples


def build(data_dir=DEFAULT_DATA, max_steps=64):
    data_dir = Path(data_dir)
    paths = [data_dir / "planner_sft.train.jsonl", data_dir / "planner_sft.dev.jsonl", data_dir / "reference_traces.jsonl", data_dir / "REFERENCE_MANIFEST.json"]
    if any(path.exists() for path in paths):
        raise ValueError("reference output already exists; use a new snapshot version")
    env = ResearchEnvironment(data_dir, max_steps=max_steps)
    traces, samples = [], {"train": [], "dev": []}
    for task in load_tasks(data_dir=data_dir):
        for route in ("search_read", "catalog_read"):
            trace, rows = execute_reference(env, task, route)
            replay = replay_trace(trace, data_dir)
            if not replay["valid"] or not replay["passed"]:
                raise ValueError("independent replay rejected a reference")
            identifier = "ref-" + canonical_hash([task["id"], route])[:24]
            traces.append({"id": identifier, "task_id": task["id"], "split": task["split"], "pair_id": task["pair_id"],
                           "route": route, "target_provenance": "executed_rule_demonstration", "model_generated": False,
                           "environment_trace": trace})
            for index, row in enumerate(rows):
                samples[task["split"]].append({"id": f"{identifier}-step-{index}", "task_id": task["id"], "split": task["split"],
                    "pair_id": task["pair_id"], "group_id": task["group_id"], "route": route, "reference_trace_id": identifier,
                    "reference_trace_sha256": trace["trace_sha256"], "trace_step_index": index,
                    "messages": row["messages"], "target_provenance": "executed_rule_demonstration", "expert_trajectory": False,
                    "environment_task_sha256": trace["task_sha256"], "input_char_count": sum(len(m["content"]) for m in row["messages"][:-1]),
                    "supervision": "last_assistant_action_only; preceding tool observations are context, not targets"})
    write_jsonl(paths[0], samples["train"])
    write_jsonl(paths[1], samples["dev"])
    write_jsonl(paths[2], traces)
    report = {"schema_version": "rp-rule-reference.v1", "task_count": len(load_tasks(data_dir=data_dir)),
              "trace_count": len(traces), "all_actual_executions_passed": True, "all_independent_replays_passed": True,
              "sft_rows": {key: len(value) for key, value in samples.items()}, "max_steps": max_steps,
              "route_counts": dict(Counter(row["route"] for row in traces)), "route_cost_proxy_mean": {},
              "max_input_chars": max(row["input_char_count"] for rows in samples.values() for row in rows),
              "max_actual_calls": max(len(row["environment_trace"]["steps"]) for row in traces),
              "dataset_manifest_sha256": file_hash(data_dir / "MANIFEST.json"),
              "files": {path.name: file_hash(path) for path in paths[:3]},
              "limitations": ["Automatic rule demonstrations are not expert or model trajectories.",
                 "Catalog-to-candidate-read is a genuinely different executable route, often dominated in cost by exact search.",
                 "These literal tasks validate multi-path engineering, not the necessity or scientific benefit of RL.",
                 "cost_proxy is fixed local action/observation cost, not API charges; actual wall-clock and call counts are telemetry.",
                 "Long histories must pass tokenizer length checks; do not silently truncate source evidence or final targets.",
                 "64-step reference budget is common to both routes; tighter model rollout budgets may fail and must be evaluated separately."]}
    for route in report["route_counts"]:
        costs = [row["environment_trace"]["cost_proxy"] for row in traces if row["route"] == route]
        report["route_cost_proxy_mean"][route] = sum(costs) / len(costs)
    paths[3].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_budget_audit(data_dir=DEFAULT_DATA, actor_budget=6):
    """Label every reference route against the separately chosen actor budget.

    This never truncates trajectories or claims that a successful 64-step rule
    route was solved by a 6-step policy. Token-length filtering belongs to the
    actual tokenizer/trainer and must have its own exclusion report.
    """
    from .build_data import records
    data_dir = Path(data_dir)
    target = data_dir / "REFERENCE_BUDGET.json"
    if target.exists():
        raise ValueError("budget audit already exists")
    refs = records(data_dir / "reference_traces.jsonl")
    rows = [{"reference_id": row["id"], "task_id": row["task_id"], "split": row["split"], "route": row["route"],
             "actual_actor_calls": len(row["environment_trace"]["steps"]),
             "within_actor_budget": len(row["environment_trace"]["steps"]) <= actor_budget,
             "reference_execution_budget": row["environment_trace"]["max_steps"]} for row in refs]
    counts = Counter((row["split"], row["route"], row["within_actor_budget"]) for row in rows)
    result = {"actor_step_budget": actor_budget, "reference_file_sha256": file_hash(data_dir / "reference_traces.jsonl"),
              "counts": [{"split": key[0], "route": key[1], "within_actor_budget": key[2], "count": value} for key, value in sorted(counts.items())],
              "routes": rows, "token_length_audit": "not performed here; use actual tokenizer before SFT and record every excluded sample; do not truncate",
              "claim_boundary": "Rule-route feasibility is not model success, GRPO improvement, or proof that a longer route is preferable."}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "routes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument("--actor-budget", type=int, default=6)
    args = parser.parse_args()
    result = write_budget_audit(args.data_dir, args.actor_budget) if args.budget_only else build(args.data_dir, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))
