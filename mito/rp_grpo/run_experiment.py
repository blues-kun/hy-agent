"""Portable B0-B3 launcher; plan only unless --execute is explicit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command_for(args):
    config = json.loads((ROOT / "rp_grpo/experiment_config.json").read_text())
    shared = config["shared"]
    if args.arm == "B0" and args.mode == "train":
        raise ValueError("B0 is an evaluation-only SFT-plus-rules baseline, not an RL training arm")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or ROOT / "runs" / f"{args.arm}-{args.mode}-{args.seed}-{stamp}"
    absolute = lambda path: str(Path(path).expanduser().resolve())
    argv = [sys.executable, "-m", "rp_grpo.train_policy", "--arm", args.arm, "--mode", args.mode,
            "--train-data", str(ROOT / "rp_grpo/data/tasks.jsonl"),
            "--scorers", str(ROOT / "rp_grpo/data/scorers.jsonl"),
            "--model-path", absolute(args.model_path or ROOT / "models/Qwen3-4B-Instruct-2507"),
            "--initial-adapter", absolute(args.initial_adapter or ROOT / "rp_grpo/checkpoints/tool_sft_v2"),
            "--sft-gate", absolute(args.sft_gate or ROOT / "rp_grpo/SFT_GATE.json"),
            "--output-dir", absolute(output), "--split", "dev" if args.mode == "evaluate" else "train",
            "--seed", str(args.seed), "--group-size", str(shared["group_size_per_environment"]),
            "--max-turns", str(shared["max_turns"]), "--max-new-tokens", str(shared["max_new_tokens_per_action"]),
            "--max-context-tokens", str(shared["max_context_tokens"]),
            "--max-steps", str(args.max_steps), "--max-seconds", str(args.max_seconds),
            "--learning-rate", str(shared["learning_rate"]), "--beta", str(shared["beta"]),
            "--clip-epsilon", str(shared["clip_epsilon"]),
            "--eta", str(0.0 if args.arm == "LOO" else shared["eta_B3"])]
    if args.mode == "evaluate":
        task_lines = (ROOT / "rp_grpo/data/tasks.jsonl").read_text().splitlines()
        count = sum(json.loads(line).get("split") == "dev" for line in task_lines if line.strip())
        argv.extend(["--max-eval-tasks", str(args.max_eval_tasks or count),
                     "--eval-rollouts", str(args.eval_rollouts)])
        if args.eval_sample:
            argv.append("--eval-sample")
    if args.gpu is not None:
        argv.extend(["--gpu", str(args.gpu)])
    if args.execute:
        argv.append("--execute")
    return argv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("B0", "B1", "B2", "B3", "LOO"), required=True)
    p.add_argument("--mode", choices=("validate", "evaluate", "train"), default="validate")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--initial-adapter", type=Path)
    p.add_argument("--sft-gate", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--gpu", type=int)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--max-eval-tasks", type=int, help="Defaults to the entire development split")
    p.add_argument("--eval-rollouts", type=int, default=1)
    p.add_argument("--eval-sample", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    command = command_for(args)
    print(json.dumps({"schema": "mito.rp-experiment-launch.v1", "package_root": str(ROOT),
                      "arm": args.arm, "mode": args.mode, "execute": args.execute,
                      "command": shlex.join(command), "production_changes": False}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
