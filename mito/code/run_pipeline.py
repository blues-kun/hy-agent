"""Portable offline SFT -> base/dev -> adapter/dev -> paired comparison launcher.

Default is plan-only. --execute is required to load a model or create a run.
Only child process groups created here may be terminated at the global deadline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def commands(args, root, output):
    code = root / "code"
    model = root / "models/Qwen3-4B-Instruct-2507"
    data = root / "data"
    shared = [sys.executable, str(code / "train_verifier.py"), "--model-path", str(model),
              "--max-length", str(args.max_length), "--max-new-tokens", str(args.max_new_tokens),
              "--seed", str(args.seed)]
    datasets = ["--train-data", str(data / "train.jsonl"), "--eval-data", str(data / "dev.jsonl")]
    jobs = [("validate", shared + ["--mode", "validate"] + datasets)]
    training = shared + ["--mode", "train"] + datasets + [
        "--gpu", str(args.gpu), "--output-dir", str(output / "sft"),
        "--epochs", str(args.epochs), "--learning-rate", str(args.learning_rate),
        "--batch-size", "1", "--grad-accum", "8", "--lora-r", "16", "--save-steps", "50"]
    if args.smoke:
        training += ["--max-steps", "2"]
    jobs.append(("sft", training))
    for name in ("base-dev", "adapter-dev"):
        command = shared + ["--mode", "evaluate", "--eval-data", str(data / "dev.jsonl"),
                            "--gpu", str(args.gpu), "--output-dir", str(output / name)]
        if name == "adapter-dev":
            command += ["--adapter", str(output / "sft/adapter")]
        if args.smoke:
            command += ["--limit", "2"]
        jobs.append((name, command))
    jobs.append(("compare", [sys.executable, str(code / "compare_verifier_finetune.py"),
                             "--base", str(output / "base-dev/predictions.jsonl"),
                             "--finetuned", str(output / "adapter-dev/predictions.jsonl"),
                             "--train", str(data / "train.jsonl"), "--out", str(output / "comparison.json")]))
    return jobs


def stop_owned_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        child.wait(timeout=10)


def write_status(path, status):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="Explicit physical idle GPU index on the destination server")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two optimizer steps and two dev rows only; not a full experiment")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)
    if not 0 < args.max_hours <= 8 or args.gpu < 0 or not 1 <= args.epochs <= 3:
        parser.error("Use <=8 positive hours, nonnegative GPU, and 1-3 epochs")
    root = args.package_root.resolve(strict=True)
    for name in ("data/train.jsonl", "data/dev.jsonl", "code/train_verifier.py", "code/compare_verifier_finetune.py", "models/Qwen3-4B-Instruct-2507/config.json"):
        if not (root / name).is_file():
            parser.error(f"Missing package file: {name}")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or root / "runs" / (suffix + ("-smoke" if args.smoke else "-sft"))).resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing runs are never overwritten")
    jobs = commands(args, root, output)
    if not args.execute:
        import shlex
        print("PLAN ONLY: no GPU use, no model load, no run files created.")
        for name, command in jobs:
            print(name + ": " + shlex.join(command))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    begin = time.monotonic()
    deadline = begin + args.max_hours * 3600
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "pid": os.getpid(), "gpu": args.gpu, "smoke_only": args.smoke,
             "production_ready": False, "maximum_hours": args.max_hours, "stages": []}
    child = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminate_launcher(signum, frame):
        raise KeyboardInterrupt("Launcher received SIGTERM; cancelling its own child only")
    signal.signal(signal.SIGTERM, terminate_launcher)
    try:
        for name, planned in jobs:
            remaining = int(deadline - time.monotonic())
            if remaining < 180:
                raise TimeoutError("Global budget exhausted; prior artifacts are retained")
            command = list(planned)
            if name in {"sft", "base-dev", "adapter-dev"}:
                # Reserve budget for later stages; the outer timeout remains authoritative.
                stages_left = {"sft": 3, "base-dev": 2, "adapter-dev": 1}[name]
                stage_budget = min(7200, (remaining - 60) // stages_left)
                if stage_budget <= 120:
                    raise TimeoutError("Insufficient safe stage budget")
                command += ["--max-seconds", str(stage_budget), "--safety-margin-seconds", "120"]
            else:
                stage_budget = min(1800, remaining - 30)
            record = {"stage": name, "command": command, "status": "running"}
            state["stages"].append(record)
            write_status(output / "pipeline.json", state)
            print(f"Starting {name}; log: {output / 'logs' / (name + '.log')}", flush=True)
            with (output / "logs" / (name + ".log")).open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record["child_pid"] = child.pid
                write_status(output / "pipeline.json", state)
                code = child.wait(timeout=min(remaining - 45, stage_budget + 30))
            if code:
                record.update(status="failed", exit_code=code)
                raise RuntimeError(f"{name} exited {code}; inspect its log, do not hide a failed stage")
            if name in {"base-dev", "adapter-dev"}:
                evaluation = json.loads((output / name / "evaluation.json").read_text())
                if not evaluation.get("all_selected_evaluated"):
                    record["status"] = "budget_exhausted"
                    raise TimeoutError(f"{name} evaluation incomplete; no full comparison is claimed")
            record.update(status="completed", exit_code=0)
            write_status(output / "pipeline.json", state)
        state["status"] = "completed"
        print(f"Completed. Inspect {output / 'comparison.json'}; this is development evidence, not a production release.")
        return 0
    except BaseException as exc:
        if child:
            stop_owned_child(child)
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error_type=type(exc).__name__, error=str(exc))
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = state["status"]
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        state["elapsed_seconds"] = time.monotonic() - begin
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status(output / "pipeline.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
