"""Isolated, local-only evidence-verifier SFT and JSON-label evaluation.

Run as ``python code/train_verifier.py`` in a migration bundle, or from
``agent/`` with ``python -m training.qwen_llm.train_verifier``.
``--mode validate`` loads only the local tokenizer, never model weights/CUDA.
Train/evaluate require an explicit idle ``--gpu`` and a new output directory.
Inputs: {id, task_id, prompt:[system,user], completion:[assistant], metadata:{split}}.
The assistant's content is JSON containing ``labels.evidence_support``.
Input provenance remains in the audit record and never enters supervised tokens.
This experiment does not publish weights or certify scientific correctness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/Qwen3-4B-Instruct-2507"
SCHEMA = "mito.evidence-verifier-training.v1"
SUPPORT_LABELS = ("supported", "contradicted", "insufficient", "mixed")


class VerifierError(ValueError):
    """Refused configuration/data; no implicit fallback or data repair."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise VerifierError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise VerifierError(f"Non-finite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise VerifierError(f"Non-finite JSON number: {value}")
        return number

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)
    except (ValueError, TypeError) as exc:
        raise VerifierError(f"Invalid strict JSON: {exc}") from exc


def target_labels(content: str) -> dict:
    value = strict_json(content)
    labels = value.get("labels") if isinstance(value, dict) else None
    if not isinstance(labels, dict) or labels.get("evidence_support") not in SUPPORT_LABELS:
        raise VerifierError("labels.evidence_support must be supported/contradicted/insufficient/mixed")
    # List-valued causal evidence/error types are valid training targets, but
    # are not silently treated as scalar classes by the evaluation metrics.
    for name, item in labels.items():
        if not isinstance(name, str) or not name:
            raise VerifierError("Label names must be nonempty strings")
        if item is not None and not isinstance(item, (str, bool, int, float, list, dict)):
            raise VerifierError(f"Unsupported label type: {name}")
    return labels


def read_records(path: Path, *, split: str) -> list[dict]:
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = strict_json(line)
            if not isinstance(row, dict):
                raise VerifierError(f"{path}:{line_no}: record must be an object")
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "task_id")):
                raise VerifierError(f"{path}:{line_no}: id and task_id are required")
            if row["id"] in seen:
                raise VerifierError(f"Duplicate record id: {row['id']}")
            seen.add(row["id"])
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("split") != split:
                raise VerifierError(f"{row['id']}: metadata.split must equal {split!r}")
            usage = str(metadata.get("use", "")).lower()
            dataset_mode = str(metadata.get("dataset_mode", ""))
            if "pending" in usage or "not_for_training" in usage or dataset_mode == "raw_source_pending_label_transfer":
                raise VerifierError(f"{row['id']}: pending/untransferred labels are not valid train or evaluation targets")
            expected_usage = "exploratory_training" if split == "train" else "evaluation_only"
            if usage and usage != expected_usage:
                raise VerifierError(f"{row['id']}: metadata.use must be {expected_usage!r} for split {split!r}")
            if dataset_mode == "expert_evidence_summary_conditioned" and metadata.get("labels_from_exact_reviewed_original") is not True:
                raise VerifierError(f"{row['id']}: summary labels must bind to the exact reviewed original")
            prompt, completion = row.get("prompt"), row.get("completion")
            if not isinstance(prompt, list) or [m.get("role") if isinstance(m, dict) else None for m in prompt] != ["system", "user"]:
                raise VerifierError(f"{row['id']}: prompt must contain exactly system then user")
            if not isinstance(completion, list) or len(completion) != 1 or not isinstance(completion[0], dict) or completion[0].get("role") != "assistant":
                raise VerifierError(f"{row['id']}: completion must contain exactly one assistant")
            for message in prompt + completion:
                if set(message) != {"role", "content"} or not isinstance(message["content"], str) or not message["content"].strip():
                    raise VerifierError(f"{row['id']}: messages require nonempty content and role only")
            target_labels(completion[0]["content"])
            rows.append(row)
    if not rows:
        raise VerifierError(f"Empty dataset: {path}")
    return rows


def check_split_overlap(train: list[dict], dev: list[dict]) -> dict:
    overlaps = {}
    for key, collect in (("id", lambda r: r["id"]), ("task_id", lambda r: r["task_id"]),
                         ("prompt_sha256", lambda r: canonical_hash(r["prompt"]))):
        common = {collect(r) for r in train} & {collect(r) for r in dev}
        overlaps[key] = sorted(common)
    if any(overlaps.values()):
        raise VerifierError("Train/eval overlap: " + json.dumps(overlaps, ensure_ascii=False))
    return {"id_task_prompt_overlap": overlaps,
            "source_group_isolation": "must be supplied by the release/export audit; not inferred from text"}


def encode_record(row: dict, tokenizer, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise VerifierError(f"{row['id']}: chat template is not prefix-stable")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    all_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if all_ids[:len(prompt_ids)] != prompt_ids:
        raise VerifierError(f"{row['id']}: token boundary is not prefix-stable")
    if len(all_ids) > max_length:
        raise VerifierError(f"{row['id']}: {len(all_ids)} tokens exceeds {max_length}; refusing truncation")
    if len(all_ids) <= len(prompt_ids):
        raise VerifierError(f"{row['id']}: no supervised completion tokens")
    if tokenizer.eos_token_id not in all_ids[len(prompt_ids):]:
        raise VerifierError(f"{row['id']}: assistant end-of-turn token is missing")
    labels = [-100] * len(prompt_ids) + all_ids[len(prompt_ids):]
    return {"features": {"input_ids": all_ids, "labels": labels}, "prompt_ids": prompt_ids,
            "prompt_tokens": len(prompt_ids), "completion_tokens": len(all_ids) - len(prompt_ids),
            "total_tokens": len(all_ids), "prompt_sha256": canonical_hash(row["prompt"]),
            "completion_sha256": canonical_hash(row["completion"])}


def length_report(encoded: list[dict]) -> dict:
    values = sorted(row["total_tokens"] for row in encoded)
    return {"records": len(values), "total_tokens": sum(values),
            "supervised_tokens": sum(row["completion_tokens"] for row in encoded),
            "prompt_max": max(row["prompt_tokens"] for row in encoded),
            "completion_max": max(row["completion_tokens"] for row in encoded),
            "min": values[0], "median": values[len(values) // 2],
            "p95": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
            "max": values[-1], "truncated_records": 0}


def model_snapshot(model: Path) -> dict:
    model = model.resolve(strict=True)
    config = strict_json((model / "config.json").read_text())
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != ("qwen3", 2560, 36):
        raise VerifierError("Local model is not the locked Qwen3-4B architecture")
    manifest = strict_json((model / "download_manifest.json").read_text())
    if manifest.get("repo") != MODEL_ID or manifest.get("verified") is not True:
        raise VerifierError("Missing verified local model identity")
    index = strict_json((model / "model.safetensors.index.json").read_text())
    entries = {row["name"]: row for row in manifest["files"]}
    names = set(index["weight_map"].values()) | {"config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"}
    names |= {name for name in ("generation_config.json", "special_tokens_map.json", "added_tokens.json",
                               "chat_template.jinja", "vocab.json", "merges.txt") if (model / name).exists()}
    hashes = {}
    for name in sorted(names):
        path = (model / name).resolve(strict=True)
        if not path.is_relative_to(model) or name not in entries:
            raise VerifierError(f"Unbound or unsafe model file: {name}")
        digest = sha256_file(path)
        if digest != entries[name]["sha256"]:
            raise VerifierError(f"Model file hash mismatch: {name}")
        hashes[name] = digest
    return {"path": str(model), "model_id": MODEL_ID, "files_sha256": hashes,
            "snapshot_sha256": canonical_hash(hashes), "remote_revision": manifest.get("resolved_revision"),
            "max_position_embeddings": config.get("max_position_embeddings")}


def installed_versions() -> dict:
    result = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def output_path(path: Path | None) -> Path:
    if path is None:
        raise VerifierError("Train/evaluate require --output-dir")
    result = path.expanduser().resolve()
    protected = {Path(result.anchor), Path.home().resolve(), Path.cwd().resolve(), Path(__file__).resolve().parent}
    if result in protected:
        raise VerifierError("Experiment output cannot be a filesystem/home/workspace/code root")
    if result.exists():
        raise VerifierError("Output exists; choose a new run name, never overwrite prior results")
    return result


def inspect_gpu(gpu: int) -> dict:
    if gpu < 0:
        raise VerifierError("GPU index must be nonnegative")
    result = subprocess.run(["nvidia-smi", "-i", str(gpu),
                             "--query-gpu=uuid,name,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    values = [v.strip() for v in result.stdout.strip().split(",")]
    if len(values) != 4:
        raise VerifierError("Could not resolve a single GPU")
    gpu_uuid, name, memory, util = values
    if int(memory) > 1024 or int(util) > 5:
        raise VerifierError(f"GPU {gpu} is busy ({memory} MiB, {util}%); no processes will be stopped")
    return {"physical_index": gpu, "uuid": gpu_uuid, "name": name,
            "initial_memory_mib": int(memory), "initial_utilization_percent": int(util)}


@contextmanager
def gpu_lease(gpu: int):
    import fcntl
    # Advisory protection for this launcher only; independent workloads can
    # still race the idle check. Never terminate them or silently move GPUs.
    with Path(f"/tmp/mito-verifier-gpu-{gpu}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise VerifierError(f"Another verifier run holds GPU {gpu}") from exc
        info = inspect_gpu(gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = info["uuid"]
        try:
            yield info
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def parse_prediction(text: str) -> dict:
    try:
        parsed = strict_json(text)
    except VerifierError as exc:
        return {"json_valid": False, "label_schema_valid": False, "labels": {}, "error": str(exc)}
    try:
        labels = target_labels(text)
    except VerifierError as exc:
        return {"json_valid": True, "label_schema_valid": False, "labels": {}, "parsed": parsed, "error": str(exc)}
    return {"json_valid": True, "label_schema_valid": True, "labels": labels, "parsed": parsed, "error": None}


def class_key(value: Any) -> str | None:
    if value is None or isinstance(value, (list, dict)):
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def score_predictions(rows: list[dict]) -> dict:
    """Invalid/missing predictions count as false negatives, never get dropped."""
    count = len(rows)
    fields = sorted({key for row in rows for key in row["target_labels"]})
    reports = {}
    for field in fields:
        pairs, not_applicable, nonscalar = [], 0, 0
        for row in rows:
            value = row["target_labels"].get(field)
            if value is None:
                not_applicable += 1
                continue
            if isinstance(value, (list, dict)):
                nonscalar += 1
                continue
            pred = row["prediction"]["labels"].get(field) if row["prediction"]["label_schema_valid"] else None
            pairs.append((class_key(value), class_key(pred)))
        truths = sorted({truth for truth, _ in pairs})
        defined = {class_key(label) for label in SUPPORT_LABELS} if field == "evidence_support" else set()
        labels = sorted(set(truths) | {pred for _, pred in pairs if pred is not None} | defined)
        per_class = {}
        for label in labels:
            tp = sum(t == label and p == label for t, p in pairs)
            fp = sum(t != label and p == label for t, p in pairs)
            fn = sum(t == label and p != label for t, p in pairs)
            precision, recall = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
            per_class[label] = {"label": strict_json(label), "support": sum(t == label for t, _ in pairs),
                                "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
        reports[field] = {"evaluated": len(pairs), "not_applicable_or_absent": not_applicable,
                          "non_scalar_targets_not_scored": nonscalar, "per_class": per_class,
                          "invalid_or_missing_predictions": sum(p is None for _, p in pairs),
                          "macro_f1": sum(per_class[label]["f1"] for label in truths) / len(truths) if truths else None,
                          "macro_f1_denominator": "classes present in reference targets",
                          "reference_absent_classes": [strict_json(label) for label in sorted(defined - set(truths))],
                          "macro_f1_all_four_classes": sum(per_class[label]["f1"] for label in defined) / len(defined) if defined and pairs else None,
                          "accuracy": sum(t == p for t, p in pairs) / len(pairs) if pairs else None}
    return {"evaluated_records": count,
            "json_valid_rate": sum(r["prediction"]["json_valid"] for r in rows) / count if count else None,
            "label_schema_valid_rate": sum(r["prediction"]["label_schema_valid"] for r in rows) / count if count else None,
            "generation_truncated_count": sum(bool(r.get("generation_truncated")) for r in rows),
            "fields": reports, "primary_metric": "fields.evidence_support.macro_f1",
            "scope": "label agreement, not proof of scientific correctness or source truth"}


def budget_expired(started: float, seconds: int, margin: int) -> bool:
    return time.monotonic() - started >= seconds - margin


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "train", "evaluate"), default="validate")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true", help="Explicitly open frozen test; never use for tuning")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--adapter", type=Path, help="Evaluate an adapter produced by this entry point; absent = base model")
    parser.add_argument("--output-dir", type=Path, help="New experiment directory; any filesystem, never an existing/root directory")
    parser.add_argument("--gpu", type=int, help="Required physical GPU index for train/evaluate; busy GPU refused")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--safety-margin-seconds", type=int, default=120)
    parser.add_argument("--limit", type=int, help="Explicit evaluation subset only; report retains total planned count")
    return parser


def validate_args(args) -> None:
    for name in ("max_length", "max_new_tokens", "batch_size", "grad_accum", "learning_rate", "lora_r", "save_steps", "logging_steps", "max_seconds", "safety_margin_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise VerifierError(f"{name} must be finite and positive")
    if not math.isfinite(args.epochs) or not 1 <= args.epochs <= 3:
        raise VerifierError("--epochs must be between 1 and 3")
    if args.max_steps is not None and args.max_steps <= 0:
        raise VerifierError("--max-steps must be positive")
    if args.limit is not None and args.limit <= 0:
        raise VerifierError("--limit must be positive")
    if args.max_seconds > 8 * 3600 or args.safety_margin_seconds >= args.max_seconds:
        raise VerifierError("Budget must be <=8h and larger than safety margin")
    if args.eval_split == "test" and (not args.allow_test or args.mode == "train"):
        raise VerifierError("Test requires --allow-test and is never a training validation split")
    if args.mode in ("validate", "train") and args.train_data is None:
        raise VerifierError("--train-data is required for validate/train")
    if args.mode == "evaluate" and args.eval_data is None:
        raise VerifierError("--eval-data is required for evaluate")
    if args.mode != "validate" and (args.gpu is None or args.gpu < 0):
        raise VerifierError("Train/evaluate require an explicit nonnegative --gpu")
    if args.mode != "evaluate" and args.adapter is not None:
        raise VerifierError("--adapter is only for evaluate; training starts from the locked base")
    if args.mode != "evaluate" and args.limit is not None:
        raise VerifierError("--limit is evaluation-only; no silent training subset")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise VerifierError("Use one isolated GPU; distributed launch is not configured")


def prepare(args) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], Any]:
    validate_args(args)
    train = read_records(args.train_data, split="train") if args.train_data else []
    evaluation = read_records(args.eval_data, split=args.eval_split) if args.eval_data else []
    split_audit = check_split_overlap(train, evaluation)
    snapshot = model_snapshot(args.model_path)
    if args.max_length > snapshot["max_position_embeddings"]:
        raise VerifierError("Requested context exceeds local base-model context")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    encoded_train = [encode_record(row, tokenizer, args.max_length) for row in train]
    encoded_eval = [encode_record(row, tokenizer, args.max_length) for row in evaluation]
    if args.mode == "evaluate":
        for row, encoded in zip(evaluation, encoded_eval, strict=True):
            if encoded["prompt_tokens"] + args.max_new_tokens > args.max_length:
                raise VerifierError(f"{row['id']}: prompt plus generation budget exceeds --max-length; do not truncate evidence")
    report = {"schema": SCHEMA, "status": "validated", "mode": args.mode, "created_at": now(),
              "model": snapshot, "versions": installed_versions(), "split_audit": split_audit,
              "data": {key: {"path": str(path.resolve()), "sha256": sha256_file(path),
                             **length_report(encoded)} for key, path, encoded in
                       (("train", args.train_data, encoded_train), (args.eval_split, args.eval_data, encoded_eval)) if path},
              "loss_scope": "assistant_completion_only_explicit_labels", "production_ready": False,
              "scientific_improvement_proven": False, "external_model_calls": 0,
              "script_sha256": sha256_file(Path(__file__)),
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    if encoded_eval:
        report["evaluation_generation_budget"] = {
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_plus_new_tokens": max(row["prompt_tokens"] for row in encoded_eval) + args.max_new_tokens,
            "context_limit": args.max_length,
            "all_fit": all(row["prompt_tokens"] + args.max_new_tokens <= args.max_length for row in encoded_eval),
            "overlength_ids": [row["id"] for row, item in zip(evaluation, encoded_eval, strict=True)
                               if item["prompt_tokens"] + args.max_new_tokens > args.max_length],
            "reference_completion_max": max(row["completion_tokens"] for row in encoded_eval),
        }
    return report, train, evaluation, encoded_train, encoded_eval, tokenizer


def train(args, report, train_rows, eval_rows, tokenizer, output: Path, started: float) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    class AuditCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            if budget_expired(started, args.max_seconds, args.safety_margin_seconds):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(self, args_, state, control, logs=None, **kwargs):
            clean = {key: value if not isinstance(value, float) or math.isfinite(value) else None for key, value in (logs or {}).items()}
            record = {"at": now(), "step": state.global_step, "elapsed_seconds": time.monotonic() - started, **clean}
            append_json(output / "metrics.jsonl", record)
            write_json(output / "progress.json", record)

    config = SFTConfig(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        warmup_ratio=0.05, lr_scheduler_type="cosine", seed=args.seed, data_seed=args.seed,
        completion_only_loss=True, assistant_only_loss=False, packing=False,
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=args.max_length,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, tf32=True, logging_steps=args.logging_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=3,
        eval_strategy="steps" if eval_rows else "no", eval_steps=args.save_steps,
        report_to=[], push_to_hub=False, disable_tqdm=True, eos_token=tokenizer.eos_token,
        model_init_kwargs={"dtype": torch.bfloat16, "local_files_only": True,
                           "trust_remote_code": False, "attn_implementation": "sdpa", "use_cache": False},
    )
    trainer = SFTTrainer(
        model=str(args.model_path), args=config,
        train_dataset=Dataset.from_list([row["features"] for row in train_rows]),
        eval_dataset=Dataset.from_list([row["features"] for row in eval_rows]) if eval_rows else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM", bias="none"),
        callbacks=[AuditCallback()],
    )
    if budget_expired(started, args.max_seconds, args.safety_margin_seconds):
        raise VerifierError("Preparation/model loading exhausted the training budget before optimization")
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    report.update(status="completed", steps=trainer.state.global_step, metrics=result.metrics,
                  ended_by="wall_clock" if budget_expired(started, args.max_seconds, args.safety_margin_seconds) else "configured_steps_or_epochs",
                  adapter_sha256=sha256_file(output / "adapter" / "adapter_model.safetensors"),
                  adapter_config_sha256=sha256_file(output / "adapter" / "adapter_config.json"))
    return report


def verify_adapter(path: Path, snapshot: dict) -> dict:
    path = path.resolve(strict=True)
    source = path.parent / "run_manifest.json"
    manifest = strict_json(source.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed" or manifest.get("mode") != "train":
        raise VerifierError("Adapter is not a completed verifier training artifact")
    if manifest.get("model", {}).get("snapshot_sha256") != snapshot["snapshot_sha256"]:
        raise VerifierError("Adapter base-model byte identity differs")
    for name, key in (("adapter_model.safetensors", "adapter_sha256"), ("adapter_config.json", "adapter_config_sha256")):
        if sha256_file(path / name) != manifest.get(key):
            raise VerifierError(f"Adapter hash mismatch: {name}")
    return {"path": str(path), "run_manifest_sha256": sha256_file(source),
            "adapter_sha256": manifest["adapter_sha256"], "training_data_sha256": manifest.get("data", {}).get("train", {}).get("sha256")}


def evaluate(args, report, evaluation, encoded, tokenizer, output: Path, started: float) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, set_seed
    set_seed(args.seed)
    if args.adapter:
        report["adapter"] = verify_adapter(args.adapter, report["model"])
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True,
                                                trust_remote_code=False, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda:0")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False, local_files_only=True)
    model.eval()

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return budget_expired(started, args.max_seconds, args.safety_margin_seconds)

    selected = list(zip(evaluation, encoded, strict=True))
    if args.limit is not None:
        selected = selected[:args.limit]
    outputs = []
    stop_ids = {tokenizer.eos_token_id}
    for row, item in selected:
        if budget_expired(started, args.max_seconds, args.safety_margin_seconds):
            break
        begin = time.monotonic()
        ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=args.max_new_tokens, do_sample=False,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                       stopping_criteria=StoppingCriteriaList([Deadline()]), use_cache=True)
        generated_ids = generated[0, len(item["prompt_ids"]):].tolist()
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prediction = parse_prediction(raw)
        record = {"id": row["id"], "task_id": row["task_id"], "metadata": row["metadata"],
                  "prompt_sha256": item["prompt_sha256"], "target_sha256": item["completion_sha256"],
                  "target_labels": target_labels(row["completion"][0]["content"]),
                  "prediction": prediction, "raw_generation": raw, "generated_token_ids": generated_ids,
                  "generation_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                  "generated_tokens": len(generated_ids), "elapsed_seconds": time.monotonic() - begin,
                  "generation_truncated": not generated_ids or generated_ids[-1] not in stop_ids,
                  "budget_stop": budget_expired(started, args.max_seconds, args.safety_margin_seconds)}
        outputs.append(record)
        append_json(output / "predictions.jsonl", record)
        write_json(output / "progress.json", {"at": now(), "evaluated": len(outputs), "selected": len(selected),
                                              "elapsed_seconds": time.monotonic() - started})
    metrics = score_predictions(outputs)
    strata = sorted({str(r["metadata"].get("annotation_version", "unspecified")) for r in outputs})
    metrics["by_annotation_version"] = {stratum: score_predictions([r for r in outputs if str(r["metadata"].get("annotation_version", "unspecified")) == stratum]) for stratum in strata}
    metrics.update(dataset_records=len(evaluation), selected_records=len(selected),
                   all_selected_evaluated=len(outputs) == len(selected), split=args.eval_split,
                   evaluation_role="development_only_not_unseen_test" if args.eval_split == "dev" else "explicitly_opened_frozen_test")
    write_json(output / "evaluation.json", metrics)
    report.update(status="completed" if len(outputs) == len(selected) else "budget_exhausted",
                  evaluation=metrics, predictions_sha256=sha256_file(output / "predictions.jsonl") if outputs else None)
    return report


def run(args) -> dict:
    started = time.monotonic()
    validate_args(args)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.mode == "validate":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        report, *_ = prepare(args)
        return report
    output = output_path(args.output_dir)
    # Acquire/resolve the requested physical GPU before importing torch or the
    # tokenizer: changing CUDA_VISIBLE_DEVICES after CUDA initialization is unsafe.
    with gpu_lease(args.gpu) as gpu:
        report, _, evaluation, encoded_train, encoded_eval, tokenizer = prepare(args)
        import torch
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise VerifierError("Exactly one visible CUDA GPU with BF16 is required")
        output.mkdir(parents=True, exist_ok=False)
        report.update(status="running", started_at=now(), pid=os.getpid(), gpu=gpu,
                      budget_scope="whole command including validation; checked at optimization/token boundaries, not a hard kill")
        write_json(output / "run_manifest.json", report)
        try:
            if args.mode == "train":
                report = train(args, report, encoded_train, encoded_eval, tokenizer, output, started)
            else:
                report = evaluate(args, report, evaluation, encoded_eval, tokenizer, output, started)
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error_type=type(exc).__name__, error=str(exc), finished_at=now(),
                          elapsed_seconds=time.monotonic() - started)
            write_json(output / "run_manifest.json", report)
            raise
        report.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "run_manifest.json", report)
        return report


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (VerifierError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__, "reason": str(exc),
                          "production_ready": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
