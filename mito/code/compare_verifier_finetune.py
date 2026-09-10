"""Paired development comparison; never claims independent scientific validation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

LABELS = ("supported", "contradicted", "insufficient", "mixed")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction(row):
    result = row["prediction"]
    value = result.get("labels", {}).get("evidence_support")
    return value if result.get("label_schema_valid") and value in LABELS else "invalid"


def metrics(truths, predictions):
    if not truths or len(truths) != len(predictions):
        raise ValueError("Nonempty aligned labels are required")
    confusion = {t: {p: 0 for p in (*LABELS, "invalid")} for t in LABELS}
    for t, p in zip(truths, predictions, strict=True):
        if t not in LABELS or p not in (*LABELS, "invalid"):
            raise ValueError("Unexpected support label")
        confusion[t][p] += 1
    scores = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        fp = sum(confusion[t][label] for t in LABELS if t != label)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else None
        scores[label] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                         "precision": precision, "recall": recall,
                         "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    observed = [v["f1"] for v in scores.values() if v["support"]]
    supported_n = sum(t == "supported" for t in truths)
    nonsupported_n = len(truths) - supported_n
    false_support = sum(t != "supported" and p == "supported" for t, p in zip(truths, predictions))
    withheld_support = sum(t == "supported" and p == "insufficient" for t, p in zip(truths, predictions))
    return {"n": len(truths), "accuracy": sum(t == p for t, p in zip(truths, predictions)) / len(truths),
            "macro_f1_observed_classes": sum(observed) / len(observed),
            "macro_f1_fixed_four_classes": sum(v["f1"] for v in scores.values()) / 4,
            "unrepresented_classes": [k for k, v in scores.items() if not v["support"]],
            "per_class": scores, "confusion": confusion,
            "predicted_distribution": dict(Counter(predictions)),
            "false_support_count": false_support,
            "false_support_rate_among_reference_nonsupported": false_support / nonsupported_n if nonsupported_n else None,
            "supported_marked_insufficient_count": withheld_support,
            "supported_marked_insufficient_rate": withheld_support / supported_n if supported_n else None,
            "invalid_predictions": predictions.count("invalid")}


def paired_records(base, tuned):
    if len({r["id"] for r in base}) != len(base) or len({r["id"] for r in tuned}) != len(tuned):
        raise ValueError("Duplicate evaluation IDs")
    index = {r["id"]: r for r in tuned}
    if set(index) != {r["id"] for r in base}:
        raise ValueError("Evaluation coverage differs; do not silently compare intersections")
    pairs = []
    for row in base:
        other = index[row["id"]]
        for field in ("task_id", "prompt_sha256", "target_sha256", "target_labels"):
            if row[field] != other[field]:
                raise ValueError(f"Different evaluation object: {row['id']}:{field}")
        pairs.append((row, other))
    return pairs


def cluster_bootstrap(pairs, *, repeats=2000, seed=20260908):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[0]["task_id"]].append(pair)
    keys = sorted(groups)
    if len(keys) < 2:
        return {"status": "insufficient_independent_tasks", "task_count": len(keys)}
    rng, values = random.Random(seed), []
    for _ in range(repeats):
        chosen = [pair for key in rng.choices(keys, k=len(keys)) for pair in groups[key]]
        truths = [p[0]["target_labels"]["evidence_support"] for p in chosen]
        a = metrics(truths, [prediction(p[0]) for p in chosen])["macro_f1_observed_classes"]
        b = metrics(truths, [prediction(p[1]) for p in chosen])["macro_f1_observed_classes"]
        values.append(b - a)
    values.sort()
    return {"status": "exploratory", "unit": "task_id", "task_count": len(keys), "seed": seed,
            "resamples": repeats, "delta_macro_f1_percentile_interval_95":
            [values[int((repeats - 1) * .025)], values[int((repeats - 1) * .975)]],
            "warning": "Few task clusters; this interval is unstable and is not independent test confirmation."}


def compare(base_path, tuned_path, train_path):
    manifests = [json.loads(Path(p).with_name("run_manifest.json").read_text()) for p in (base_path, tuned_path)]
    for manifest, path in zip(manifests, (base_path, tuned_path), strict=True):
        if manifest.get("status") != "completed" or manifest.get("mode") != "evaluate":
            raise ValueError("Only completed evaluation manifests can be compared")
        if manifest.get("predictions_sha256") != digest(path):
            raise ValueError("Prediction file no longer matches its evaluation manifest")
        if not manifest.get("evaluation", {}).get("all_selected_evaluated"):
            raise ValueError("Evaluation is incomplete")
    base_manifest, tuned_manifest = manifests
    if base_manifest.get("adapter") or not tuned_manifest.get("adapter"):
        raise ValueError("Expected base model versus a declared trained adapter")
    if base_manifest["model"]["snapshot_sha256"] != tuned_manifest["model"]["snapshot_sha256"]:
        raise ValueError("Different base-model weights")
    if base_manifest["script_sha256"] != tuned_manifest["script_sha256"]:
        raise ValueError("Different evaluation implementations")
    for field in ("max_length", "max_new_tokens", "seed", "eval_split", "limit"):
        if base_manifest["config"].get(field) != tuned_manifest["config"].get(field):
            raise ValueError("Different evaluation configuration: " + field)
    split = base_manifest["config"]["eval_split"]
    if split != "dev" or base_manifest["data"][split]["sha256"] != tuned_manifest["data"][split]["sha256"]:
        raise ValueError("This comparator requires identical development data")
    if tuned_manifest["adapter"]["training_data_sha256"] != digest(train_path):
        raise ValueError("Majority-baseline training data differs from adapter training data")
    base, tuned = read_rows(base_path), read_rows(tuned_path)
    pairs = paired_records(base, tuned)
    truths = [p[0]["target_labels"]["evidence_support"] for p in pairs]
    train = read_rows(train_path)
    train_labels = Counter(json.loads(r["completion"][0]["content"])["labels"]["evidence_support"] for r in train)
    if {r["task_id"] for r in train} & {r["task_id"] for r in base}:
        raise ValueError("Training tasks leaked into evaluation")
    majority = train_labels.most_common(1)[0][0]
    reports = {"base": metrics(truths, [prediction(p[0]) for p in pairs]),
               "finetuned": metrics(truths, [prediction(p[1]) for p in pairs]),
               "train_majority_baseline": metrics(truths, [majority] * len(truths))}
    changes = []
    for a, b in pairs:
        if prediction(a) != prediction(b):
            changes.append({"id": a["id"], "task_id": a["task_id"],
                            "target": a["target_labels"]["evidence_support"],
                            "base": prediction(a), "finetuned": prediction(b)})
    return {"schema": "mito.verifier-paired-dev.v1", "production_ready": False,
            "scientific_improvement_proven": False,
            "scope": "Given annotated evidence summaries and synthetic observations; development label agreement only.",
            "evaluated_records": len(pairs), "independent_tasks": len({p[0]["task_id"] for p in pairs}),
            "full_development_coverage": len(pairs) == base_manifest["evaluation"]["dataset_records"],
            "explicit_evaluation_limit": base_manifest["config"].get("limit"),
            "source_files": {str(p): digest(p) for p in (base_path, tuned_path, train_path)},
            "training_label_distribution": dict(train_labels), "majority_label_chosen_from_train": majority,
            "metrics": reports,
            "delta_macro_f1_observed_classes": reports["finetuned"]["macro_f1_observed_classes"] - reports["base"]["macro_f1_observed_classes"],
            "paired_task_bootstrap": cluster_bootstrap(pairs), "changed_predictions": changes,
            "warnings": ["Dev is not an untouched independent final test.",
                         "No new source-truth, mechanism reasoning or tool-policy ability is certified.",
                         "Human labels are single-reviewer attested; model-assisted drafting provenance is retained.",
                         "Missing classes cannot be evaluated; do not treat them as zero-error performance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base, args.finetuned, args.train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("evaluated_records", "independent_tasks", "delta_macro_f1_observed_classes", "paired_task_bootstrap")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
