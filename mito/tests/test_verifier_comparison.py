"""CPU-only scientific metric/paired-artifact contracts for migration bundles."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


_code = Path(__file__).parent
if not (_code / "compare_verifier_finetune.py").is_file():
    _code = Path(__file__).parents[1] / "code"
spec = importlib.util.spec_from_file_location("isolated_verifier_comparison", _code / "compare_verifier_finetune.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def row(identifier="c1", task="dev1", target="supported", predicted="supported"):
    return {"id": identifier, "task_id": task, "prompt_sha256": "prompt-" + identifier,
            "target_sha256": "target-" + identifier, "target_labels": {"evidence_support": target},
            "prediction": {"label_schema_valid": predicted != "invalid", "labels": {"evidence_support": predicted}}}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def bundle(tmp_path):
    base = write_rows(tmp_path / "base/predictions.jsonl", [row(), row("c2", "dev2", "contradicted", "supported")])
    tuned = write_rows(tmp_path / "tuned/predictions.jsonl", [row(), row("c2", "dev2", "contradicted", "contradicted")])
    train = write_rows(tmp_path / "train.jsonl", [{"id": "tr1", "task_id": "train1",
                        "completion": [{"role": "assistant", "content": '{"labels":{"evidence_support":"supported"}}'}]}])
    for path in (base, tuned):
        manifest = {"status": "completed", "mode": "evaluate", "predictions_sha256": compare.digest(path),
                    "evaluation": {"all_selected_evaluated": True, "dataset_records": 2},
                    "model": {"snapshot_sha256": "base-model-hash"}, "script_sha256": "same-script",
                    "config": {"max_length": 8192, "max_new_tokens": 1536, "seed": 7, "eval_split": "dev", "limit": None},
                    "data": {"dev": {"sha256": "same-dev-hash"}}}
        if path == tuned:
            manifest["adapter"] = {"training_data_sha256": compare.digest(train)}
        path.with_name("run_manifest.json").write_text(json.dumps(manifest))
    return base, tuned, train


def test_confusion_f1_invalid_and_high_risk_directions_are_correct():
    result = compare.metrics(["supported", "supported", "contradicted", "insufficient"],
                             ["supported", "insufficient", "supported", "invalid"])
    assert result["accuracy"] == 0.25
    assert result["per_class"]["supported"]["precision"] == 0.5
    assert result["per_class"]["supported"]["recall"] == 0.5
    assert result["per_class"]["supported"]["f1"] == 0.5
    assert result["false_support_rate_among_reference_nonsupported"] == 0.5
    assert result["supported_marked_insufficient_rate"] == 0.5
    assert result["invalid_predictions"] == 1
    assert result["unrepresented_classes"] == ["mixed"]
    assert result["per_class"]["mixed"]["recall"] is None


def test_paired_report_uses_training_majority_and_frozen_prediction_objects(tmp_path):
    base, tuned, train = bundle(tmp_path)
    report = compare.compare(base, tuned, train)
    assert report["majority_label_chosen_from_train"] == "supported"
    assert report["evaluated_records"] == 2 and report["full_development_coverage"]
    assert report["metrics"]["finetuned"]["macro_f1_observed_classes"] == 1
    assert report["delta_macro_f1_observed_classes"] > 0
    assert len(report["changed_predictions"]) == 1
    assert report["paired_task_bootstrap"]["unit"] == "task_id"
    assert not report["production_ready"] and not report["scientific_improvement_proven"]


@pytest.mark.parametrize("mutation,reason", [
    (lambda m: m.update(status="budget_exhausted"), "completed"),
    (lambda m: m["evaluation"].update(all_selected_evaluated=False), "incomplete"),
    (lambda m: m["model"].update(snapshot_sha256="other"), "base-model"),
    (lambda m: m.update(script_sha256="other"), "implementations"),
    (lambda m: m["config"].update(max_new_tokens=512), "configuration"),
    (lambda m: m["config"].update(max_length=4096), "configuration"),
    (lambda m: m["config"].update(seed=123), "configuration"),
    (lambda m: m["config"].update(limit=1), "configuration"),
    (lambda m: m["data"]["dev"].update(sha256="other"), "identical development"),
    (lambda m: m["adapter"].update(training_data_sha256="other"), "training data differs"),
])
def test_manifest_mismatch_refuses_unfair_comparison(tmp_path, mutation, reason):
    base, tuned, train = bundle(tmp_path)
    path = tuned.with_name("run_manifest.json")
    manifest = json.loads(path.read_text())
    mutation(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        compare.compare(base, tuned, train)


def test_modified_prediction_bytes_are_rejected(tmp_path):
    base, tuned, train = bundle(tmp_path)
    tuned.write_text(tuned.read_text() + "\n")
    with pytest.raises(ValueError, match="no longer matches"):
        compare.compare(base, tuned, train)


@pytest.mark.parametrize("field", ["task_id", "prompt_sha256", "target_sha256", "target_labels"])
def test_pairing_never_matches_merely_on_id(field):
    first, second = row(), row()
    second[field] = {} if field == "target_labels" else "different"
    with pytest.raises(ValueError, match="Different evaluation object"):
        compare.paired_records([first], [second])


def test_pairing_rejects_duplicate_ids_and_different_coverage():
    with pytest.raises(ValueError, match="Duplicate"):
        compare.paired_records([row(), row()], [row()])
    with pytest.raises(ValueError, match="coverage"):
        compare.paired_records([row()], [row("other")])


def test_single_cluster_does_not_receive_spurious_bootstrap_interval():
    pairs = [(row(), row()), (row("c2"), row("c2"))]
    result = compare.cluster_bootstrap(pairs)
    assert result["status"] == "insufficient_independent_tasks"
    assert "delta_macro_f1_percentile_interval_95" not in result


def test_bootstrap_is_reproducible_and_resamples_whole_tasks():
    pairs = [(row(), row()), (row("c2", "dev2", "contradicted", "supported"), row("c2", "dev2", "contradicted", "contradicted"))]
    result = compare.cluster_bootstrap(pairs, repeats=50, seed=8)
    assert result == compare.cluster_bootstrap(deepcopy(pairs), repeats=50, seed=8)
    assert result["task_count"] == 2 and result["resamples"] == 50


def test_partial_explicit_dev_subset_is_not_full_coverage(tmp_path):
    base, tuned, train = bundle(tmp_path)
    for path in (base, tuned):
        manifest_path = path.with_name("run_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["limit"] = 2
        manifest["evaluation"]["dataset_records"] = 51
        manifest_path.write_text(json.dumps(manifest))
    report = compare.compare(base, tuned, train)
    assert not report["full_development_coverage"] and report["explicit_evaluation_limit"] == 2
