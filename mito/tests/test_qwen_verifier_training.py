"""Verifier data/masking/metric guards; no model weights or GPU execution."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_script = Path(os.environ.get("VERIFIER_SCRIPT", str(Path(__file__).parents[1] / "training/qwen_llm/train_verifier.py")))
if not _script.is_file():
    _script = Path(__file__).parents[1] / "code/train_verifier.py"
_spec = importlib.util.spec_from_file_location("isolated_test_verifier", _script)
verifier = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(verifier)
if os.environ.get("VERIFIER_MODEL_PATH"):
    verifier.MODEL_PATH = Path(os.environ["VERIFIER_MODEL_PATH"])


def record(identifier="c1", task="t1", split="train", support="supported"):
    return {"id": identifier, "task_id": task,
            "prompt": [{"role": "system", "content": "依据给定原文核验，只输出JSON。"},
                       {"role": "user", "content": "证据：TMRM是膜电位代理。主张：TMRM直接测量ATP。"}],
            "completion": [{"role": "assistant", "content": json.dumps({"labels": {
                "evidence_support": support, "condition_match": False, "uncertain": None,
                "error_types": ["proxy_as_direct_measurement"]}, "reason": "应保留测量边界。"}, ensure_ascii=False)}],
            "metadata": {"split": split, "annotation_version": "expert-snapshot", "private_origin": "HIDDEN_REVIEWER"}}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


class Tokenizer:
    eos_token_id = 3
    pad_token_id = 3
    eos_token = "\x03"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        return "".join(m["role"] + ":" + m["content"] + "\x03\n" for m in messages) + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def args(*extra):
    return verifier.build_parser().parse_args(["--train-data", "/does-not-load.jsonl", *extra])


def test_valid_json_conversation_keeps_lists_metadata_and_raw_bytes(tmp_path):
    original = record()
    file = write_rows(tmp_path / "train.jsonl", [original])
    before = file.read_bytes()
    assert verifier.read_records(file, split="train") == [original]
    assert file.read_bytes() == before
    assert verifier.target_labels(original["completion"][0]["content"])["error_types"] == ["proxy_as_direct_measurement"]


@pytest.mark.parametrize("content", [
    '{"labels":{"evidence_support":"supported","evidence_support":"refuted"}}',
    '{"labels":{"evidence_support":"supported","score":NaN}}',
    '{"labels":{"evidence_support":"supported","score":1e999}}',
    '{"labels":{"evidence_support":"unknown"}}',
    '```json\n{"labels":{"evidence_support":"supported"}}\n```',
    '{"labels": {"evidence_support": null}}',
    '{"labels": {"evidence_support": ""}}',
    '{"evidence_support":"supported"}',
    '[]',
])
def test_strict_targets_refuse_duplicate_nonfinite_fenced_or_wrong_schema(tmp_path, content):
    row = record()
    row["completion"][0]["content"] = content
    with pytest.raises(verifier.VerifierError):
        verifier.read_records(write_rows(tmp_path / "train.jsonl", [row]), split="train")


@pytest.mark.parametrize("mutation", [
    lambda r: r["metadata"].update(split="dev"),
    lambda r: r.update(id=""),
    lambda r: r.update(task_id=None),
    lambda r: r["prompt"].reverse(),
    lambda r: r["prompt"].append({"role": "assistant", "content": "leaked answer"}),
    lambda r: r["completion"].append({"role": "assistant", "content": "extra"}),
    lambda r: r["prompt"][1].update(content=""),
    lambda r: r["prompt"][1].update(tools=[]),
])
def test_schema_split_and_message_scope_fail_closed(tmp_path, mutation):
    row = record()
    mutation(row)
    with pytest.raises(verifier.VerifierError):
        verifier.read_records(write_rows(tmp_path / "train.jsonl", [row]), split="train")


def test_duplicate_ids_and_empty_data_refused(tmp_path):
    with pytest.raises(verifier.VerifierError, match="Duplicate"):
        verifier.read_records(write_rows(tmp_path / "dupe.jsonl", [record(), record()]), split="train")
    with pytest.raises(verifier.VerifierError, match="Empty"):
        verifier.read_records(write_rows(tmp_path / "empty.jsonl", []), split="train")


@pytest.mark.parametrize("split,use,mode", [
    ("train", "pending_review", ""), ("dev", "pending_review", ""),
    ("train", "not_for_training", ""), ("dev", "not_for_training", ""),
    ("train", "exploratory_training", "raw_source_pending_label_transfer"),
    ("dev", "evaluation_only", "raw_source_pending_label_transfer"),
    ("train", "evaluation_only", ""), ("dev", "exploratory_training", ""),
])
def test_pending_raw_source_and_wrong_usage_are_refused_for_both_splits(tmp_path, split, use, mode):
    row = record(split=split)
    row["metadata"].update(use=use, dataset_mode=mode)
    with pytest.raises(verifier.VerifierError):
        verifier.read_records(write_rows(tmp_path / "rows.jsonl", [row]), split=split)


def test_expert_summary_exact_label_binding_is_required_when_declared(tmp_path):
    row = record()
    row["metadata"].update(dataset_mode="expert_evidence_summary_conditioned", use="exploratory_training")
    path = write_rows(tmp_path / "train.jsonl", [row])
    with pytest.raises(verifier.VerifierError, match="exact reviewed original"):
        verifier.read_records(path, split="train")
    row["metadata"]["labels_from_exact_reviewed_original"] = True
    assert verifier.read_records(write_rows(path, [row]), split="train") == [row]


def test_split_overlap_checks_ids_tasks_and_prompt_not_metadata_only():
    train, dev = record(), record("d1", "dt1", "dev")
    with pytest.raises(verifier.VerifierError, match="prompt_sha256"):
        verifier.check_split_overlap([train], [dev])
    dev["prompt"][1]["content"] = "另一组原文。"
    assert not any(verifier.check_split_overlap([train], [dev])["id_task_prompt_overlap"].values())
    dev["task_id"] = train["task_id"]
    with pytest.raises(verifier.VerifierError, match="task_id"):
        verifier.check_split_overlap([train], [dev])


def test_explicit_mask_supervises_only_completion_including_eos_no_metadata_leak():
    row = record()
    original = deepcopy(row)
    item = verifier.encode_record(row, Tokenizer(), 8192)
    ids, labels = item["features"]["input_ids"], item["features"]["labels"]
    boundary = item["prompt_tokens"]
    assert labels[:boundary] == [-100] * boundary
    assert labels[boundary:] == ids[boundary:]
    assert 3 in labels[boundary:]
    assert "HIDDEN_REVIEWER" not in "".join(chr(value) for value in ids)
    assert row == original
    assert item["prompt_sha256"] == verifier.canonical_hash(row["prompt"])
    assert len(ids) == len(labels) == item["total_tokens"]


def test_exact_length_allowed_but_any_overflow_is_refused_not_clipped():
    row = record()
    length = verifier.encode_record(row, Tokenizer(), 8192)["total_tokens"]
    assert verifier.encode_record(row, Tokenizer(), length)["total_tokens"] == length
    with pytest.raises(verifier.VerifierError, match="refusing truncation"):
        verifier.encode_record(row, Tokenizer(), length - 1)


def test_unstable_chat_template_and_token_boundary_are_refused():
    class BadTemplate(Tokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return ("wrong" if add_generation_prompt else "other")
    with pytest.raises(verifier.VerifierError, match="template"):
        verifier.encode_record(record(), BadTemplate(), 8192)

    class BadTokens(Tokenizer):
        def __call__(self, text, *, add_special_tokens):
            return {"input_ids": [len(text)] + [ord(c) for c in text]}
    with pytest.raises(verifier.VerifierError, match="token boundary"):
        verifier.encode_record(record(), BadTokens(), 8192)


def test_eos_must_be_in_supervised_completion():
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 999999
    with pytest.raises(verifier.VerifierError, match="end-of-turn"):
        verifier.encode_record(record(), tokenizer, 8192)


def prediction_row(target, prediction, **labels):
    raw = prediction if isinstance(prediction, str) else json.dumps({"labels": prediction})
    return {"target_labels": {"evidence_support": target, **labels},
            "prediction": verifier.parse_prediction(raw), "generation_truncated": False}


def test_metric_invalid_predictions_penalized_not_removed_and_lists_not_scalar():
    rows = [prediction_row("supported", {"evidence_support": "supported"}, errors=["e1"], unused=None),
            prediction_row("supported", "not json", errors=[], unused=None),
            prediction_row("contradicted", {"evidence_support": "supported"}, errors=[], unused=None),
            prediction_row("contradicted", {"evidence_support": "contradicted"}, errors=[], unused=None)]
    result = verifier.score_predictions(rows)
    metric = result["fields"]["evidence_support"]
    assert result["json_valid_rate"] == 0.75
    assert result["label_schema_valid_rate"] == 0.75
    assert metric["evaluated"] == 4
    assert metric["invalid_or_missing_predictions"] == 1
    assert metric["per_class"]['"supported"']["precision"] == 0.5
    assert metric["per_class"]['"supported"']["recall"] == 0.5
    assert metric["per_class"]['"contradicted"']["f1"] == pytest.approx(2 / 3)
    assert metric["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert metric["reference_absent_classes"] == ["insufficient", "mixed"]
    assert metric["macro_f1_all_four_classes"] == pytest.approx((0.5 + 2 / 3) / 4)
    assert result["fields"]["errors"]["non_scalar_targets_not_scored"] == 4
    assert result["fields"]["unused"]["not_applicable_or_absent"] == 4


def test_json_valid_but_missing_labels_is_not_format_success():
    pred = verifier.parse_prediction('{"answer":"无法判断"}')
    assert pred["json_valid"] and not pred["label_schema_valid"]
    result = verifier.score_predictions([{"target_labels": {"evidence_support": "supported"}, "prediction": pred}])
    assert result["label_schema_valid_rate"] == 0
    assert result["fields"]["evidence_support"]["macro_f1"] == 0
    invalid = verifier.parse_prediction('{"labels":{"evidence_support":"unknown"}}')
    assert invalid["json_valid"] and not invalid["label_schema_valid"]


def test_bool_integer_string_classes_do_not_collapse():
    assert len({verifier.class_key(False), verifier.class_key(0), verifier.class_key("false")}) == 3


@pytest.mark.parametrize("options,pattern", [
    (["--mode", "train"], "explicit"),
    (["--mode", "evaluate", "--gpu", "1"], "eval-data"),
    (["--epochs", "4"], "epochs"),
    (["--epochs", "nan"], "epochs"),
    (["--max-seconds", "28801"], "8h"),
    (["--max-seconds", "100"], "safety margin"),
    (["--max-steps", "0"], "max-steps"),
    (["--limit", "1"], "evaluation-only"),
    (["--adapter", "/local/adapter"], "only for evaluate"),
    (["--eval-split", "test"], "Test requires"),
    (["--mode", "train", "--gpu", "1", "--eval-split", "test", "--allow-test"], "never a training"),
])
def test_cli_resource_and_test_gates(options, pattern):
    with pytest.raises(verifier.VerifierError, match=pattern):
        verifier.validate_args(args(*options))


def test_single_gpu_only_and_future_idle_gpu_indices_allowed(monkeypatch):
    verifier.validate_args(args("--mode", "train", "--gpu", "0"))
    verifier.validate_args(args("--mode", "train", "--gpu", "3"))
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(verifier.VerifierError, match="distributed"):
        verifier.validate_args(args("--mode", "train", "--gpu", "1"))


def test_busy_gpu_is_never_stopped_or_silently_reassigned(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="GPU-unit, NVIDIA GeForce RTX 5090, 10311, 0\n")
    monkeypatch.setattr(verifier.subprocess, "run", run)
    with pytest.raises(verifier.VerifierError, match="no processes will be stopped"):
        verifier.inspect_gpu(0)
    assert len(calls) == 1 and calls[0][0] == "nvidia-smi"
    monkeypatch.setattr(verifier.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="GPU-unit, RTX 5090, 15, 0\n"))
    assert verifier.inspect_gpu(2)["uuid"] == "GPU-unit"


def test_output_must_be_new_directory_and_is_portable(tmp_path):
    with pytest.raises(verifier.VerifierError, match="output-dir"):
        verifier.output_path(None)
    assert verifier.output_path(tmp_path / "new") == tmp_path / "new"
    for broad in (Path("/"), Path.home(), Path.cwd()):
        with pytest.raises(verifier.VerifierError, match="root"):
            verifier.output_path(broad)
    with pytest.raises(verifier.VerifierError, match="Output exists"):
        verifier.output_path(tmp_path)


def test_adapter_requires_completed_training_hash_and_same_base(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights-not-a-model")
    (adapter / "adapter_config.json").write_text("{}")
    manifest = {"schema": verifier.SCHEMA, "status": "completed", "mode": "train",
                "model": {"snapshot_sha256": "base"},
                "adapter_sha256": verifier.sha256_file(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": verifier.sha256_file(adapter / "adapter_config.json")}
    verifier.write_json(tmp_path / "run_manifest.json", manifest)
    assert verifier.verify_adapter(adapter, {"snapshot_sha256": "base"})["adapter_sha256"] == manifest["adapter_sha256"]
    with pytest.raises(verifier.VerifierError, match="base-model"):
        verifier.verify_adapter(adapter, {"snapshot_sha256": "other"})
    (adapter / "adapter_config.json").write_text('{"tampered":true}')
    with pytest.raises(verifier.VerifierError, match="hash mismatch"):
        verifier.verify_adapter(adapter, {"snapshot_sha256": "base"})


def test_model_identity_verifies_actual_shard_and_tokenizer_hashes(tmp_path):
    model = tmp_path / "local-model"
    model.mkdir()
    documents = {"config.json": {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36, "max_position_embeddings": 262144},
                 "model.safetensors.index.json": {"weight_map": {"weight": "weights.safetensors"}},
                 "tokenizer.json": {}, "tokenizer_config.json": {}}
    for name, value in documents.items():
        (model / name).write_text(json.dumps(value))
    (model / "weights.safetensors").write_bytes(b"fixture-bytes-not-model-weights")
    entries = [{"name": path.name, "sha256": verifier.sha256_file(path)} for path in model.iterdir()]
    verifier.write_json(model / "download_manifest.json", {"repo": verifier.MODEL_ID, "verified": True, "files": entries})
    snapshot = verifier.model_snapshot(model)
    assert set(snapshot["files_sha256"]) == set(documents) | {"weights.safetensors"}
    (model / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(verifier.VerifierError, match="hash mismatch"):
        verifier.model_snapshot(model)


def test_validation_mode_never_requests_gpu_or_writes_output(monkeypatch):
    monkeypatch.setattr(verifier, "gpu_lease", lambda *a: pytest.fail("validation cannot lease GPU"))
    monkeypatch.setattr(verifier, "prepare", lambda a: ({"status": "validated"}, [], [], [], [], None))
    assert verifier.run(args())["status"] == "validated"
    assert verifier.os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert verifier.os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.requires_model
def test_real_local_qwen_and_trl_collator_mask_prompt_and_padding(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl.trainer.sft_trainer")
    if not (verifier.MODEL_PATH / "tokenizer_config.json").is_file():
        pytest.skip("local tokenizer absent; never download")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(verifier.MODEL_PATH), local_files_only=True, trust_remote_code=False)
    first, second = record(), record("c2", "t2")
    second["prompt"][1]["content"] = "更短证据。"
    encoded = [verifier.encode_record(row, tokenizer, 8192) for row in (first, second)]
    batch = trl.DataCollatorForLanguageModeling(pad_token_id=tokenizer.eos_token_id)([item["features"] for item in encoded])
    assert batch["labels"].device.type == "cpu"
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all().item()
    for index, item in enumerate(encoded):
        n = item["prompt_tokens"]
        assert (batch["labels"][index, :n] == -100).all().item()
        assert batch["labels"][index, n:item["total_tokens"]].tolist() == item["features"]["input_ids"][n:]
    import torch
    assert not torch.cuda.is_initialized()


def test_training_config_cannot_retokenize_or_silently_truncate():
    import ast
    tree = ast.parse(Path(verifier.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SFTConfig"]
    assert len(calls) == 1
    settings = {item.arg: item.value for item in calls[0].keywords}
    assert ast.literal_eval(settings["completion_only_loss"]) is True
    assert ast.literal_eval(settings["packing"]) is False
    assert ast.literal_eval(settings["dataset_kwargs"]) == {"skip_prepare_dataset": True}


@pytest.mark.requires_model
def test_actual_trl_trainer_dataloader_preserves_prebuilt_labels_on_cpu(tmp_path, monkeypatch):
    """Instantiate only a tiny random CPU model; never train/load 4B weights."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")
    trl = pytest.importorskip("trl")
    if not (verifier.MODEL_PATH / "tokenizer_config.json").is_file():
        pytest.skip("local tokenizer absent; never download")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(verifier.MODEL_PATH), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = verifier.encode_record(record(), tokenizer, 8192)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=8192, eos_token_id=tokenizer.eos_token_id))
    config = trl.SFTConfig(output_dir=str(tmp_path / "cpu-contract"), use_cpu=True,
                           bf16=False, fp16=False, tf32=False, report_to=[], max_steps=1,
                           per_device_train_batch_size=1, completion_only_loss=True,
                           dataset_kwargs={"skip_prepare_dataset": True}, packing=False,
                           max_length=8192, dataloader_pin_memory=False)
    trainer = trl.SFTTrainer(model=model, args=config,
                            train_dataset=datasets.Dataset.from_list([encoded["features"]]),
                            processing_class=tokenizer)
    batch = next(iter(trainer.get_train_dataloader()))
    assert batch["labels"].device.type == "cpu"
    assert batch["labels"][0].tolist() == encoded["features"]["labels"]
    assert trainer.state.global_step == 0  # No optimizer step or model generation.
    import torch
    assert not torch.cuda.is_initialized()


def test_partial_evaluation_has_explicit_empty_denominator():
    report = verifier.score_predictions([])
    assert report["evaluated_records"] == 0
    assert report["json_valid_rate"] is None
    assert report["fields"] == {}
