import copy
import json
from pathlib import Path

from rp_grpo.train_planner_sft import (
    encode_last_action, encode_demonstrations, load_reference_index,
    read_demonstrations, coverage_report, verifier_helpers,
    filter_reference_budget, encode_complete_trajectories,
)
import pytest


class Tokenizer:
    eos_token_id = 256

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not enable_thinking
        text = "".join(f"<{m['role']}>" + m["content"] + "</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode()) if tokenize else text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode())}

    def decode(self, tokens, **kwargs):
        parts, start = [], 0
        for index, token in enumerate(tokens):
            if token == self.eos_token_id:
                parts.extend([bytes(tokens[start:index]).decode(), "<eos>"])
                start = index + 1
        parts.append(bytes(tokens[start:]).decode())
        return "".join(parts)


def row():
    return {"id": "test", "messages": [
        {"role": "system", "content": "tool contract"},
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "ACTUAL TOOL OBSERVATION"},
        {"role": "assistant", "content": '{"tool":"finish","arguments":{}}'},
    ]}


def test_only_last_assistant_supervised():
    item = row()
    result = encode_last_action(item, Tokenizer(), 4096)
    target = Tokenizer().decode([x for x in result["labels"] if x != -100])
    assert target == item["messages"][-1]["content"] + "<eos>"
    assert "ACTUAL TOOL OBSERVATION" not in target
    assert "old action" not in target
    assert len(result["input_ids"]) == len(result["labels"]) == len(result["attention_mask"])


def test_no_silent_truncation():
    with pytest.raises(ValueError, match="silently truncated"):
        encode_last_action(row(), Tokenizer(), 10)


def test_prefix_is_identical_to_runtime_not_full_training_template():
    from rp_grpo.train_policy import format_prompt
    class DifferentFullTemplate(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is True
            return super().apply_chat_template(messages, **kwargs)
    tokenizer = DifferentFullTemplate()
    item = row()
    expected = tokenizer(format_prompt(tokenizer, item["messages"][:-1]), add_special_tokens=False)["input_ids"]
    result = encode_last_action(item, tokenizer, 4096)
    assert result["input_ids"][:len(expected)] == expected
    assert result["labels"][:len(expected)] == [-100] * len(expected)


def test_batch_encoding_container_has_identical_mask_to_list_api():
    class ListTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__(text, **kwargs)["input_ids"]
    assert encode_last_action(row(), ListTokenizer(), 4096) == encode_last_action(row(), Tokenizer(), 4096)


@pytest.mark.parametrize("value", [{"attention_mask": [1]}, {"input_ids": [[1, 2]]}, [1, True], [1, -1]])
def test_malformed_or_batched_token_ids_rejected(value):
    class InvalidTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            return value
    with pytest.raises(ValueError, match="one unpadded sequence"):
        encode_last_action(row(), InvalidTokenizer(), 4096)


@pytest.fixture(scope="module")
def actual_reference(tmp_path_factory):
    """Use existing manifest-bound CPU data; never alter the frozen snapshot."""
    from rp_grpo.environment import DEFAULT_DATA, load_tasks
    with (DEFAULT_DATA / "reference_traces.jsonl").open(encoding="utf-8") as stream:
        reference = json.loads(next(stream))
    tasks = {task["id"]: task for task in load_tasks(data_dir=DEFAULT_DATA)}
    samples = []
    with (DEFAULT_DATA / f"planner_sft.{reference['split']}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["reference_trace_id"] == reference["id"]:
                samples.append(sample)
            elif samples:
                break
    directory = tmp_path_factory.mktemp("sft-reference")
    path = directory / "reference_traces.jsonl"
    path.write_text(json.dumps(reference, ensure_ascii=False) + "\n", encoding="utf-8")
    index = load_reference_index(path, tasks, data_dir=DEFAULT_DATA)
    return {"data_dir": DEFAULT_DATA, "reference": reference, "index": index, "tasks": tasks, "samples": samples}


def read_rows(tmp_path, actual_reference, rows):
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    return read_demonstrations(path, actual_reference["reference"]["split"],
                               actual_reference["tasks"], actual_reference["index"])


def test_actual_replayed_reference_without_self_reported_success(tmp_path, actual_reference):
    samples = actual_reference["samples"]
    assert all("reference_success" not in item for item in samples)
    assert read_rows(tmp_path, actual_reference, samples) == samples
    assert sum(len(item["steps"]) for item in actual_reference["index"].values()) == len(samples)


@pytest.mark.parametrize("field,value,match", [
    ("reference_trace_sha256", "0" * 64, "hash"),
    ("environment_task_sha256", "0" * 64, "hash"),
    ("trace_step_index", True, "reference step"),
    ("trace_step_index", 999, "reference step"),
    ("reference_trace_id", "nonexistent", "reference step"),
    ("pair_id", "another-pair", "metadata"),
    ("group_id", "another-source", "metadata"),
    ("route", "fabricated-route", "metadata"),
    ("expert_trajectory", True, "provenance"),
])
def test_reference_bindings_cannot_be_replaced_by_success_flag(tmp_path, actual_reference, field, value, match):
    samples = copy.deepcopy(actual_reference["samples"])
    samples[0][field] = value
    samples[0]["reference_success"] = True
    with pytest.raises(ValueError, match=match):
        read_rows(tmp_path, actual_reference, samples)


@pytest.mark.parametrize("mode", ["observation", "target", "hidden_gold"])
def test_exact_actor_history_and_target_are_bound(tmp_path, actual_reference, mode):
    samples = copy.deepcopy(actual_reference["samples"])
    sample = samples[-1]
    if mode == "target":
        action = json.loads(sample["messages"][-1]["content"])
        action["arguments"]["reason"] = "Forged terminal rationale"
        sample["messages"][-1]["content"] = json.dumps(action)
    elif mode == "hidden_gold":
        sample["messages"].insert(-1, {"role": "user", "content": "gold: available; reward: 1.0"})
    else:
        sample["messages"][1]["content"] += " fabricated context"
    with pytest.raises(ValueError, match="actual replayed history/action"):
        read_rows(tmp_path, actual_reference, samples)


def test_every_reference_step_must_exist_once_before_filter(tmp_path, actual_reference):
    samples = copy.deepcopy(actual_reference["samples"])
    duplicate = copy.deepcopy(samples[0])
    duplicate["id"] += "-duplicate"
    with pytest.raises(ValueError, match="same reference step"):
        read_rows(tmp_path, actual_reference, [*samples, duplicate])
    with pytest.raises(ValueError, match="every bound reference step"):
        read_rows(tmp_path, actual_reference, samples[:-1])


@pytest.mark.parametrize("resign", [False, True])
def test_reference_is_independently_replayed_not_only_hash_checked(tmp_path, actual_reference, resign):
    reference = copy.deepcopy(actual_reference["reference"])
    trace = reference["environment_trace"]
    trace["steps"][0]["observation"]["invented_success"] = True
    if resign:
        helper = verifier_helpers()
        trace["steps"][0]["observation_sha256"] = helper.canonical_hash(trace["steps"][0]["observation"])
        trace["trace_sha256"] = helper.canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"})
    path = tmp_path / "references.jsonl"
    path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="replay rejected"):
        load_reference_index(path, actual_reference["tasks"], data_dir=actual_reference["data_dir"])


def test_unfinished_reference_cannot_warm_start_policy(tmp_path, actual_reference):
    reference = copy.deepcopy(actual_reference["reference"])
    reference["environment_trace"]["done"] = False
    reference["environment_trace"]["terminal"] = None
    path = tmp_path / "references.jsonl"
    path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="successful terminal"):
        load_reference_index(path, actual_reference["tasks"], data_dir=actual_reference["data_dir"])


@pytest.mark.parametrize("split", ["train", "dev"])
def test_explicit_long_example_filter_keeps_exact_targets_and_reports_exclusion(split):
    short = {**row(), "task_id": "short", "split": split}
    long = copy.deepcopy(short)
    long.update(id="long", task_id="long")
    long["messages"][1]["content"] *= 2000
    encoded, retained, excluded = encode_demonstrations([short, long], Tokenizer(), 4096, skip_overlength=True)
    assert retained == [short] and len(encoded) == 1
    assert excluded[0]["id"] == "long" and excluded[0]["tokens"] > excluded[0]["limit"] == 4096
    assert excluded[0]["reason"] == "overlength_not_truncated"
    assert Tokenizer().decode([token for token in encoded[0]["labels"] if token != -100]) == short["messages"][-1]["content"] + "<eos>"
    with pytest.raises(ValueError, match="silently truncated"):
        encode_demonstrations([short, long], Tokenizer(), 4096)


def coverage_fixture():
    tasks, refs = {}, {}
    for side in ("available", "view_without_target"):
        task_id = "task-" + side
        tasks[task_id] = {"id": task_id, "split": "train", "pair_id": "pair", "side": side}
        refs[side] = {"task_id": task_id, "split": "train", "pair_id": "pair", "side": side,
                      "route": "catalog_read", "reference_max_steps": 64,
                      "steps": [{"tool": tool, "answerability": "answerable" if side == "available" else "insufficient"}
                                for tool in ("list_documents", "read_passage", "finalize")]}
    rows = [{"reference_trace_id": "available", "trace_step_index": index} for index in range(3)]
    return tasks, refs, rows


def test_surviving_first_step_is_not_terminal_or_complete_pair():
    tasks, refs, rows = coverage_fixture()
    rows.append({"reference_trace_id": "view_without_target", "trace_step_index": 0})
    report = coverage_report(rows, refs, tasks, "train")
    assert report["tasks_with_any_step"] == 2
    assert report["tasks_with_terminal_step"] == report["tasks_with_complete_trajectory"] == 1
    assert report["pairs_with_both_sides_terminal"] == report["pairs_with_both_sides_complete_trajectory"] == 0
    missing = next(item for item in report["traces"] if item["side"] == "view_without_target")
    assert missing["missing_step_indices"] == [1, 2]


def test_terminal_survives_without_complete_supervision():
    tasks, refs, rows = coverage_fixture()
    rows.append({"reference_trace_id": "view_without_target", "trace_step_index": 2})
    report = coverage_report(rows, refs, tasks, "train")
    assert report["tasks_with_terminal_step"] == 2 and report["complete_trajectories"] == 1
    assert report["pairs_with_both_sides_terminal"] == 1
    assert report["pairs_with_both_sides_complete_trajectory"] == 0
    assert report["terminal_answerability_counts"] == {"answerable": 1, "insufficient": 1}
    assert report["routes"]["catalog_read"]["traces_with_terminal_step"] == 2


def test_filter_can_report_fully_excluded_trace_and_pair():
    tasks, refs, _ = coverage_fixture()
    report = coverage_report([], refs, tasks, "train")
    assert report["tasks_with_any_step"] == report["terminal_step_records"] == 0
    assert report["routes"]["catalog_read"]["fully_excluded_traces"] == 2
    assert len(report["missing_terminal_task_ids"]) == 2


def test_fully_excluded_partition_can_be_reported_without_enabling_training():
    sample = {**row(), "task_id": "test"}
    encoded, retained, excluded = encode_demonstrations([sample], Tokenizer(), 10,
                                                       skip_overlength=True, allow_empty=True)
    assert encoded == retained == [] and len(excluded) == 1
    with pytest.raises(ValueError, match="No complete"):
        encode_demonstrations([sample], Tokenizer(), 10, skip_overlength=True)


def test_target_bytes_and_one_eos_are_not_silently_normalized():
    sample = row()
    sample["messages"][-1]["content"] = '{"tool":"finalize","arguments":{"reason":"仅限当前授权快照"}}'
    result = encode_last_action(sample, Tokenizer(), 4096)
    target = [token for token in result["labels"] if token != -100]
    assert target.count(Tokenizer.eos_token_id) == 1 and target[-1] == Tokenizer.eos_token_id
    assert Tokenizer().decode(target[:-1]) == sample["messages"][-1]["content"]

    class NormalizingTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return super().decode(tokens, **kwargs).replace("当前", "其他")
    with pytest.raises(ValueError, match="target action bytes"):
        encode_last_action(sample, NormalizingTokenizer(), 4096)


def test_step_budget_excludes_entire_long_route_not_its_first_six_steps():
    tasks, refs, rows = coverage_fixture()
    refs["view_without_target"]["steps"] = [{"tool": "read_passage"}] * 6 + [{"tool": "finalize"}]
    rows.extend({"reference_trace_id": "view_without_target", "trace_step_index": index} for index in range(7))
    kept, excluded = filter_reference_budget(rows, refs, 6)
    assert len(kept) == 3 and all(item["reference_trace_id"] == "available" for item in kept)
    assert excluded[0]["actor_steps"] == excluded[0]["excluded_step_records"] == 7
    assert excluded[0]["reason"] == "whole_trajectory_exceeds_actor_step_budget"
    opt_in, excluded = filter_reference_budget(rows, refs, 64)
    assert opt_in == rows and excluded == []


def test_one_long_step_excludes_its_whole_trajectory_but_keeps_other_complete_route():
    _, refs, _ = coverage_fixture()
    samples = []
    for ref_id, ref in refs.items():
        for index in range(len(ref["steps"])):
            sample = copy.deepcopy(row())
            sample.update(id=f"{ref_id}-{index}", task_id=ref["task_id"], route=ref["route"],
                          reference_trace_id=ref_id, trace_step_index=index)
            if ref_id == "available" and index == 2:
                sample["messages"][1]["content"] *= 2000
            samples.append(sample)
    encoded, kept, excluded = encode_complete_trajectories(samples, refs, Tokenizer(), 4096, skip_overlength=True)
    assert len(encoded) == len(kept) == len(excluded) == 3
    assert all(item["reference_trace_id"] == "view_without_target" for item in kept)
    reasons = [item["reason"] for item in excluded]
    assert reasons.count("overlength_not_truncated") == 1
    assert reasons.count("whole_trajectory_excluded_due_to_another_overlength_step") == 2
    with pytest.raises(ValueError, match="complete reference trajectories"):
        encode_complete_trajectories(samples[:-1], refs, Tokenizer(), 4096, skip_overlength=True)
