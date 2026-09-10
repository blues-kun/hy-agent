"""Small synthetic tool fixtures; not research data or model evaluations."""
from copy import deepcopy
import hashlib
import json

import pytest

from rp_grpo.build_data import VERSION, canonical_hash, file_hash
from rp_grpo.environment import ResearchEnvironment, replay_trace


@pytest.fixture
def synthetic_snapshot(tmp_path):
    text = "Synthetic fixture: the test sample is blue."
    passage = {"evidence_id": "synthetic-passage", "paper_id": "synthetic-document",
               "section": "fixture", "text": text,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "source_sha256": hashlib.sha256(b"synthetic source").hexdigest()}
    views, tasks, scorers = [], [], []
    for available in (True, False):
        side = "available" if available else "view_without_target"
        view = {"id": "synthetic-view-" + side,
                "passage_ids": [passage["evidence_id"]] if available else [],
                "paper_ids": [passage["paper_id"]] if available else [],
                "measurement_row_ids": []}
        identifier = "synthetic-task-" + side
        task = {"id": identifier, "split": "dev", "pair_id": "synthetic-pair",
                "side": side, "group_id": "synthetic-group", "view_id": view["id"],
                "view_sha256": canonical_hash({k: v for k, v in view.items() if k != "id"}),
                "actor_observation": {"question": "Locate the synthetic sample description.",
                                      "retrieval_query": "test sample is blue"}}
        scorer = {"task_id": identifier, "kind": "exact_passage", "anchor": "test sample is blue",
                  "answerability": "answerable" if available else "insufficient",
                  "allowed_evidence_ids": [passage["evidence_id"]] if available else [],
                  "expected_claims": [{"evidence_id": passage["evidence_id"], "text": text}] if available else []}
        views.append(view)
        tasks.append(task)
        scorers.append(scorer)
    files = {"tasks.jsonl": tasks, "views.jsonl": views,
             "passages.jsonl": [passage], "scorers.jsonl": scorers}
    for name, rows in files.items():
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "measurements.csv").write_text("row_id\n", encoding="utf-8")
    manifest = {"schema_version": VERSION, "fixture_only": True,
                "files": {name: file_hash(tmp_path / name) for name in [*files, "measurements.csv"]}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, tasks, passage


def submit(env, passage):
    return env.step({"tool": "finalize", "arguments": {
        "answerability": "answerable", "scope": "authorized_snapshot_only",
        "claims": [{"evidence_id": passage["evidence_id"], "quote": passage["text"]}],
        "measurements": []}})


def test_source_reading_and_exact_submission_replay(synthetic_snapshot):
    path, tasks, passage = synthetic_snapshot
    env = ResearchEnvironment(path, max_steps=6)
    observation = env.reset(tasks[0])
    found = env.step({"tool": "search_evidence", "arguments": {"query": observation["retrieval_query"]}})
    assert found["observation"]["hits"][0]["evidence_id"] == passage["evidence_id"]
    env.step({"tool": "read_passage", "arguments": {"evidence_id": passage["evidence_id"]}})
    result = submit(env, passage)
    assert result["info"]["passed"] and result["reward"] > 0
    assert "reward" not in result["observation"]
    assert replay_trace(env.export_trace(), path)["passed"]


def test_correct_quote_without_reading_does_not_pass(synthetic_snapshot):
    path, tasks, passage = synthetic_snapshot
    env = ResearchEnvironment(path)
    env.reset(tasks[0])
    result = submit(env, passage)
    assert not result["info"]["passed"]
    assert not result["info"]["dimensions"]["citation_exact_and_read"]


def test_missing_view_keeps_initial_prompt_and_requires_scope_check(synthetic_snapshot):
    path, tasks, passage = synthetic_snapshot
    env = ResearchEnvironment(path)
    first, second = env.reset(tasks[0]), env.reset(tasks[1])
    assert first == second
    unavailable = env.step({"tool": "read_passage", "arguments": {"evidence_id": passage["evidence_id"]}})
    assert unavailable["observation"]["status"] == "unavailable_in_authorized_snapshot"
    env.step({"tool": "search_evidence", "arguments": {"query": second["retrieval_query"]}})
    result = env.step({"tool": "stop", "arguments": {
        "scope": "authorized_snapshot_only", "reason": "No match in this synthetic view."}})
    assert result["info"]["passed"]


def test_snapshot_and_task_tampering_are_rejected(synthetic_snapshot):
    path, tasks, _ = synthetic_snapshot
    env = ResearchEnvironment(path)
    altered = deepcopy(tasks[0])
    altered["actor_observation"]["question"] = "tampered"
    with pytest.raises(ValueError, match="differs from registered"):
        env.reset(altered)
    (path / "passages.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot file hash mismatch"):
        ResearchEnvironment(path)
