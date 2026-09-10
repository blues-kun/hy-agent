"""Portable CPU-only policy-environment tests; never load model weights."""
import copy
import json
import statistics
from collections import defaultdict

import pytest

from rp_grpo.environment import ResearchEnvironment, load_tasks, replay_trace, DEFAULT_DATA
from rp_grpo.build_data import records, canonical_hash
from rp_grpo.build_references import execute_reference

pytestmark = pytest.mark.requires_data


@pytest.fixture
def env():
    return ResearchEnvironment(max_steps=64)


def task(kind="text_evidence", side="available", variant=None):
    return next(row for row in load_tasks() if row["task_kind"] == kind and row["side"] == side
                and (variant is None or row["variant"] == variant))


def call(env, tool, **arguments):
    return env.step({"tool": tool, "arguments": arguments})


def locate(env, selected=None):
    observation = env.reset(selected or task())
    result = call(env, "search_evidence", query=observation["retrieval_query"])
    hit = next(row for row in result["observation"]["hits"] if row["exact_match"])
    return call(env, "read_passage", evidence_id=hit["evidence_id"])["observation"]


def finish(env, claims=None, measurements=None, answerability="answerable", reason="当前授权快照。"):
    return call(env, "finalize", answerability=answerability, scope="authorized_snapshot_only",
                claims=claims or [], measurements=measurements or [], reason=reason)


def test_split_group_pair_and_actor_observation_isolation(env):
    groups, pairs = defaultdict(set), defaultdict(list)
    for row in load_tasks():
        assert row["split"] in {"train", "dev"}
        groups[row["group_id"]].add(row["split"])
        pairs[row["pair_id"]].append(row)
    assert all(len(parts) == 1 for parts in groups.values())
    for members in pairs.values():
        assert len(members) == 2
        assert {row["side"] for row in members} == {"available", "view_without_target"}
        first, second = [env.reset(row) for row in members]
        assert first == second
        assert not ({"side", "split", "pair_id", "expected", "answerability", "id", "task_id"} & set(first))
    manifest = json.loads((DEFAULT_DATA / "MANIFEST.json").read_text())
    assert set(manifest["input_sha256"]) == {"primary_tool_tasks_v3/train.jsonl", "primary_tool_tasks_v3/development.jsonl", "family_a_splits.json", "compendium/isletmito.duckdb"}


@pytest.mark.parametrize("side", ["available", "view_without_target"])
def test_two_real_nonmacro_paths_execute_and_replay(env, side):
    selected = task(side=side)
    trace_a, _ = execute_reference(env, selected, "search_read")
    trace_b, _ = execute_reference(env, selected, "catalog_read")
    assert trace_a["terminal"]["passed"] and trace_b["terminal"]["passed"]
    assert trace_a["steps"][0]["action"]["tool"] == "search_evidence"
    assert trace_b["steps"][0]["action"]["tool"] == "list_documents"
    assert not any(s["action"]["tool"] == "search_evidence" for s in trace_b["steps"])
    assert replay_trace(json.loads(json.dumps(trace_a)))["passed"]
    assert replay_trace(json.loads(json.dumps(trace_b)))["passed"]
    assert trace_a["telemetry"]["call_count"] == len(trace_a["steps"])
    assert trace_a["telemetry"]["wall_time_s"] >= 0


def test_correct_gold_bytes_without_real_read_get_no_reward(env):
    selected = task()
    env.reset(selected)
    expected = next(row for row in records(DEFAULT_DATA / "scorers.jsonl") if row["task_id"] == selected["id"])
    claims = [{"evidence_id": row["evidence_id"], "quote": row["text"]} for row in expected["expected_claims"]]
    result = finish(env, claims=claims)
    assert result["done"] and not result["info"]["passed"] and result["reward"] <= 0
    assert "passed" not in result["observation"] and "reward" not in result["observation"]


def test_exact_citation_and_quote_are_checked_after_actual_read(env):
    passage = locate(env)
    result = finish(env, claims=[{"evidence_id": passage["evidence_id"], "quote": passage["text"] + " invented"}])
    assert not result["info"]["passed"]
    passage = locate(env)
    result = finish(env, claims=[{"evidence_id": passage["evidence_id"], "quote": passage["text"]}])
    assert result["info"]["passed"] and result["reward"] > 0


@pytest.mark.parametrize("bad", [
    {"tool": "run_workflow", "arguments": {}},
    {"tool": "query_metrics", "arguments": {"sql": "SELECT * FROM private"}},
    {"tool": "search_evidence", "arguments": {"query": "text", "path": "../../scorers.jsonl"}},
    [{"tool": "list_documents", "arguments": {}}, {"tool": "finalize", "arguments": {}}],
    {"tool": "macro", "arguments": {"actions": ["search", "read", "answer"]}},
])
def test_unknown_tools_macros_paths_and_sql_do_not_execute(env, bad):
    env.reset(task())
    result = env.step(bad)
    assert result["observation"]["status"] == "invalid_action"
    assert not result["done"] and result["reward"] is None


def test_cross_view_source_cannot_be_read(env):
    selected = task(side="view_without_target")
    observation = env.reset(selected)
    papers = set(observation["paper_ids"])
    foreign = next(row for row in records(DEFAULT_DATA / "passages.jsonl") if row["paper_id"] not in papers)
    result = call(env, "read_passage", evidence_id=foreign["evidence_id"])
    assert result["observation"]["status"] == "unavailable_in_authorized_snapshot"
    assert "text" not in result["observation"]


def test_unread_is_not_missing_and_missing_is_not_never_measured(env):
    missing = task(side="view_without_target")
    env.reset(missing)
    early = call(env, "stop", scope="authorized_snapshot_only", reason="输入不足")
    assert not early["info"]["passed"]
    observation = env.reset(missing)
    call(env, "search_evidence", query=observation["retrieval_query"])
    legitimate = call(env, "stop", scope="authorized_snapshot_only", reason="当前授权视图没有所需原文，不能推断其他资料。")
    assert legitimate["info"]["passed"]
    observation = env.reset(missing)
    call(env, "search_evidence", query=observation["retrieval_query"])
    overclaim = finish(env, answerability="insufficient", reason="真实实验没有测量该变量")
    assert not overclaim["info"]["passed"]


def test_wrong_query_does_not_establish_scope_exhaustion(env):
    env.reset(task(side="view_without_target"))
    call(env, "search_evidence", query="an entirely unrelated query")
    assert not finish(env, answerability="insufficient")["info"]["passed"]


def test_literal_statement_preserves_exact_span_without_claiming_graph_truth(env):
    passage = locate(env)
    statement = call(env, "read_statement", evidence_id=passage["evidence_id"], statement_index=0)["observation"]["statement"]
    assert passage["text"][statement["char_start"]:statement["char_end"]] == statement["text"]
    assert statement["kind"] == "literal_sentence_span"
    assert "causal" not in statement and "supported" not in statement


def test_metrics_use_real_csv_values_and_islet_unit(env):
    selected = task(kind="finite_metric")
    obs = env.reset(selected)
    args = {key: value for key, value in obs["measurement_request"].items() if key != "unit"}
    result = env.step({"tool": "query_metrics", "arguments": args})["observation"]
    rows = [row for row in env._measurements.values() if row["row_id"] in result["row_ids"]]
    groups = defaultdict(list)
    for row in rows:
        groups[row["islet_id"]].append(float(row["value"]))
    values = [statistics.median(value) for value in groups.values()]
    aggregate = statistics.mean if args["aggregation"] == "mean" else statistics.median
    assert result["value"] == aggregate(values)
    assert result["analysis_unit"] == "islet" and result["n_islets"] == len(groups) and result["n_mice"] is None
    submitted = finish(env, measurements=[{key: result[key] for key in ("result_id", "value", "unit")}])
    assert submitted["info"]["passed"]


def test_measurement_answer_cannot_fabricate_result_id(env):
    env.reset(task(kind="finite_metric"))
    fake = finish(env, measurements=[{"result_id": "not-run", "value": 1.0, "unit": "a.u."}])
    assert not fake["info"]["passed"]
    obs = env.reset(task(kind="finite_metric", side="view_without_target"))
    args = {key: value for key, value in obs["measurement_request"].items() if key != "unit"}
    result = env.step({"tool": "query_metrics", "arguments": args})
    assert result["observation"]["status"] == "unavailable_in_authorized_snapshot"
    assert finish(env, answerability="insufficient")["info"]["passed"]


def test_tool_error_setup_is_actual_feedback_and_not_a_model_action(env):
    selected = task(variant="tool_error_repair")
    obs = env.reset(selected)
    assert obs["previous_tool_feedback"]["observation"]["status"] == "invalid_action"
    assert len(env.export_trace()["setup_steps"]) == 1 and not env.export_trace()["steps"]
    trace, samples = execute_reference(env, selected, "search_read")
    assert trace["terminal"]["passed"] and trace["steps"][0]["action"]["arguments"]["query"]
    assert replay_trace(trace)["passed"]
    assert not any(json.loads(row["messages"][-1]["content"])["arguments"].get("query") == "" for row in samples)


def test_replay_rejects_forged_observation_even_when_outer_hash_is_recomputed(env):
    trace, _ = execute_reference(env, task(), "search_read")
    trace["steps"][0]["observation"]["exact_match_count"] = 777
    trace["steps"][0]["observation_sha256"] = canonical_hash(trace["steps"][0]["observation"])
    trace["trace_sha256"] = canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"})
    result = replay_trace(trace)
    assert not result["valid"] and not result["passed"]


def test_unfinished_trace_replays_as_unfinished_not_success(env):
    env.reset(task())
    result = replay_trace(env.export_trace())
    assert result["valid"] and not result["passed"] and result["reward"] is None


def test_replay_only_ignores_nondeterministic_timings_not_call_counts(env):
    trace, _ = execute_reference(env, task(), "search_read")
    trace["telemetry"]["call_count"] += 1
    trace["trace_sha256"] = canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"})
    assert not replay_trace(trace)["valid"]


def test_repeated_tools_increase_cost_not_reward(env):
    selected = task()
    direct, _ = execute_reference(env, selected, "search_read")
    obs = env.reset(selected)
    for _ in range(3):
        call(env, "list_documents")
    hits = call(env, "search_evidence", query=obs["retrieval_query"])["observation"]["hits"]
    hit = next(row for row in hits if row["exact_match"])
    source = call(env, "read_passage", evidence_id=hit["evidence_id"])["observation"]
    delayed = finish(env, claims=[{"evidence_id": source["evidence_id"], "quote": source["text"]}])
    assert delayed["info"]["passed"] and delayed["reward"] < direct["reward"]
