"""金标语料工具链（evaluator/gold.py）的离线测试。"""
from __future__ import annotations

import json
from pathlib import Path

from evaluator.gold import (
    check_split_leakage,
    load_gold_records,
    load_split_map,
    validate_corpus,
)
from evaluator.schemas import QuestionGold

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "eval" / "data" / "questions.sample.jsonl"


def minimal_record(question_id="Q1", doi="10.1234/demo.2025", **overrides) -> dict:
    record = {
        "question_id": question_id,
        "question": "示例问题？",
        "answerability": "answerable",
        "required_claims": [
            {
                "claim_id": "C1",
                "text": "示例主张。",
                "is_core": True,
                "citations": [
                    {"doi_or_pmid": doi, "paper_id": "P1", "evidence_span_ids": ["S1"]}
                ],
            }
        ],
        "evidence_papers": [
            {"paper_id": "P1", "doi_or_pmid": doi, "source_access": "fulltext"}
        ],
        "evidence_spans": [
            {
                "span_id": "S1",
                "paper_id": "P1",
                "doi_or_pmid": doi,
                "source_access": "fulltext",
                "anchor": {"exact": "示例原文片段。"},
            }
        ],
    }
    record.update(overrides)
    return record


def as_gold(payload: dict) -> QuestionGold:
    return QuestionGold.model_validate(payload)


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------


def test_sample_corpus_loads_and_validates_clean():
    records, line_errors = load_gold_records(SAMPLE_PATH)
    assert line_errors == []
    assert [r.question_id for r in records] == ["Q-SAMPLE-001", "Q-SAMPLE-002"]

    report = validate_corpus(records)
    assert report.ok, report.errors
    assert report.n_records == 2
    assert report.answerability_counts == {"answerable": 1, "out_of_scope": 1}
    assert report.n_required_claims == 1
    assert report.n_evidence_spans == 3


def test_bad_json_line_is_reported_without_killing_the_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps(minimal_record("Q1"), ensure_ascii=False),
        "{ 这不是合法 JSON",
        json.dumps(minimal_record("Q2"), ensure_ascii=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    records, line_errors = load_gold_records(path)
    assert [r.question_id for r in records] == ["Q1", "Q2"]
    assert len(line_errors) == 1 and "第 2 行" in line_errors[0]


def test_schema_violation_is_reported_with_question_id(tmp_path):
    bad = minimal_record("Q-BAD")
    bad["required_claims"][0]["is_core"] = False  # required_claims 必须 is_core=True（方案 8.1）
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    records, line_errors = load_gold_records(path)
    assert records == []
    assert len(line_errors) == 1 and "Q-BAD" in line_errors[0]


def test_comment_and_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "# 注释\n\n" + json.dumps(minimal_record("Q1"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records, line_errors = load_gold_records(path)
    assert len(records) == 1 and line_errors == []


# ---------------------------------------------------------------------------
# 语料级检查
# ---------------------------------------------------------------------------


def test_duplicate_question_id_is_an_error():
    records = [as_gold(minimal_record("Q1")), as_gold(minimal_record("Q1"))]
    report = validate_corpus(records)
    assert not report.ok
    assert any("question_id 重复" in e for e in report.errors)


def test_span_referencing_missing_paper_is_an_error():
    payload = minimal_record("Q1")
    payload["evidence_spans"][0]["paper_id"] = "P404"
    report = validate_corpus([as_gold(payload)])
    assert any("不在 evidence_papers 中" in e for e in report.errors)


def test_claim_citing_missing_span_is_an_error():
    payload = minimal_record("Q1")
    payload["required_claims"][0]["citations"][0]["evidence_span_ids"] = ["S404"]
    report = validate_corpus([as_gold(payload)])
    assert any("不存在的 span_id" in e for e in report.errors)


def test_invalid_identifier_syntax_is_an_error():
    payload = minimal_record("Q1", doi="not-a-doi-or-pmid")
    report = validate_corpus([as_gold(payload)])
    assert any("标识符语法非法" in e for e in report.errors)


def test_duplicate_span_id_is_an_error():
    payload = minimal_record("Q1")
    payload["evidence_spans"].append(dict(payload["evidence_spans"][0]))
    report = validate_corpus([as_gold(payload)])
    assert any("span_id 重复" in e for e in report.errors)


def test_known_conflicts_without_flagged_paper_is_a_warning_not_error():
    payload = minimal_record("Q1", known_conflicts=["P1 与 P2 方向不一致"])
    report = validate_corpus([as_gold(payload)])
    assert report.ok  # 语义自洽问题不阻断（不裁决方案未定义口径）
    assert any("is_conflict_or_negative" in w for w in report.warnings)


def test_insufficient_with_required_claims_is_a_warning():
    payload = minimal_record("Q1", answerability="insufficient")
    report = validate_corpus([as_gold(payload)])
    assert report.ok
    assert any("answerability=insufficient" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# 校准/盲测分集泄漏（方案 9.2）
# ---------------------------------------------------------------------------


def test_same_paper_across_splits_is_leakage_error():
    # Q1 与 Q2 共享同一论文（DOI 大小写不同也要规范化后识破）。
    q1 = as_gold(minimal_record("Q1", doi="10.1234/shared.2025"))
    q2 = as_gold(minimal_record("Q2", doi="10.1234/SHARED.2025"))
    errors, _ = check_split_leakage([q1, q2], {"Q1": "calibration", "Q2": "blind"})
    assert len(errors) == 1 and "跨分集泄漏" in errors[0]


def test_same_paper_within_one_split_is_fine():
    q1 = as_gold(minimal_record("Q1", doi="10.1234/shared.2025"))
    q2 = as_gold(minimal_record("Q2", doi="10.1234/shared.2025"))
    errors, _ = check_split_leakage([q1, q2], {"Q1": "blind", "Q2": "blind"})
    assert errors == []


def test_unassigned_question_is_a_warning():
    q1 = as_gold(minimal_record("Q1"))
    errors, warnings = check_split_leakage([q1], {})
    assert errors == []
    assert any("未分配分集" in w for w in warnings)


def test_split_map_rejects_unknown_split_value(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"Q1": "test"}), encoding="utf-8")
    split_map, errors = load_split_map(path)
    assert split_map == {}
    assert len(errors) == 1 and "非法" in errors[0]


def test_validate_corpus_wires_split_counts_and_leakage():
    q1 = as_gold(minimal_record("Q1", doi="10.1234/a.2025"))
    q2 = as_gold(minimal_record("Q2", doi="10.1234/a.2025"))
    report = validate_corpus([q1, q2], split_map={"Q1": "calibration", "Q2": "blind"})
    assert report.split_counts == {"calibration": 1, "blind": 1}
    assert any("跨分集泄漏" in e for e in report.errors)
