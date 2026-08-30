#!/usr/bin/env python3
"""金标语料校验 CLI（方案 9.2 Schema + 语料级完整性 + 分集泄漏检查）。

用法：
  python scripts/validate_gold.py eval/data/questions.sample.jsonl
  python scripts/validate_gold.py eval/data/questions.jsonl --split eval/data/splits.json
  python scripts/validate_gold.py eval/data/questions.jsonl --out results/gold_report.json

分集文件为 {question_id: "calibration"|"blind"} 的 JSON 对象；给出后检查
同一论文（规范化 DOI/PMID）是否跨校准/盲测集合泄漏（方案 9.2 禁止）。
退出码：0 = 无 error（warning 不阻断）；1 = 存在 error。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from evaluator.gold import load_gold_records, load_split_map, validate_corpus

    ap = argparse.ArgumentParser(description="校验金标语料 questions.jsonl")
    ap.add_argument("input", type=Path, help="QuestionGold JSONL 文件")
    ap.add_argument("--split", type=Path, help="分集文件 {question_id: calibration|blind}")
    ap.add_argument("--out", type=Path, help="JSON 报告输出路径")
    args = ap.parse_args()

    records, line_errors = load_gold_records(args.input)
    split_map = None
    if args.split:
        split_map, split_errors = load_split_map(args.split)
        line_errors = line_errors + split_errors
    report = validate_corpus(records, line_errors=line_errors, split_map=split_map)

    print(f"# 金标语料校验：{args.input}")
    print(
        f"记录 {report.n_records} 条；required_claims {report.n_required_claims}，"
        f"evidence_papers {report.n_evidence_papers}，evidence_spans {report.n_evidence_spans}"
    )
    print(f"answerability 分布：{report.answerability_counts or '（空）'}")
    if report.split_counts:
        print(f"分集：{report.split_counts}（方案 9.2 目标 calibration=10 / blind=30）")
    for error in report.errors:
        print(f"ERROR   {error}")
    for warning in report.warnings:
        print(f"WARNING {warning}")
    print(f"结论：{'通过（无 error）' if report.ok else f'不通过（{len(report.errors)} 个 error）'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 报告：{args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
