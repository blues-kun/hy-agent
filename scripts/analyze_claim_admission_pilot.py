#!/usr/bin/env python3
"""Hash-audit and analyze a completed Claim-admission Pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分析 Hy3 与单一合并专家参考的 Claim 准入一致性"
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.claim_admission_pilot import analyze_claim_admission_pilot, write_analysis

    args = build_parser().parse_args(argv)
    result = analyze_claim_admission_pilot(args.suite_dir, repo_root=REPO_ROOT)
    write_analysis(args.output, result)
    metrics = result["classification"]
    denominator = result["denominators"]
    print(f"结果：{args.output}")
    print(
        f"调用：成功 {denominator['succeeded_calls']}/{denominator['expected_calls']}；"
        f"失败 {denominator['failed_calls']}"
    )
    print(f"n={metrics['n']}")
    print(f"Raw accuracy={json.dumps(metrics['raw_accuracy'])}")
    print(f"Cohen κ={json.dumps(metrics['cohen_kappa'])}")
    print(f"Macro-F1={json.dumps(metrics['macro_f1'])}")
    baseline = result["baselines"]["four_class_majority"]
    print(
        "四分类多数类基线="
        f"{baseline['correct']}/{baseline['total']}="
        f"{json.dumps(baseline['accuracy'])}"
    )
    binary = result["binary_keep_candidate"]
    print(
        "二元可保留候选（accuracy/sensitivity/specificity/precision/F1）="
        + json.dumps(binary["metrics"], ensure_ascii=False, sort_keys=True)
    )
    print(
        "Predicted uncertain/abstain rate="
        + json.dumps(
            result["abstention_and_failure"]["predicted_uncertain_rate_among_succeeded"]
        )
    )
    print("注意：这是系统-单一专家共识参考一致性，不是专家间一致性。")
    print("二元视角只衡量候选召回/拦截能力，不能替代四分类 Claim 准入分析。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
