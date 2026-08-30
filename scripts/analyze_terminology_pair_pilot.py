#!/usr/bin/env python3
"""Audit and analyze a completed terminology-pair Pilot suite."""
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
        description="分析术语/条件错误成对 Pilot；不解释为全文证据一致性"
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.terminology_pair_pilot import (
        analyze_terminology_pair_pilot,
        write_analysis,
    )

    args = build_parser().parse_args(argv)
    result = analyze_terminology_pair_pilot(args.suite_dir, repo_root=REPO_ROOT)
    write_analysis(args.output, result)
    metrics = result["metrics"]
    denominators = result["denominators"]
    print(f"结果：{args.output}")
    print(
        "调用："
        f"成功 {denominators['succeeded_calls']}/{denominators['expected_calls']}；"
        f"失败 {denominators['failed_calls']}"
    )
    print(
        "Pair accuracy（所有结构有效调用，弃权计非正确）："
        f"{json.dumps(metrics['pair_accuracy_all_schema_valid_calls'])}"
    )
    print(
        "攻击误判率（选择 wrong 侧/所有结构有效调用）："
        f"{json.dumps(metrics['attack_misjudgment_rate_all_schema_valid_calls'])}"
    )
    print(
        "重复两两一致率："
        f"{json.dumps(metrics['repeat_pairwise_agreement_rate'])}"
    )
    print(
        "长度-only 基线："
        f"{json.dumps(metrics['length_only_baseline_accuracy_non_tied_available_pairs'])}"
    )
    print("解释边界：术语/条件错误成对判别，不是全文检索、引用或 Claim-Evidence 核验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
