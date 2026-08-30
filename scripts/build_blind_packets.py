#!/usr/bin/env python3
"""Build an optional neutral package for a *new* two-rater study.

Example:
  python scripts/build_blind_packets.py \
    --batch-id pilot-v1 --guideline-version rubric-0.2 \
    --out results/blind_packets/pilot-v1

The existing 127 records are already designated consolidated expert gold.  This
command is not required to make them gold and cannot recover their historical
inter-expert agreement.  If a separate future reliability study is requested,
the destination must be new; the command never overwrites an assignment and
never copies source-gold decisions/reasons into ``neutral_items.jsonl``.
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
    from evaluator.blind import BlindWorkflowError, build_blind_package

    parser = argparse.ArgumentParser(description="为可选的新可靠性研究生成不含现有金标的 A/B 包")
    parser.add_argument("--batch-id", required=True, help="不可复用的任务批次 ID")
    parser.add_argument("--guideline-version", required=True, help="专家标注指南版本")
    parser.add_argument("--out", type=Path, required=True, help="新输出目录（拒绝覆盖）")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=REPO_ROOT / "annotation_prelabel",
        help="现有专家共识金标目录；仅抽取来源字段，排除金标标签",
    )
    parser.add_argument(
        "--evidence-manifest",
        type=Path,
        default=REPO_ROOT / "eval" / "data" / "evidence_pool_manifest.json",
        help="综述池唯一事实来源",
    )
    args = parser.parse_args()

    try:
        manifest = build_blind_package(
            annotation_root=args.annotation_root,
            evidence_manifest_path=args.evidence_manifest,
            output_dir=args.out,
            batch_id=args.batch_id,
            guideline_version=args.guideline_version,
        )
    except BlindWorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"盲标包：{args.out.resolve()}")
    print(f"记录：{json.dumps(manifest['counts'], ensure_ascii=False)}")
    print(f"输入快照 SHA-256：{manifest['input_snapshot_sha256']}")
    print(
        "中性来源 SHA-256："
        f"{manifest['outputs']['neutral_items.jsonl']['sha256']}"
    )
    print("A/B 文件是可选新研究的空白模板；它们不是当前127条金标的缺失组成部分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
