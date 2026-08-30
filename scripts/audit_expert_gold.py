#!/usr/bin/env python3
"""Audit the project-owner-designated expert consensus gold snapshot.

This command is offline.  It verifies JSON parsing, record counts, unique IDs,
manifest SHA-256 values and field completeness.  It never fills a missing label
and explicitly reports that inter-expert agreement is unavailable when only the
consolidated annotation is present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json_atomic(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计 127 条专家共识金标（不补造标签）")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "annotation_prelabel" / "expert_gold_manifest.json",
        help="专家金标哈希与字段 designation manifest",
    )
    parser.add_argument("--out", type=Path, help="可选 JSON 报告；省略时只输出 stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    from evaluator.expert_gold import ExpertGoldAuditError, audit_expert_gold

    args = build_parser().parse_args(argv)
    try:
        report = audit_expert_gold(args.manifest, repo_root=REPO_ROOT)
    except ExpertGoldAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.out:
        _write_json_atomic(args.out, report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
