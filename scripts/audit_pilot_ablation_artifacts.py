#!/usr/bin/env python3
"""Audit Pilot A/B/C/D cell files and cross-arm artifact bindings."""
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


def _write(path: str, value: object) -> None:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    if path == "-":
        print(rendered)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "独立审计 Pilot A/B/C/D artifact 文件、manifest hash 与跨臂绑定；"
            "不等同于 suite_state 网格审计"
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", default="-")
    parser.add_argument(
        "--allow-test-fixture",
        action="store_true",
        help=(
            "仅用于离线测试 fixture；报告仍标 non_production=true "
            "且 production_ready=false"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from evaluator.ablation_artifacts import audit_pilot_ablation_artifacts

    args = build_parser().parse_args(argv)
    try:
        result = audit_pilot_ablation_artifacts(
            args.suite_dir,
            allow_test_fixture=args.allow_test_fixture,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"A/B/C/D artifact 审计无法启动：{exc}") from exc
    _write(args.output, result)
    accepted = (
        bool(result["test_fixture_audit_ok"])
        if args.allow_test_fixture
        else bool(result["production_ready"])
    )
    if accepted:
        print(
            f"artifact audit {'TEST-ONLY OK' if args.allow_test_fixture else 'PRODUCTION OK'}："
            f"{result['records']['audited_cells']} cells；"
            f"报告 {args.output}",
            file=sys.stderr,
        )
        return 0
    if result.get("structural_audit_ok") and result.get("legacy_structural_only"):
        print(
            "artifact audit STRUCTURAL INTEGRITY OK / NON-FORMAL："
            f"{result['records']['audited_cells']} cells；input_schema="
            f"{result['input_schema']}，production_ready 需要 "
            "mitoevidence.pilot-ablation.v3 或 v4 正式协议；"
            f"报告 {args.output}",
            file=sys.stderr,
        )
        return 2
    print(
        f"artifact audit FAILED：{len(result['errors'])} errors；报告 {args.output}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
