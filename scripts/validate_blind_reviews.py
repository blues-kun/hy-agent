#!/usr/bin/env python3
"""Validate locked A/B reviews or a completed third-expert adjudication.

Pair validation mechanically reports disagreements, raw agreement and Cohen's
kappa per item type.  Undefined kappa is reported as ``null`` with an explicit
marginal-degeneracy reason; it is never replaced with zero or one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expert-a", type=Path, required=True)
    parser.add_argument("--expert-b", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, help="neutral_items.jsonl（强烈建议提供）")
    parser.add_argument("--manifest", type=Path, help="盲标包 manifest.json")
    parser.add_argument("--out", type=Path, help="JSON 校验报告")


def main() -> int:
    from evaluator.blind import (
        BlindWorkflowError,
        make_adjudication_template,
        validate_adjudication,
        validate_blind_pair,
    )

    parser = argparse.ArgumentParser(description="校验人工双盲标注与第三人裁决")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair_parser = subparsers.add_parser("pair", help="校验 A/B 并生成一致性和分歧")
    _common_arguments(pair_parser)
    pair_parser.add_argument(
        "--adjudication-template",
        type=Path,
        help="仅在 A/B 校验通过时生成；final_decision 全部保持 null",
    )

    adjudication_parser = subparsers.add_parser("adjudication", help="校验第三人裁决")
    _common_arguments(adjudication_parser)
    adjudication_parser.add_argument("--adjudication", type=Path, required=True)
    args = parser.parse_args()

    common = {
        "neutral_packet_path": args.neutral,
        "package_manifest_path": args.manifest,
    }
    if args.command == "pair":
        report = validate_blind_pair(args.expert_a, args.expert_b, **common)
        if args.adjudication_template:
            try:
                template = make_adjudication_template(report)
            except BlindWorkflowError as exc:
                report["errors"].append(str(exc))
                report["ok"] = False
            else:
                _write_json(args.adjudication_template, template)
                print(f"空白裁决表：{args.adjudication_template}")
    else:
        report = validate_adjudication(
            args.adjudication,
            args.expert_a,
            args.expert_b,
            **common,
        )

    if args.out:
        _write_json(args.out, report)
        print(f"校验报告：{args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
