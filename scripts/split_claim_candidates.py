#!/usr/bin/env python3
"""Create independent, review-required atomic-claim candidates from an answer.

The input JSON contract contains only ``output_id``, ``question`` and ``answer``.
Any tested-system self-reported claim list is rejected as an extra field.
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


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def _write(path: str, rendered: str) -> None:
    if path == "-":
        sys.stdout.write(rendered + "\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线生成可审计的原子主张候选；结果须经人工确认，不能直接作为正式分母"
    )
    parser.add_argument("--input", required=True, help="输入 JSON；- 表示 stdin")
    parser.add_argument("--output", default="-", help="输出 JSON；- 表示 stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from evaluator.claim_splitter import ClaimSplitRequest, split_claim_candidates

    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(_read(args.input))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"输入 JSON 读取失败：{exc}") from exc
    try:
        request = ClaimSplitRequest.model_validate(payload)
        result = split_claim_candidates(request)
    except (ValidationError, ValueError) as exc:
        raise SystemExit(f"拆分输入不合规：{exc}") from exc
    _write(args.output, json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
