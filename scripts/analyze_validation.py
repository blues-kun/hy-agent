#!/usr/bin/env python3
"""Analyse evaluation-method validity from one explicit JSON document.

Examples::

    python scripts/analyze_validation.py --input validation.json --output results.json
    cat validation.json | python scripts/analyze_validation.py --input - --output -
    python scripts/analyze_validation.py --print-input-schema

The command is fully offline and never imputes missing ratings.  Undefined or
under-sized metrics are emitted as JSON ``null`` together with warnings.
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


def _read_json(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _write_json(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        sys.stdout.write("\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线分析评估方法的判别力、评审一致性、稳定性与对抗鲁棒性"
    )
    parser.add_argument("--input", default="-", help="ValidationInput JSON；默认从 stdin 读取")
    parser.add_argument("--output", default="-", help="分析结果 JSON；默认写 stdout")
    parser.add_argument(
        "--print-input-schema",
        action="store_true",
        help="输出严格输入 JSON Schema 后退出，不读取 --input",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from evaluator.validation import ValidationInput, analyze_validation

    args = build_parser().parse_args(argv)
    if args.print_input_schema:
        rendered = json.dumps(ValidationInput.model_json_schema(), ensure_ascii=False, indent=2)
        _write_json(args.output, rendered)
        return 0

    try:
        payload = json.loads(_read_json(args.input))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"输入 JSON 读取失败：{exc}") from exc
    try:
        data = ValidationInput.model_validate(payload)
        result = analyze_validation(data)
    except (ValidationError, ValueError) as exc:
        raise SystemExit(f"验证数据不合规：{exc}") from exc

    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    _write_json(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
