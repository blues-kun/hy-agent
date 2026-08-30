#!/usr/bin/env python3
"""Assemble one auditable nine-dimensional evaluation from JSON.

Examples::

    python scripts/assemble_evaluation.py --input assessment.json --output result.json
    cat assessment.json | python scripts/assemble_evaluation.py --input - --output -

This command is fully offline.  It does not call Hy3, Crossref, NCBI or any
other network service; those upstream tools must materialize their explicit
records before this assembly step.
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
    # Same-directory temporary file keeps os.replace atomic on one filesystem.
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
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
        description="离线汇总九维评估输入、致命错误审计与发布决策"
    )
    parser.add_argument("--input", default="-", help="EvaluationAssemblyInput JSON；默认 stdin")
    parser.add_argument("--output", default="-", help="结果 JSON；- 表示 stdout")
    parser.add_argument("--rubric", type=Path, help="可选量表 YAML；默认 configs/rubric_v0_1.yaml")
    parser.add_argument(
        "--print-input-schema",
        action="store_true",
        help="输出严格输入 JSON Schema 后退出，不读取 --input",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from evaluator.assembly import EvaluationAssemblyInput, assemble_evaluation
    from evaluator.rubric import load_rubric

    args = build_parser().parse_args(argv)
    if args.print_input_schema:
        from evaluator.assembly import EvaluationAssemblyInput

        rendered = json.dumps(
            EvaluationAssemblyInput.model_json_schema(), ensure_ascii=False, indent=2
        )
        _write_json(args.output, rendered)
        return 0
    try:
        payload = json.loads(_read_json(args.input))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"输入 JSON 读取失败：{exc}") from exc

    try:
        data = EvaluationAssemblyInput.model_validate(payload)
        config = load_rubric(args.rubric) if args.rubric else None
        result = assemble_evaluation(data, config=config)
    except (ValidationError, ValueError) as exc:
        raise SystemExit(f"评估输入不合规：{exc}") from exc

    rendered = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
    _write_json(args.output, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
