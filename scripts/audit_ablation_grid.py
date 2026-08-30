#!/usr/bin/env python3
"""Audit A/B/C/D run-grid completeness without dropping failures."""
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


def _write(path: str, rendered: str) -> None:
    if path == "-":
        print(rendered)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from evaluator.experiment_protocol import AblationInput, ablation_grid_audit

    parser = argparse.ArgumentParser(
        description="审计 A/B/C/D × question × replicate 网格；失败记录保留在分母"
    )
    parser.add_argument("--input", default="-")
    parser.add_argument("--output", default="-")
    parser.add_argument("--print-input-schema", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="存在缺失网格单元时返回 2；显式 outcome=failed 不算缺失",
    )
    args = parser.parse_args(argv)
    if args.print_input_schema:
        _write(
            args.output,
            json.dumps(AblationInput.model_json_schema(), ensure_ascii=False, indent=2),
        )
        return 0
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        data = AblationInput.model_validate_json(raw)
    except (OSError, ValidationError, ValueError) as exc:
        raise SystemExit(f"A/B/C/D 网格输入不合规：{exc}") from exc
    result = ablation_grid_audit(data)
    _write(args.output, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if args.require_complete and not result["grid_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
