#!/usr/bin/env python3
"""Bind real Hy3 Pilot answerability to the pinned expert reference."""
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


def _write(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    from evaluator.experiment_protocol import build_pilot_answerability_concordance

    parser = argparse.ArgumentParser(
        description="从真实 Hy3 五题套件构造 answerability—专家参考配对（缺失运行不删除）"
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_pilot_answerability_concordance(REPO_ROOT, args.suite_dir)
    except ValueError as exc:
        raise SystemExit(f"无法构造 Pilot 专家 concordance：{exc}") from exc
    _write(
        args.output,
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    missing = sum(row.automatic_label is None for row in result.nominal)
    print(f"完成：{args.output}；配对 {len(result.nominal) - missing}/{len(result.nominal)}，缺失 {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
