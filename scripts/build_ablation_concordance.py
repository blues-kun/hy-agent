#!/usr/bin/env python3
"""Bind A/B/C/D Pilot answerability outputs to the expert gold snapshot."""
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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    from evaluator.experiment_protocol import build_ablation_answerability_concordance

    parser = argparse.ArgumentParser(
        description="将 A/B/C/D Pilot answerability 与专家金标配对；失败 cell 不删除"
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help=(
            "显式允许 fixture/legacy/未完成 suite 的诊断性配对；输出不会标为正式 Hy3 concordance"
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = build_ablation_answerability_concordance(
            REPO_ROOT,
            args.suite_dir,
            allow_nonformal=args.allow_nonformal,
        )
    except ValueError as exc:
        raise SystemExit(f"无法构造 A/B/C/D 专家 concordance：{exc}") from exc
    _write(
        args.output,
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    missing = sum(row.automatic_label is None for row in result.nominal)
    print(
        f"完成：{args.output}；完整配对 {len(result.nominal) - missing}/"
        f"{len(result.nominal)}，失败/缺失 {missing}"
    )
    print("该结果只验证 Pilot answerability，不替代 60 份九维专家评分一致性实验。")
    if args.allow_nonformal:
        print("警告：--allow-nonformal 已启用；该输出仅供诊断，不是 production concordance。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
