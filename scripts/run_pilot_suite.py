#!/usr/bin/env python3
"""依次运行五道 Pilot，并生成套件级审计摘要。

默认调用 Hy3。``--offline-smoke`` 只验证五题编排与安全降级，不是模型效果实验，
摘要中的 ``formal_status`` 会持续保留这一边界。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json_atomic(path: Path, value: object) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main() -> int:
    from app.corpus import FrozenReviewCorpus
    from app.hy3_review import Hy3ReviewModel
    from app.offline import OfflineSmokeModel
    from app.pipeline import ReviewRunner, load_pilot_request
    from app.schemas import RunKind
    from evaluator.judge.config import default_judge_config

    ap = argparse.ArgumentParser(description="运行 PILOT-01…05 的可追溯综述套件")
    ap.add_argument(
        "--pilot-file",
        type=Path,
        default=REPO_ROOT / "annotation_prelabel" / "pilot_questions" / "pilot_5_questions.jsonl",
    )
    ap.add_argument("--pilot-id", action="append", help="只运行指定题；可重复")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "pilot_suites")
    ap.add_argument("--suite-id", help="固定套件目录名；默认使用UTC时间")
    ap.add_argument("--offline-smoke", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    pilot_ids = args.pilot_id or [f"PILOT-{index:02d}" for index in range(1, 6)]
    if len(set(pilot_ids)) != len(pilot_ids):
        raise SystemExit("--pilot-id 不得重复")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = args.suite_id or f"pilot-suite-{stamp}"
    suite_dir = (args.out_root / suite_id).resolve()
    if suite_dir.exists():
        raise SystemExit(f"套件目录已存在：{suite_dir}")
    suite_dir.mkdir(parents=True)

    corpus = FrozenReviewCorpus(REPO_ROOT)
    if args.offline_smoke:
        model = OfflineSmokeModel()
        run_kind = RunKind.OFFLINE_SMOKE
    else:
        cfg = default_judge_config()
        if not cfg.resolve_api_key():
            raise SystemExit(
                "缺少 HY3_API_KEY。旧 Key 已暴露，必须先在腾讯控制台轮换；"
                "仅验证工程可加 --offline-smoke。"
            )
        model = Hy3ReviewModel(config=cfg)
        run_kind = RunKind.HY3
    runner = ReviewRunner(model=model, corpus=corpus, run_kind=run_kind)

    records: list[dict] = []
    for pilot_id in pilot_ids:
        try:
            request = load_pilot_request(args.pilot_file, pilot_id)
            artifact = runner.run(request, top_k=args.top_k)
            run_dir = runner.write_run(
                artifact,
                out_root=suite_dir,
                run_id=pilot_id,
            )
            records.append(
                {
                    "pilot_id": pilot_id,
                    "ok": True,
                    "run_dir": run_dir.name,
                    "formal_status": artifact.formal_status,
                    "answerability": artifact.review.answerability.value,
                    "passage_count": len(artifact.passages),
                    "claim_count": len(artifact.review.claims),
                    "warning_count": len(artifact.warnings),
                }
            )
        except Exception as exc:  # noqa: BLE001 - 套件摘要必须保留单题失败
            records.append(
                {
                    "pilot_id": pilot_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                break

    summary = {
        "suite_id": suite_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_kind": run_kind.value,
        "formal_status": (
            "offline_engineering_smoke_not_model_result"
            if run_kind is RunKind.OFFLINE_SMOKE
            else "hy3_engineering_suite_pending_human_gold"
        ),
        # 审计包不应泄露参赛机器上的绝对目录。仓库内输入记录相对路径；
        # 仓库外自定义输入只记录文件名，并始终以 SHA-256 固定实际内容。
        "pilot_file": (
            str(args.pilot_file.resolve().relative_to(REPO_ROOT))
            if args.pilot_file.resolve().is_relative_to(REPO_ROOT)
            else args.pilot_file.name
        ),
        "pilot_file_sha256": hashlib.sha256(args.pilot_file.read_bytes()).hexdigest(),
        "requested": pilot_ids,
        "completed": sum(1 for item in records if item["ok"]),
        "failed": sum(1 for item in records if not item["ok"]),
        "records": records,
    }
    _write_json_atomic(suite_dir / "suite_summary.json", summary)
    print(f"套件：{suite_dir}")
    print(f"状态：{summary['formal_status']}")
    print(f"完成 {summary['completed']}/{len(pilot_ids)}；失败 {summary['failed']}")
    for record in records:
        detail = (
            f"passages={record['passage_count']} claims={record['claim_count']}"
            if record["ok"]
            else f"{record['error_type']}: {record['error']}"
        )
        print(f"- {record['pilot_id']}: {'OK' if record['ok'] else 'FAILED'} · {detail}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
