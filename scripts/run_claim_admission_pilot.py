#!/usr/bin/env python3
"""Run the real-Hy3 blinded Claim-admission concordance Pilot.

There is deliberately no offline execution mode.  Unit tests inject a fake
model; this CLI requires a configured real Hy3 endpoint.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="真实 Hy3 Claim 准入四分类盲法 Pilot（系统-单一专家参考一致性）"
    )
    parser.add_argument("--limit", type=int, default=50, help="哈希固定样本数，1..50")
    parser.add_argument("--repeats", type=int, default=1, help="每条 Claim 的采样次数")
    parser.add_argument(
        "--selection-seed",
        default="mitoevidence-claim-admission-selection-v1",
        help="决定 limit<50 时的固定子集；产物只保存其 SHA-256",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "claim_admission_pilots",
    )
    parser.add_argument("--suite-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="严格核验原配置、冻结输入和已落盘 cell 后继续缺失网格",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.claim_admission_pilot import (
        CellOutcome,
        ClaimAdmissionPilotRunner,
        Hy3ClaimAdmissionModel,
    )
    from evaluator.judge import Hy3Transport
    from evaluator.judge.config import default_judge_config

    args = build_parser().parse_args(argv)
    if not 1 <= args.limit <= 50:
        raise SystemExit("--limit 必须在 1..50")
    if args.repeats <= 0:
        raise SystemExit("--repeats 必须为正整数")
    if not 0 <= args.temperature <= 2:
        raise SystemExit("--temperature 必须在 0..2")
    if args.resume and not args.suite_id:
        raise SystemExit("--resume 必须同时提供原 --suite-id")

    config = default_judge_config()
    if not config.resolve_api_key():
        raise SystemExit("缺少 HY3_API_KEY；该 CLI 不提供离线伪实验模式")
    transport = Hy3Transport(
        api_key=config.resolve_api_key(),
        base_url=config.resolve_base_url(),
        max_rps=float(config.transport["max_rps"]),
        max_retries=int(config.transport["max_retries"]),
        timeout=float(config.transport["timeout_s"]),
        trust_env=bool(config.transport["trust_env"]),
    )
    model = Hy3ClaimAdmissionModel(config=config, transport=transport)
    runner = ClaimAdmissionPilotRunner(model=model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = args.suite_id or f"claim-admission-hy3-{stamp}"
    suite_dir, state = runner.run_suite(
        repo_root=REPO_ROOT,
        out_root=args.out_root,
        suite_id=suite_id,
        limit=args.limit,
        repeats=args.repeats,
        selection_seed=args.selection_seed,
        temperature=args.temperature,
        base_seed=args.base_seed,
        resume=args.resume,
    )
    succeeded = sum(row.outcome is CellOutcome.SUCCEEDED for row in state.records)
    failed = len(state.records) - succeeded
    print(f"套件：{suite_dir}")
    print(f"网格：{len(state.records)}/{state.expected_calls}；成功 {succeeded}；失败 {failed}")
    print("盲法：模型仅见 triple/evidence/conditions 与 paper/section/source metadata")
    print("金标：manifest 指定的历史 ai_decision；未向模型暴露")
    print("解释边界：系统-单一专家共识参考一致性，不是专家间一致性")
    print("下一步：运行 scripts/analyze_claim_admission_pilot.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
