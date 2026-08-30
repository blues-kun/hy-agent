#!/usr/bin/env python3
"""Run the real-Hy3 blinded terminology wrong/correct pair Pilot.

There is intentionally no offline mode.  Tests inject a fake PairModel, while
artifacts produced by this CLI require a real HY3_API_KEY.
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
        description=(
            "真实 Hy3 术语/条件错误盲法成对判别 Pilot；不是全文证据一致性实验"
        )
    )
    parser.add_argument("--limit", type=int, default=60, help="哈希固定抽样上限，1..60")
    parser.add_argument("--repeats", type=int, default=3, help="每对独立采样次数")
    parser.add_argument(
        "--selection-seed",
        default="mitoevidence-terminology-selection-v1",
        help="决定样本子集；只在产物中保存其 SHA-256",
    )
    parser.add_argument(
        "--order-seed",
        default="mitoevidence-terminology-left-right-v1",
        help="决定 correct/wrong 的哈希固定左右顺序；只保存 SHA-256",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "terminology_pair_pilots",
    )
    parser.add_argument("--suite-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="严格核验原配置、输入和已落盘 cell 后继续缺失网格",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.terminology_pair_pilot import (
        CellOutcome,
        Hy3TerminologyPairModel,
        TerminologyPairPilotRunner,
    )
    from evaluator.judge import Hy3Transport
    from evaluator.judge.config import default_judge_config

    args = build_parser().parse_args(argv)
    if not 1 <= args.limit <= 60:
        raise SystemExit("--limit 必须在 1..60")
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
    model = Hy3TerminologyPairModel(config=config, transport=transport)
    runner = TerminologyPairPilotRunner(model=model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = args.suite_id or f"terminology-pair-hy3-{stamp}"
    suite_dir, state = runner.run_suite(
        repo_root=REPO_ROOT,
        out_root=args.out_root,
        suite_id=suite_id,
        limit=args.limit,
        repeats=args.repeats,
        selection_seed=args.selection_seed,
        order_seed=args.order_seed,
        temperature=args.temperature,
        base_seed=args.base_seed,
        resume=args.resume,
    )
    succeeded = sum(record.outcome is CellOutcome.SUCCEEDED for record in state.records)
    failed = len(state.records) - succeeded
    print(f"套件：{suite_dir}")
    print(f"网格：{len(state.records)}/{state.expected_calls}；成功 {succeeded}；失败 {failed}")
    print("盲法输入：模型仅见 left_text/right_text；左右顺序由哈希固定")
    print("标签来源：专家金标 wrong/correct 字段角色；不存在独立 approve/reject 列")
    print("边界：术语/因果强度/条件错误成对判别，不是全文 Claim-Evidence 一致性")
    print("请运行 scripts/analyze_terminology_pair_pilot.py 计算预注册分母上的指标")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
