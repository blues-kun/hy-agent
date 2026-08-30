#!/usr/bin/env python3
"""Run the real-Hy3 A/B/C/D Pilot generation suite.

There is intentionally no offline-smoke flag: deterministic test doubles are
available to tests, but a CLI artifact named as an ablation must use Hy3.
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
        description="真实 Hy3 Pilot A/B/C/D：direct / sparse TF-IDF / frozen graph / Judge gate"
    )
    parser.add_argument(
        "--pilot-file",
        type=Path,
        default=REPO_ROOT / "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl",
    )
    parser.add_argument("--pilot-id", action="append", help="只运行指定 Pilot；可重复")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--judge-k",
        type=int,
        default=1,
        help="D 每条 claim 的 Judge 采样数；Pilot 默认 1，正式自一致性可设 7",
    )
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-base-seed", type=int)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_ablations",
    )
    parser.add_argument("--suite-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.ablation import Hy3ClaimGate, PilotAblationRunner
    from app.corpus import FrozenReviewCorpus
    from app.hy3_review import Hy3ReviewModel
    from app.pipeline import load_pilot_request
    from evaluator.judge import Hy3Client, Hy3Transport
    from evaluator.judge.config import default_judge_config

    args = build_parser().parse_args(argv)
    if args.replicates <= 0 or args.top_k <= 0 or args.judge_k <= 0:
        raise SystemExit("--replicates、--top-k、--judge-k 必须为正整数")
    ids = args.pilot_id or [f"PILOT-{index:02d}" for index in range(1, 6)]
    if len(ids) != len(set(ids)):
        raise SystemExit("--pilot-id 不得重复")
    requests = [load_pilot_request(args.pilot_file, pilot_id) for pilot_id in ids]

    config = default_judge_config()
    if not config.resolve_api_key():
        raise SystemExit("缺少 HY3_API_KEY；该 CLI 不提供离线伪实验模式")
    # Generation and Judge share one transport instance so the model-level
    # 60 RPM throttle applies across both call types, not separately per client.
    transport = Hy3Transport(
        api_key=config.resolve_api_key(),
        base_url=config.resolve_base_url(),
        max_rps=float(config.transport["max_rps"]),
        max_retries=int(config.transport["max_retries"]),
        timeout=float(config.transport["timeout_s"]),
        trust_env=bool(config.transport["trust_env"]),
    )
    model = Hy3ReviewModel(config=config, transport=transport)
    judge_client = Hy3Client(config=config, transport=transport)
    gate = Hy3ClaimGate(
        judge_client,
        config=config,
        k=args.judge_k,
        temperature=args.judge_temperature,
        base_seed=args.judge_base_seed,
    )
    runner = PilotAblationRunner(
        model=model,
        corpus=FrozenReviewCorpus(REPO_ROOT),
        claim_gate=gate,
        top_k=args.top_k,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = args.suite_id or f"pilot-abcd-hy3-{stamp}"
    suite_dir, state = runner.run_suite(
        requests,
        replicates=args.replicates,
        out_root=args.out_root,
        suite_id=suite_id,
        input_path=args.pilot_file,
    )
    succeeded = sum(record.outcome.value == "succeeded" for record in state.records)
    failed = len(state.records) - succeeded
    print(f"套件：{suite_dir}")
    print(f"网格：{len(state.records)}/{state.expected_grid_cells}；成功 {succeeded}；失败 {failed}")
    print("B 定义：稀疏 TF-IDF full-text vector（不是 dense embedding RAG）")
    print("C 定义：冻结文本/元数据 evidence graph（不读取专家标签）")
    print(f"D 定义：同一 C 草稿的 Hy3 Judge gate，k={state.judge_k}，不追加检索")
    print("状态：pilot_ablation_generation_unscored；仍需用专家金标完成评分")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
