#!/usr/bin/env python3
"""Run the real-Hy3 A/B/C/D Pilot generation suite.

There is intentionally no offline-smoke flag: deterministic test doubles are
available to tests, but a CLI artifact named as an ablation must use Hy3.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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
    parser.add_argument(
        "--judge-base-seed",
        type=int,
        help=(
            "formal v3 必填；先按 question/replicate/claim 哈希派生，"
            "再以 derived_base_seed + sample_index 生成 k 个 sample seed"
        ),
    )
    parser.add_argument(
        "--generator-base-seed",
        type=int,
        default=20260831,
        help="v3 A/B/C generation 的固定派生种子根；逐 question/replicate/arm/stage 哈希派生",
    )
    parser.add_argument(
        "--generator-cache-namespace",
        default="mitoevidence-ablation-v3",
        help="v3 generator cache/session namespace 根",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_ablations",
    )
    parser.add_argument("--suite-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "安全恢复开关（当前 fail-closed 拒绝；不会把异模型/异配置结果混入旧 suite）"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.ablation import Hy3ClaimGate, PilotAblationRunner
    from app.corpus import FrozenReviewCorpus
    from app.hy3_review import Hy3ReviewModel
    from app.pipeline import load_pilot_request
    from evaluator.judge import Hy3Client, Hy3Transport
    from evaluator.judge.config import default_judge_config
    from evaluator.expert_gold import audit_expert_gold
    from evaluator.pilot_identity import is_formal_hy3_metadata

    args = build_parser().parse_args(argv)
    if args.replicates <= 0 or args.top_k <= 0 or args.judge_k <= 0:
        raise SystemExit("--replicates、--top-k、--judge-k 必须为正整数")
    if args.resume:
        raise SystemExit(
            "拒绝 --resume：严格恢复尚未实现对已有 cell 的逐文件审计、孤儿 cell "
            "处理与原子续写；请使用新的 --suite-id 重跑"
        )
    if args.judge_base_seed is None:
        raise SystemExit(
            "formal v3 要求显式 --judge-base-seed；无 seed 的 Judge 运行只能是 nonformal"
        )
    gold_audit = audit_expert_gold(
        REPO_ROOT / "annotation_prelabel/expert_gold_manifest.json",
        repo_root=REPO_ROOT,
    )
    if not gold_audit.get("ok"):
        raise SystemExit(
            "formal v3 专家金标审计失败："
            + "；".join(str(item) for item in gold_audit.get("errors") or [])
        )
    expected_input_sha256 = gold_audit["datasets"]["pilot_questions"]["sha256"]
    try:
        actual_input_sha256 = hashlib.sha256(args.pilot_file.read_bytes()).hexdigest()
    except OSError as exc:
        raise SystemExit(f"无法读取 --pilot-file：{exc}") from exc
    if actual_input_sha256 != expected_input_sha256:
        raise SystemExit(
            "formal v3 --pilot-file 必须精确绑定 expert manifest 的 pilot_questions hash；"
            "自定义/投影输入只能用于独立 nonformal runner"
        )
    ids = args.pilot_id or [f"PILOT-{index:02d}" for index in range(1, 6)]
    if len(ids) != len(set(ids)):
        raise SystemExit("--pilot-id 不得重复")
    requests = [load_pilot_request(args.pilot_file, pilot_id) for pilot_id in ids]

    config = default_judge_config()
    resolved_base_url = config.resolve_base_url().rstrip("/")
    resolved_endpoint = urlsplit(f"{resolved_base_url}/chat/completions")
    shared_hy3_identity = {
        "execution_kind": "remote_hy3",
        "provider": "tencent-tokenhub",
        "model": config.resolve_model(),
        "endpoint_origin": (
            f"{resolved_endpoint.scheme}://{resolved_endpoint.netloc}"
        ),
        "endpoint_url": f"{resolved_base_url}/chat/completions",
    }
    if not is_formal_hy3_metadata(shared_hy3_identity):
        raise SystemExit(
            "formal v3 Generator/Judge 身份未通过共享 allowlist："
            "model 必须精确为 hy3，endpoint 必须是腾讯 TokenHub 官方 HTTPS 主机"
        )
    if not config.resolve_api_key():
        raise SystemExit("缺少 HY3_API_KEY；该 CLI 不提供离线伪实验模式")
    # Generation and Judge share one transport instance so the model-level
    # 60 RPM throttle applies across both call types, not separately per client.
    transport = Hy3Transport(
        api_key=config.resolve_api_key(),
        base_url=resolved_base_url,
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
        generator_base_seed=args.generator_base_seed,
        generator_cache_namespace=args.generator_cache_namespace,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = args.suite_id or f"pilot-abcd-hy3-{stamp}"
    suite_dir, state = runner.run_suite(
        requests,
        replicates=args.replicates,
        out_root=args.out_root,
        suite_id=suite_id,
        input_path=args.pilot_file,
        resume=False,
    )
    succeeded = sum(record.outcome.value == "succeeded" for record in state.records)
    failed = len(state.records) - succeeded
    print(f"套件：{suite_dir}")
    print(f"网格：{len(state.records)}/{state.expected_grid_cells}；成功 {succeeded}；失败 {failed}")
    print("B 定义：稀疏 TF-IDF full-text vector（不是 dense embedding RAG）")
    print("C 定义：冻结文本/元数据 evidence graph（不读取专家标签）")
    print(f"D 定义：同一 C 草稿的 Hy3 Judge gate，k={state.judge_k}，不追加检索")
    print("状态：pilot_ablation_generation_unscored；仍需用专家金标完成评分")
    print(
        "Generator v3 provenance："
        f"base_seed={state.generator_provenance.base_seed}；"
        f"namespace={state.generator_provenance.cache_namespace}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
