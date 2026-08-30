#!/usr/bin/env python3
"""运行MitoEvidence-Hy3最小应用闭环。

默认调用Hy3；``--offline-smoke``只验证编排与文件契约，输出会醒目标注为非模型结果。
当前冻结语料是12篇综述中可合法取得XML的7篇，不能冒充完整系统综述。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from app.corpus import FrozenReviewCorpus
    from app.hy3_review import Hy3ReviewModel
    from app.offline import OfflineSmokeModel
    from app.pipeline import ReviewRunner, load_pilot_request
    from app.schemas import ReviewRequest, RunKind
    from evaluator.judge.config import default_judge_config

    ap = argparse.ArgumentParser(description="MitoEvidence-Hy3：可追溯快速证据综述")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pilot-id", help="运行专家金标Pilot（PILOT-01…PILOT-05）")
    source.add_argument("--question", help="自定义研究问题")
    ap.add_argument(
        "--pilot-file",
        type=Path,
        default=REPO_ROOT / "annotation_prelabel" / "pilot_questions" / "pilot_5_questions.jsonl",
    )
    ap.add_argument("--question-id", default="USER-QUESTION")
    ap.add_argument("--scope", default="")
    ap.add_argument("--source-pmid", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "review_runs")
    ap.add_argument("--run-id", help="固定run目录名（回归测试用）")
    ap.add_argument(
        "--offline-smoke",
        action="store_true",
        help="不调用Hy3；只验证编排，结果不得用作模型/科学性能",
    )
    args = ap.parse_args()

    if args.pilot_id:
        request = load_pilot_request(args.pilot_file, args.pilot_id)
    else:
        request = ReviewRequest(
            question_id=args.question_id,
            question=args.question,
            scope=args.scope,
            source_pmids=args.source_pmid,
        )

    corpus = FrozenReviewCorpus(REPO_ROOT)
    if args.offline_smoke:
        model = OfflineSmokeModel()
        kind = RunKind.OFFLINE_SMOKE
    else:
        cfg = default_judge_config()
        if not cfg.resolve_api_key():
            raise SystemExit(
                "缺少HY3_API_KEY；请通过环境变量提供；"
                "若只验工程链路可加 --offline-smoke。"
            )
        model = Hy3ReviewModel(config=cfg)
        kind = RunKind.HY3
    runner = ReviewRunner(model=model, corpus=corpus, run_kind=kind)
    artifact = runner.run(request, top_k=args.top_k)
    output = runner.write_run(artifact, out_root=args.out_root, run_id=args.run_id)
    print(f"完成：{output}")
    print(f"状态：{artifact.formal_status}")
    print(f"召回段落：{len(artifact.passages)}；原子主张：{len(artifact.review.claims)}")
    print(f"下一步Judge输入：{output / 'judge_input.jsonl'}")
    if artifact.warnings:
        print("警告：")
        for warning in artifact.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
