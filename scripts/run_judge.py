#!/usr/bin/env python3
"""L2 语义 Judge CLI：claims+evidence JSONL → 逐主张判定 + 升级队列 + 成本汇总。

用法：
  # 密钥只从环境或密钥管理器读取（不落盘）；变量名见 .env.example。
  python scripts/run_judge.py --input eval/data/claims.jsonl \\
      --out results/judge/aggregates.jsonl \\
      --escalations results/judge/escalations.jsonl

输入 JSONL：每行一个待判定单元（Judge 只看到问题、一个原子主张与候选证据，
看不到系统名与其他主张——方案 8.4）：
  {"question": "……（可选）",
   "claim": {"claim_id": "C1", "text": "……", "is_core": true,
             "conditions": {"species": "rat", "effect_direction": "increase"},
             "citations": []},
   "evidence_spans": [{"span_id": "S1", "paper_id": "P1", "doi_or_pmid": "10.xxxx/yyyy",
                       "section": "Results", "source_access": "fulltext",
                       "anchor": {"prefix": "……", "exact": "……", "postfix": "……"}}]}

输出：
  --out           逐主张 JudgeAggregate（JSONL）；
  --escalations   agreement 低于阈值/并列/refuted 的主张进入升级队列（JSONL）；
  stdout          逐主张一行摘要 + tokens 成本汇总（含 Prompt Cache 命中率与思考开销）。

网络：默认 trust_env=False 直连（本机代理可能已失效）；确需走代理加 --trust-env。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_input_file(path: Path) -> list[dict]:
    entries: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except ValueError as exc:
            raise SystemExit(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
        if "claim" not in record:
            raise SystemExit(f"{path}:{lineno} 缺少 claim 字段")
        entries.append(record)
    return entries


def main() -> int:
    # 延迟导入：sys.path 已在模块顶部补好仓库根目录。
    from pydantic import ValidationError

    from evaluator.judge import Hy3Client, load_judge_config, run_self_consistency
    from evaluator.schemas import AtomicClaim, EvidenceSpan

    ap = argparse.ArgumentParser(description="L2 语义 Judge：自一致性判定")
    ap.add_argument("--input", type=Path, required=True, help="claims+evidence JSONL")
    ap.add_argument("--out", type=Path, help="逐主张 JudgeAggregate 输出（JSONL）")
    ap.add_argument(
        "--escalations", type=Path, default=Path("results/judge/escalations.jsonl"),
        help="升级队列输出（JSONL）",
    )
    ap.add_argument("--config", type=Path, help="Judge 配置路径（默认 configs/judge_v0_1.yaml）")
    ap.add_argument("--k", type=int, help="覆盖自一致性采样次数（默认取配置）")
    ap.add_argument("--temperature", type=float, help="覆盖采样温度（默认取配置）")
    ap.add_argument("--base-seed", type=int, help="逐样本 seed = base_seed + 样本序号")
    ap.add_argument("--limit", type=int, help="只处理前 N 条（冒烟用）")
    ap.add_argument(
        "--trust-env", action="store_true", help="使用 http_proxy/https_proxy（默认忽略，直连）"
    )
    args = ap.parse_args()

    config = load_judge_config(args.config) if args.config else load_judge_config()
    if not config.resolve_api_key():
        raise SystemExit(f"缺少 API Key：export {config.model['api_key_env']}=...（密钥不落盘）")

    import requests

    session = requests.Session()
    session.trust_env = args.trust_env
    client = Hy3Client(config=config, session=session)

    entries = parse_input_file(args.input)
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("输入为空")

    aggregates = []
    for lineno, entry in enumerate(entries, start=1):
        try:
            claim = AtomicClaim.model_validate(entry["claim"])
            spans = [EvidenceSpan.model_validate(s) for s in entry.get("evidence_spans") or []]
        except ValidationError as exc:
            raise SystemExit(f"第 {lineno} 条输入不合规：{exc}") from exc
        aggregate = run_self_consistency(
            client,
            claim,
            spans,
            question=str(entry.get("question") or ""),
            k=args.k,
            temperature=args.temperature,
            base_seed=args.base_seed,
            config=config,
        )
        aggregates.append(aggregate)
        flag = " → 升级人工" if aggregate.escalate_to_human else ""
        print(
            f"[{lineno}/{len(entries)}] {aggregate.claim_id}: "
            f"{aggregate.final_verdict.value}（{aggregate.votes}，"
            f"agreement {aggregate.agreement_rate:.2f}，有效 {aggregate.n_valid}/{aggregate.k}）{flag}",
            flush=True,
        )

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for aggregate in aggregates:
                fh.write(json.dumps(aggregate.model_dump(mode="json"), ensure_ascii=False) + "\n")

    escalated = [a for a in aggregates if a.escalate_to_human]
    if escalated:
        args.escalations.parent.mkdir(parents=True, exist_ok=True)
        with args.escalations.open("a", encoding="utf-8") as fh:
            for aggregate in escalated:
                fh.write(
                    json.dumps(
                        {
                            "timestamp_utc": timestamp,
                            "claim_id": aggregate.claim_id,
                            "final_verdict": aggregate.final_verdict.value,
                            "agreement_rate": aggregate.agreement_rate,
                            "votes": aggregate.votes,
                            "reasons": aggregate.escalate_reasons,
                            "sample_verdicts": [
                                s.verdict.verdict.value if s.verdict else f"failed: {s.error}"
                                for s in aggregate.samples
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # 成本汇总
    from evaluator.judge import TokenUsage

    total = TokenUsage()
    for aggregate in aggregates:
        total = total.merged(aggregate.usage_total)
    hit_rate = total.cache_hit_rate
    print(
        f"\n共 {len(aggregates)} 个主张，{len(escalated)} 个进入升级队列"
        f"（{args.escalations if escalated else '无写入'}）"
    )
    print(
        f"成本：requests={total.n_requests}  prompt={total.prompt_tokens}"
        f"（cached={total.cached_tokens}，命中率 {'NA' if hit_rate is None else f'{hit_rate:.0%}'}）"
        f"  completion={total.completion_tokens}（其中思考 {total.reasoning_tokens}）"
    )
    print(
        f"judge_config v{config.version} sha256={config.sha256[:12]}…  "
        f"channel={client.channel}  model={client.model}"
    )
    if args.out:
        print(f"JSONL 报告：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
