#!/usr/bin/env python3
"""可复现抽样脚本：从 512 条结构可引用 Claim 候选中抽取 50 条送 AI 预审。

本脚本只读取调用方通过 ``--kg-edges`` 指定的 `kg_edges.jsonl`，不写回任何仓库
文件。抽样为确定性系统抽样（无随机数种子依赖），相同输入可得到同一 50 条。

用法：
    python3 _extract_sample.py --kg-edges /path/to/kg_edges.jsonl
    python3 _extract_sample.py --kg-edges /path/to/kg_edges.jsonl --dump
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SAMPLE_N = 50

# 与 agent/tools/knowledge_graph.py::_is_citable_claim 逐字段一致（2026-08-28 核对）。
# 复制而非 import，是为了让本抽样在仓库重构后仍可独立复现。
def is_citable_claim(edge: dict, min_confidence: float = 0.6) -> bool:
    return bool(
        edge.get("evidence_text")
        and edge.get("evidence_match")
        and not edge.get("context_only")
        and edge.get("status", "asserted") == "asserted"
        and edge.get("review_status") != "legacy_unverified"
        and edge.get("schema_valid") is True
        and edge.get("relation") != "other"
        and float(edge.get("confidence") or 0.0) >= float(min_confidence)
    )


def load_citable(path: Path) -> list[dict]:
    edges = [json.loads(line) for line in path.open() if line.strip()]
    return [e for e in edges if is_citable_claim(e)]


def systematic_sample(citable: list[dict], n: int = SAMPLE_N) -> list[dict]:
    """按 paperId 分层 + 层内 statement_id 排序后等距取样（确定性）。"""
    by_paper: dict[str, list[dict]] = {}
    for edge in citable:
        by_paper.setdefault(str(edge.get("paperId") or ""), []).append(edge)

    # 比例分配，余数按层大小降序补齐 —— 全程无随机性
    total = len(citable)
    order = sorted(by_paper, key=lambda p: (-len(by_paper[p]), p))
    quota = {p: len(by_paper[p]) * n // total for p in order}
    for p in order:
        if sum(quota.values()) >= n:
            break
        quota[p] += 1

    picked: list[dict] = []
    for paper in order:
        rows = sorted(by_paper[paper], key=lambda e: str(e.get("statement_id") or ""))
        k = quota[paper]
        if k <= 0:
            continue
        step = len(rows) / k
        picked.extend(rows[min(len(rows) - 1, int(i * step))] for i in range(k))
    return sorted(picked, key=lambda e: (str(e.get("paperId")), str(e.get("statement_id"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-edges", type=Path, required=True, help="原始 kg_edges.jsonl")
    ap.add_argument("--dump", action="store_true", help="输出 50 条完整 JSON")
    args = ap.parse_args()

    citable = load_citable(args.kg_edges)
    sample = systematic_sample(citable)
    digest = hashlib.sha256(args.kg_edges.read_bytes()).hexdigest()

    print(f"kg_edges.jsonl sha256 = {digest}")
    print(f"结构可引用 Claim 候选 = {len(citable)}    抽样 = {len(sample)}")
    print("分层分布 =", dict(Counter(e["paperId"][:8] for e in sample)))
    if args.dump:
        for edge in sample:
            print(json.dumps(edge, ensure_ascii=False))
    else:
        for edge in sample:
            print(
                f"{edge['statement_id']}  {edge['paperId'][:8]}  {edge.get('section','')[:12]:12s} "
                f"{edge['head_name']} --{edge['relation']}--> {edge['tail_name']}"
            )


if __name__ == "__main__":
    main()
