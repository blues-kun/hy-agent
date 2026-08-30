#!/usr/bin/env python3
"""引用核验 CLI：读入 DOI/PMID 列表 → 批量核验 → 输出 JSON 报告。

用法：
  # 直接给标识符
  python scripts/verify_citations.py --doi 10.1038/s41746-025-02005-2 --doi 10.2427/12267
  python scripts/verify_citations.py --pmid 25659350 --pmid 42051080

  # 从文件读（每行一个标识符，# 开头为注释；也支持每行一个 JSON 对象携带期望元数据）
  python scripts/verify_citations.py --input citations.txt --out results/d1_report.json

  # 标注哪些标识符支撑核心结论（触发 D1 的「核心结论使用错误引用」事件上限）
  python scripts/verify_citations.py --input c.txt --core-id 10.1038/s41746-025-02005-2

JSONL 输入格式（期望元数据参与「明确冲突」判定；first_author 只比对第一作者姓氏）：
  {"id": "10.1038/s41746-025-02005-2", "title": "Evaluating clinical AI …", "year": 2025,
   "first_author": "Croxford E", "is_core": true}

环境变量（见 .env.example）：
  CROSSREF_MAILTO   进 Crossref polite pool
  NCBI_API_KEY      有 key 时 10 rps，无 key 3 rps
  NCBI_TOOL         E-utilities 必填的 tool 参数
  NCBI_EMAIL        E-utilities 必填的 email 参数

网络说明：默认 session.trust_env=False，忽略 http_proxy/https_proxy 直连
（本机代理可能已失效）。确需走代理时加 --trust-env。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # 允许以 `python scripts/verify_citations.py` 直接运行（此时 sys.path[0] 是 scripts/）。
    sys.path.insert(0, str(REPO_ROOT))


def parse_input_file(path: Path) -> list[dict]:
    """返回 [{"id":…, "title":…, "year":…, "is_core":…}, …]。"""
    entries: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            try:
                record = json.loads(text)
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
            if "id" not in record:
                raise SystemExit(f"{path}:{lineno} 缺少 id 字段")
            entries.append(record)
        else:
            entries.append({"id": text})
    return entries


def build_session(trust_env: bool):
    import requests

    session = requests.Session()
    session.trust_env = trust_env
    return session


def main() -> int:
    # 延迟到入口点再导入：sys.path 已在模块顶部补好仓库根目录。
    from evaluator.rubric import (
        EVALUATOR_VERSION,
        DimensionInput,
        default_rubric,
        score_dimension,
    )
    from evaluator.rules.identifier_check import (
        CROSSREF_MAX_BATCH,
        NCBI_MAX_BATCH,
        CrossrefClient,
        ExpectedMetadata,
        NcbiESummaryClient,
        normalize_identifier,
        summarize_verifications,
    )
    from evaluator.schemas import VerificationStatus

    ap = argparse.ArgumentParser(description="批量核验 DOI/PMID 并输出 D1 报告")
    ap.add_argument("--doi", action="append", default=[], help="可重复；直接给出 DOI")
    ap.add_argument("--pmid", action="append", default=[], help="可重复；直接给出 PMID")
    ap.add_argument("--input", type=Path, help="标识符列表文件（纯文本或 JSONL）")
    ap.add_argument("--core-id", action="append", default=[], help="可重复；标注支撑核心结论的标识符")
    ap.add_argument("--out", type=Path, help="JSON 报告输出路径；缺省只打印到 stdout")
    ap.add_argument("--mailto", default=os.environ.get("CROSSREF_MAILTO", ""))
    ap.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY", ""))
    ap.add_argument("--ncbi-tool", default=os.environ.get("NCBI_TOOL", "mitoevidence-hy3"))
    ap.add_argument("--ncbi-email", default=os.environ.get("NCBI_EMAIL", ""))
    ap.add_argument("--crossref-batch", type=int, default=CROSSREF_MAX_BATCH)
    ap.add_argument("--ncbi-batch", type=int, default=NCBI_MAX_BATCH)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--trust-env", action="store_true", help="使用 http_proxy/https_proxy（默认忽略，直连）"
    )
    args = ap.parse_args()

    entries: list[dict] = [{"id": d} for d in args.doi] + [{"id": p} for p in args.pmid]
    if args.input:
        entries += parse_input_file(args.input)
    if not entries:
        ap.error("未提供任何标识符：用 --doi / --pmid / --input")

    core_ids = {c.strip() for c in args.core_id}
    for entry in entries:
        if entry.get("is_core"):
            core_ids.add(str(entry["id"]))

    # 按类型分流
    buckets: dict[str, list[str]] = {"doi": [], "pmid": [], "pmcid": [], "unknown": []}
    expected: dict[str, ExpectedMetadata] = {}
    for entry in entries:
        raw = str(entry["id"])
        buckets[normalize_identifier(raw).id_type].append(raw)
        if entry.get("title") or entry.get("year") or entry.get("first_author"):
            expected[raw] = ExpectedMetadata(
                title=entry.get("title"),
                year=entry.get("year"),
                first_author=entry.get("first_author"),
            )

    results = {}
    notes: list[str] = []

    if buckets["doi"]:
        client = CrossrefClient(
            mailto=args.mailto,
            session=build_session(args.trust_env),
            batch_size=args.crossref_batch,
        )
        client.transport.timeout = args.timeout
        results.update(client.verify(buckets["doi"], expected=expected))
        notes.append(
            f"Crossref：{len(buckets['doi'])} 个 DOI，"
            f"限速 {client.transport.max_rps} rps，"
            f"响应头 {client.transport.rate_limit_headers or '未返回'}"
        )
        if not args.mailto:
            notes.append("警告：未设置 CROSSREF_MAILTO，未进 polite pool")

    if buckets["pmid"]:
        client = NcbiESummaryClient(
            api_key=args.ncbi_api_key,
            tool=args.ncbi_tool,
            email=args.ncbi_email,
            session=build_session(args.trust_env),
            batch_size=args.ncbi_batch,
        )
        client.transport.timeout = args.timeout
        results.update(client.verify(buckets["pmid"], expected=expected))
        notes.append(
            f"NCBI esummary：{len(buckets['pmid'])} 个 PMID，限速 {client.transport.max_rps} rps"
        )
        if not args.ncbi_email:
            notes.append("警告：未设置 NCBI_EMAIL，E-utilities 要求 tool+email")

    for raw in buckets["pmcid"] + buckets["unknown"]:
        notes.append(f"跳过 {raw!r}：本版本只核验 DOI 与 PMID（PMCID 走 Europe PMC，未实现）")

    # D1 汇总与档位
    summary = summarize_verifications(results.values())
    core_mismatch = sorted(
        raw
        for raw in core_ids
        if raw in results and results[raw].status is VerificationStatus.MISMATCH
    )
    event_flags = {
        "core_conclusion_uses_wrong_citation": bool(core_mismatch),
        "nonexistent_identifier_count": summary.nonexistent_identifier_count,
    }
    cfg = default_rubric()
    if summary.is_scorable:
        d1_input = DimensionInput(
            metric_value=summary.metadata_match_rate, event_flags=event_flags
        )
    else:
        d1_input = DimensionInput(is_na=True)
        notes.append("全部引用不可核验：D1 记 NA，并触发「必须人工复核」")
    d1_score = score_dimension("D1", d1_input, cfg)

    report = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluator_version": EVALUATOR_VERSION,
            "rubric_version": cfg.version,
            "rubric_config_sha256": cfg.sha256,
            "trust_env": args.trust_env,
            "n_input": len(entries),
        },
        "notes": notes,
        "records": [results[raw].model_dump() for raw in results],
        "summary": summary.model_dump(),
        "d1": {
            "event_flags": event_flags,
            "core_conclusion_mismatched_ids": core_mismatch,
            "score": d1_score.model_dump(),
        },
        "unresolved_unverifiable": summary.has_unresolved,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人类可读摘要
    print(f"# 引用核验报告  {report['meta']['timestamp_utc']}")
    for note in notes:
        print(f"  - {note}")
    print()
    print(f"{'状态':<11} {'标识符':<34} 题名·第一作者 / 原因")
    print("-" * 100)
    for raw, record in results.items():
        detail = record.reason or record.title or ""
        if record.first_author and not record.reason:
            detail = f"{detail} ·{record.first_author}"
        print(f"{record.status.value:<12} {(record.normalized_id or raw):<32} {str(detail)[:64]}")
    print("-" * 100)
    print(
        f"合计 {summary.total}：verified {summary.verified} / "
        f"mismatch {summary.mismatch} / unresolved {summary.unresolved}"
    )
    rate = summary.metadata_match_rate
    print(f"D1 metadata_match_rate p = {'NA' if rate is None else f'{rate:.4f}'}")
    print(
        f"D1 档位 = {'NA' if d1_score.is_na else d1_score.level}"
        f"（分档前 {d1_score.level_before_event_caps}，生效事件上限 {d1_score.event_caps_applied or '无'}）"
    )
    if summary.has_unresolved:
        print("注意：存在不可核验项 → 发布决策落 REVIEW，按方案 8.3 不得判伪造")
    if args.out:
        print(f"\nJSON 报告：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
