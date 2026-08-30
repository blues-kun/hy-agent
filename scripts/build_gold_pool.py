#!/usr/bin/env python3
"""金标证据池构建：12 篇综述引文 → 去重候选池 + 构建 manifest（方案 9.2 第一步）。

用法：
  # 引文池（默认 12 篇，核验报告 2.1）
  python scripts/build_gold_pool.py

  # 引文池 + OA 综述全文 XML（下载到 eval/data/corpus_raw/，已在 .gitignore）
  python scripts/build_gold_pool.py --fetch-fulltext

  # 自定义综述集合
  python scripts/build_gold_pool.py --pmid 37762083 --pmid 39834189

流程：逐篇 Europe PMC /references（分页）→ 空/失败且有 DOI 时走 Crossref
/works/{doi} 的 reference 字段兜底 → 按规范化 DOI/PMID 去重合并 → 输出
eval/data/evidence_pool_candidates.jsonl（每条含标识符、题名、年份、cited_by
溯源）与 evidence_pool_manifest.json（检索日期、逐篇引文数与来源、去重率、
API 版本、全文 sha256）。

网络：EPMC 保守 1 并发 + ≥1s 间隔；Crossref polite mailto 从 CROSSREF_MAILTO
环境变量读取。默认 trust_env=False 绕过本机死代理直连；ebi.ac.uk 与
api.crossref.org 均为境外域名，直连失败会在 manifest 的失败清单里如实记录。
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
    sys.path.insert(0, str(REPO_ROOT))


def build_session(trust_env: bool):
    import requests

    session = requests.Session()
    session.trust_env = trust_env
    return session


def main() -> int:
    # 延迟导入：sys.path 已在模块顶部补好仓库根目录。
    from tools.literature.crossref_refs import CrossrefRefsClient
    from tools.literature.epmc_client import EpmcClient
    from tools.literature.pool_builder import REVIEW_PMIDS, merge_references, sha256_of

    ap = argparse.ArgumentParser(description="构建金标证据池候选（12 篇综述引文合并去重）")
    ap.add_argument(
        "--pmid", action="append", default=[],
        help="可重复；缺省用核验报告 2.1 的 12 篇综述",
    )
    ap.add_argument("--out", type=Path, default=Path("eval/data/evidence_pool_candidates.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("eval/data/evidence_pool_manifest.json"))
    ap.add_argument("--fetch-fulltext", action="store_true", help="下载 OA 综述的 fullTextXML")
    ap.add_argument(
        "--corpus-dir", type=Path, default=Path("eval/data/corpus_raw"),
        help="全文 XML 输出目录（已在 .gitignore）",
    )
    ap.add_argument("--mailto", default=os.environ.get("CROSSREF_MAILTO", ""))
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument(
        "--trust-env", action="store_true", help="使用 http_proxy/https_proxy（默认忽略，直连）"
    )
    args = ap.parse_args()

    review_pmids = [p.strip() for p in args.pmid] or list(REVIEW_PMIDS)
    epmc = EpmcClient(session=build_session(args.trust_env), page_size=args.page_size)
    crossref = CrossrefRefsClient(mailto=args.mailto, session=build_session(args.trust_env))

    per_review: dict[str, list[dict]] = {}
    review_rows: list[dict] = []
    failures: list[str] = []

    for index, pmid in enumerate(review_pmids, start=1):
        row: dict = {"index": index, "pmid": pmid}
        metadata, meta_error = epmc.fetch_metadata(pmid)
        if metadata:
            row.update(
                {
                    "title": metadata["title"],
                    "doi": metadata["doi"],
                    "pmcid": metadata["pmcid"],
                    "is_open_access": metadata["is_open_access"],
                    "year": metadata["year"],
                }
            )
        else:
            row["metadata_error"] = meta_error
            failures.append(f"{pmid}: 元数据获取失败（{meta_error}）")

        refs, refs_meta, refs_error = epmc.fetch_references(pmid)
        row["epmc_hit_count"] = refs_meta.get("hit_count")
        source = "epmc"
        if refs_error or not refs:
            reason = refs_error or "EPMC references 为空"
            doi = (metadata or {}).get("doi")
            if doi:
                refs, crossref_meta, crossref_error = crossref.fetch_references(doi)
                source = "crossref"
                row["crossref_declared"] = crossref_meta.get("declared")
                row["crossref_reference_count"] = crossref_meta.get("reference_count")
                row["fallback_reason"] = reason
                if crossref_error or not refs:
                    source = "none"
                    failures.append(
                        f"{pmid}: EPMC（{reason}）与 Crossref"
                        f"（{crossref_error or '无 reference'}）均未取到引文"
                    )
            else:
                source = "none"
                failures.append(f"{pmid}: EPMC 失败（{reason}）且无 DOI 可走 Crossref")
        row["refs_source"] = source
        row["refs_used"] = len(refs)
        per_review[pmid] = refs
        review_rows.append(row)
        print(
            f"[{index}/{len(review_pmids)}] PMID {pmid}: {len(refs)} 条引文"
            f"（{source}，EPMC hitCount={row['epmc_hit_count']}）",
            flush=True,
        )

    candidates, stats = merge_references(per_review)

    fulltext_rows: list[dict] = []
    if args.fetch_fulltext:
        args.corpus_dir.mkdir(parents=True, exist_ok=True)
        for row in review_rows:
            pmcid = row.get("pmcid")
            if not pmcid:
                fulltext_rows.append(
                    {"pmid": row["pmid"], "pmcid": None, "error": "无 PMCID，无法取全文"}
                )
                continue
            xml, error = epmc.fetch_fulltext_xml(pmcid)
            entry = {
                "pmid": row["pmid"],
                "pmcid": pmcid,
                "is_open_access": row.get("is_open_access"),
            }
            if xml is None:
                entry["error"] = error
                if "404" not in error:  # 404=非 OA 属预期，不进失败清单
                    failures.append(f"{row['pmid']}: 全文下载失败（{error}）")
            else:
                data = xml.encode("utf-8")
                path = args.corpus_dir / f"{pmcid}.xml"
                path.write_bytes(data)
                entry.update({"path": str(path), "bytes": len(data), "sha256": sha256_of(data)})
            fulltext_rows.append(entry)
            status = entry.get("sha256", entry.get("error", ""))[:64]
            print(f"  全文 {pmcid}: {status}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for candidate in candidates:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "scripts/build_gold_pool.py",
        "epmc_api_version": epmc.api_version,
        "epmc_page_size": args.page_size,
        "trust_env": args.trust_env,
        "reviews": review_rows,
        "pool": stats,
        "fulltext": fulltext_rows,
        "failures": failures,
        "outputs": {"candidates": str(args.out), "corpus_dir": str(args.corpus_dir)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n候选池：{stats['unique']} 条（原始 {stats['total_raw']}，"
          f"去重率 {stats['dedup_rate']:.1%}，无标识符 {stats['n_unidentified']}，"
          f"被多篇综述共引 {stats['n_cited_by_multiple']}）")
    print(f"EPMC REST 版本：{epmc.api_version}")
    if failures:
        print(f"失败项 {len(failures)} 条：")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("失败项：无")
    print(f"候选池：{args.out}\nmanifest：{args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
