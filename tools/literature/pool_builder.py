"""引文池合并去重与构建 manifest（纯函数，离线可测）。

方案 9.2：金标准证据池从 12 篇已核验综述（核验报告 2.1，约 2,043 条参考文献）
的引文出发。本模块把逐篇引文列表合并为去重后的候选池：

  - 去重键：规范化 DOI 优先，其次 PMID（复用 D1 的规范化器）；
  - 两者皆无的条目（Crossref unstructured 引文常见）无法机械去重，按规范化
    题名聚一次，仍无题名的原样保留并计数——由人工标注阶段处理；
  - 同一论文若在 A 综述中以 PMID 出现、在 B 综述中只有 DOI，机械去重无法
    识别为同一条（需要联网互查），因此去重率是**下界**——manifest 里如实注明。

每条候选保留 cited_by 溯源列表（哪些综述引用了它），供 D3 池化关键证据
标注与「同一数据多篇报告」类对抗样本（方案 9.3 第 8 类）使用。
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from evaluator.rules.identifier_check import normalize_doi, normalize_pmid

# 核验报告 2.1：主池 12 篇已核验综述的 PMID（顺序与报告表格一致）。
REVIEW_PMIDS = (
    "25659350",  # 1  Kaufman 2015 Mol Aspects Med
    "26873508",  # 2  Baltrusch 2016 Diabetologia（无 PMC）
    "28951827",  # 3  Mulder 2017 Mol Metab
    "32894309",  # 4  Rutter 2020 Diabetologia
    "33802289",  # 5  Weiser 2021 Int J Mol Sci
    "34016598",  # 6  Pearson 2021 Diabetes（PMC 存在但非 OA，fullTextXML 404）
    "37762083",  # 7  Kabra 2023 Int J Mol Sci
    "38404962",  # 8  Rivera Nieves 2024 Front Mol Biosci
    "39834189",  # 9  Ježek 2025 Antioxid Redox Signal（无 PMC，引文走 Crossref）
    "40136648",  # 10 Zaher 2025 Cells
    "41109799",  # 11 Levi-D'Ancona 2026 Trends Endocrinol Metab（无 PMC）
    "42051080",  # 12 Li 2026 J Diabetes
)

_TITLE_TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")


def _normalized_title_key(title: str | None) -> str | None:
    tokens = _TITLE_TOKEN_RE.findall(str(title or "").lower())
    if not tokens:
        return None
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


def dedupe_key(ref: Mapping[str, Any]) -> str | None:
    """规范化去重键：DOI 优先，其次 PMID；两者皆无返回 None。"""
    doi_raw = ref.get("doi")
    if doi_raw:
        norm = normalize_doi(str(doi_raw))
        if norm.is_valid:
            return f"doi:{norm.value}"
    pmid_raw = ref.get("pmid")
    if pmid_raw:
        norm = normalize_pmid(str(pmid_raw))
        if norm.is_valid:
            return f"pmid:{norm.value}"
    return None


def merge_references(
    per_review: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """逐篇引文列表 → 去重候选池。

    per_review：{综述 PMID: [引文条目, …]}；条目字段见 epmc_client.parse_reference_entry
    与 crossref_refs.parse_crossref_reference。
    返回 (候选池列表, 统计)。
    """
    pool: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    total_raw = 0
    n_unidentified = 0

    for review_pmid, references in per_review.items():
        for ref in references:
            total_raw += 1
            key = dedupe_key(ref)
            if key is None:
                title_key = _normalized_title_key(ref.get("title") or ref.get("unstructured"))
                if title_key is None:
                    # 连题名都没有：无法机械去重，用内容哈希保留原样。
                    blob = str(sorted(ref.items()))
                    title_key = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
                key = f"title:{title_key}"
                n_unidentified += 1

            entry = pool.get(key)
            if entry is None:
                doi_norm = normalize_doi(str(ref.get("doi"))).value if ref.get("doi") else None
                pmid_norm = normalize_pmid(str(ref.get("pmid"))).value if ref.get("pmid") else None
                identifier = doi_norm or pmid_norm
                entry = {
                    "identifier": identifier,
                    "id_type": "doi" if doi_norm else ("pmid" if pmid_norm else "none"),
                    "doi": doi_norm,
                    "pmid": pmid_norm,
                    "title": ref.get("title") or None,
                    "year": ref.get("year"),
                    "unstructured": ref.get("unstructured") or None,
                    "cited_by": [],
                    "sources": [],
                }
                pool[key] = entry
                order.append(key)
            # 溯源与补全：后到的重复条目可补上先前缺失的字段。
            if review_pmid not in entry["cited_by"]:
                entry["cited_by"].append(review_pmid)
            ref_source = str(ref.get("source") or "")
            if ref_source and ref_source not in entry["sources"]:
                entry["sources"].append(ref_source)
            for field in ("title", "year", "doi", "pmid"):
                if entry.get(field) is None and ref.get(field) is not None:
                    if field == "doi":
                        entry["doi"] = normalize_doi(str(ref["doi"])).value
                    elif field == "pmid":
                        entry["pmid"] = normalize_pmid(str(ref["pmid"])).value
                    else:
                        entry[field] = ref[field]

    candidates = [pool[key] for key in order]
    unique = len(candidates)
    stats = {
        "total_raw": total_raw,
        "unique": unique,
        "duplicates_removed": total_raw - unique,
        "dedup_rate": round(1 - unique / total_raw, 4) if total_raw else None,
        "n_unidentified": n_unidentified,
        "n_cited_by_multiple": sum(1 for c in candidates if len(c["cited_by"]) > 1),
        "note": (
            "去重键为规范化 DOI/PMID；跨源同文（一处只有 PMID、另一处只有 DOI）"
            "无法机械识别，去重率为下界。无标识符条目按规范化题名聚合。"
        ),
    }
    return candidates, stats


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
