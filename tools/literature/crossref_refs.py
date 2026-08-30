"""Crossref /works/{doi} 的 reference 字段：非 OA 综述引文列表的兜底通道。

核验报告 2.1 第 9 篇（PMID 39834189，489 条引文）无 PMC 全文，Europe PMC
references 覆盖不到时走这里。复用 D1 客户端的 HttpTransport：响应头自适应
限速、429 退避、trust_env=False 直连；polite mailto 从环境变量 CROSSREF_MAILTO
读取，不落盘。

注意：Crossref reference 条目质量参差——部分只有 unstructured 文本没有 DOI，
解析结果如实保留 None，由下游去重与人工标注处理。
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from evaluator.rules.identifier_check import USER_AGENT_BASE, HttpTransport

CROSSREF_WORKS_API = "https://api.crossref.org/works"
# /works/{doi} 单条查询也保守 ≤1 rps（与 EPMC 同一管线里串行执行）。
CROSSREF_REFS_MAX_RPS = 1.0


def _parse_year(value: Any) -> int | None:
    text = str(value or "")
    return int(text[:4]) if text[:4].isdigit() else None


def parse_crossref_reference(entry: Mapping[str, Any]) -> dict[str, Any]:
    doi = str(entry.get("DOI") or "").strip().lower() or None
    title = entry.get("article-title") or entry.get("volume-title") or None
    unstructured = str(entry.get("unstructured") or "").strip() or None
    return {
        "pmid": None,  # Crossref reference 不含 PMID
        "doi": doi,
        "title": title,
        "year": _parse_year(entry.get("year")),
        "unstructured": unstructured if title is None else None,
        "key": entry.get("key"),
        "source": "crossref",
    }


class CrossrefRefsClient:
    """取单个 DOI 的 reference 列表。transport/session 可注入。"""

    def __init__(
        self,
        mailto: str | None = None,
        session: Any = None,
        transport: HttpTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.mailto = mailto if mailto is not None else os.environ.get("CROSSREF_MAILTO", "")
        user_agent = (
            f"{USER_AGENT_BASE} (mailto:{self.mailto})" if self.mailto else USER_AGENT_BASE
        )
        self.transport = transport or HttpTransport(
            session=session,
            max_rps=CROSSREF_REFS_MAX_RPS,
            sleep_fn=sleep_fn,
            user_agent=user_agent,
        )

    def fetch_references(self, doi: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """返回 (引文列表, {"reference_count","references_count"}, 错误说明)。"""
        params: dict[str, Any] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        status, body, error = self.transport.request(
            "GET", f"{CROSSREF_WORKS_API}/{doi}", params=params or None
        )
        if error:
            return [], {}, error
        if status == 404:
            return [], {}, f"DOI {doi} 未在 Crossref 注册（HTTP 404）"
        if status != 200:
            return [], {}, f"HTTP {status}"
        message = (body or {}).get("message") or {}
        raw = message.get("reference") or []
        meta = {
            # reference-count = 该文列出的引文数；references-count 为同义新键。
            "reference_count": message.get("reference-count"),
            "declared": len(raw),
        }
        if not raw:
            return [], meta, "Crossref 记录无 reference 字段（出版商未存引文）"
        return [parse_crossref_reference(entry) for entry in raw], meta, ""
