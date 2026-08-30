"""Europe PMC REST 客户端（核验报告 3.1）。

三个端点：
  - /search?query=EXT_ID:{pmid} AND SRC:MED   按 PMID 查元数据
    （resultType=core，含 isOpenAccess、pmcid、doi、pubYear）；
  - /{PMCID}/fullTextXML                       OA 全文 XML；非 OA 会 404，
    这是预期行为（核验报告 2.1 第 6 篇实测 404），调用方按容错处理；
  - /MED/{pmid}/references                     参考文献列表，分页（page/pageSize）。

限流：官方文档没有给出限流数字，因此保守处理——同步单并发 + 相邻请求间隔
≥1s，429/5xx 指数退避。网络环境：本机 http_proxy/https_proxy 可能指向已失效
代理，默认 trust_env=False 直连；ebi.ac.uk 为境外域名，直连失败按错误如实返回。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = "MitoEvidence-Hy3-literature/0.2"

# 官方无限流数字 → 保守 1 并发 + ≥1s 间隔。
EPMC_MIN_INTERVAL_S = 1.0
# references 分页大小；官方允许更大，取保守值避免超大响应。
EPMC_PAGE_SIZE = 100
# hitCount 异常时的分页保险丝。
_MAX_PAGES = 200


@dataclass
class EpmcTransport:
    """GET 执行器：≥1s 节流 + 429/5xx/传输失败退避。session/sleep_fn 可注入。"""

    session: Any = None
    min_interval: float = EPMC_MIN_INTERVAL_S
    max_retries: int = 4
    timeout: float = 60.0
    sleep_fn: Callable[[float], None] = time.sleep
    trust_env: bool = False
    _last_call: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.session is None:
            import requests

            self.session = requests.Session()
            self.session.trust_env = self.trust_env  # 本机死代理：默认直连
        headers = getattr(self.session, "headers", None)
        if hasattr(headers, "update"):
            headers.update({"User-Agent": USER_AGENT})

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self.sleep_fn(wait)
        self._last_call = time.monotonic()

    def get(
        self, url: str, params: Mapping[str, Any] | None = None, expect: str = "json"
    ) -> tuple[int, Any, str]:
        """返回 (status, JSON 对象或原文文本, 错误说明)。

        4xx（含 404）不重试、error 为空，由调用方按语义解释——
        fullTextXML 404 = 非 OA，是预期结果不是故障。
        """
        last_error = "未发起请求"
        for attempt in range(self.max_retries):
            is_last = attempt == self.max_retries - 1
            self._throttle()
            try:
                response = self.session.request("GET", url, params=params, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 —— 传输层异常统一按可重试失败处理
                last_error = f"传输失败（{type(exc).__name__}: {exc}）"
                if not is_last:
                    self.sleep_fn(min(2.0**attempt, 8.0))
                continue
            status = int(getattr(response, "status_code", 0))
            if status == 429 or 500 <= status < 600:
                last_error = f"HTTP {status}"
                if not is_last:
                    self.sleep_fn(min(2.0 ** (attempt + 1), 16.0))
                continue
            if expect == "text":
                return status, getattr(response, "text", ""), ""
            try:
                return status, response.json(), ""
            except Exception as exc:  # noqa: BLE001 —— 非 JSON 响应无法给出可信结论
                return status, None, f"响应非 JSON（{type(exc).__name__}: {exc}）"
        return -1, None, f"重试 {self.max_retries} 次后仍失败：{last_error}"


def _parse_year(value: Any) -> int | None:
    text = str(value or "")
    return int(text[:4]) if text[:4].isdigit() else None


def parse_reference_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """references 端点单条引文 → 统一结构。source==MED 的 id 才是 PMID。"""
    source = str(entry.get("source") or "")
    ref_id = str(entry.get("id") or "").strip()
    doi = str(entry.get("doi") or "").strip().lower() or None
    return {
        "pmid": ref_id if (source == "MED" and ref_id.isdigit()) else None,
        "doi": doi,
        "title": entry.get("title") or None,
        "year": _parse_year(entry.get("pubYear")),
        "cited_order": entry.get("citedOrder"),
        "source": "epmc",
    }


class EpmcClient:
    """Europe PMC 客户端。transport 可注入，离线测试不发真实请求。"""

    def __init__(
        self,
        session: Any = None,
        transport: EpmcTransport | None = None,
        page_size: int = EPMC_PAGE_SIZE,
        sleep_fn: Callable[[float], None] = time.sleep,
        base_url: str = EPMC_BASE,
    ):
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.transport = transport or EpmcTransport(session=session, sleep_fn=sleep_fn)
        # 响应体自带 REST API 版本号（如 "6.9"），记入构建 manifest。
        self.api_version: str | None = None

    def _remember_version(self, body: Any) -> None:
        version = (body or {}).get("version")
        if version:
            self.api_version = str(version)

    # -- 元数据 ---------------------------------------------------------------

    def fetch_metadata(self, pmid: str) -> tuple[dict[str, Any] | None, str]:
        """按 PMID 查元数据。返回 (记录, 错误说明)；未命中记录为 None。"""
        status, body, error = self.transport.get(
            f"{self.base_url}/search",
            params={
                "query": f"EXT_ID:{pmid} AND SRC:MED",
                "format": "json",
                "resultType": "core",
                "pageSize": 1,
            },
        )
        if error:
            return None, error
        if status != 200:
            return None, f"HTTP {status}"
        self._remember_version(body)
        results = ((body or {}).get("resultList") or {}).get("result") or []
        if not results:
            return None, f"EXT_ID:{pmid} 未命中"
        item = results[0]
        return {
            "pmid": str(item.get("pmid") or item.get("id") or pmid),
            "pmcid": item.get("pmcid") or None,
            "doi": (str(item.get("doi")).strip().lower() or None) if item.get("doi") else None,
            "title": item.get("title") or None,
            "year": _parse_year(item.get("pubYear")),
            "journal": ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
            or item.get("journalTitle"),
            "is_open_access": str(item.get("isOpenAccess") or "N") == "Y",
            "in_epmc": str(item.get("inEPMC") or "N") == "Y",
        }, ""

    # -- OA 全文 ---------------------------------------------------------------

    def fetch_fulltext_xml(self, pmcid: str) -> tuple[str | None, str]:
        """取 OA 全文 XML。非 OA 返回 404——按预期容错，错误说明注明原因。"""
        status, text, error = self.transport.get(
            f"{self.base_url}/{pmcid}/fullTextXML", expect="text"
        )
        if error:
            return None, error
        if status == 404:
            return None, "HTTP 404：非 OA 或无 XML 全文（预期内，容错处理）"
        if status != 200:
            return None, f"HTTP {status}"
        if not text or not text.lstrip().startswith("<"):
            return None, "响应为空或非 XML"
        return text, ""

    # -- 参考文献（分页） --------------------------------------------------------

    def fetch_references(self, pmid: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """取 /MED/{pmid}/references 全量（分页）。

        返回 (引文列表, {"hit_count","pages"}, 错误说明)；中途失败返回已取部分
        并带错误说明，由调用方决定是否走 Crossref 兜底。
        """
        references: list[dict[str, Any]] = []
        hit_count: int | None = None
        page = 0
        while page < _MAX_PAGES:
            page += 1
            status, body, error = self.transport.get(
                f"{self.base_url}/MED/{pmid}/references",
                params={"page": page, "pageSize": self.page_size, "format": "json"},
            )
            meta = {"hit_count": hit_count, "pages": page}
            if error:
                return references, meta, error
            if status != 200:
                return references, meta, f"HTTP {status}"
            self._remember_version(body)
            if hit_count is None:
                hit_count = int((body or {}).get("hitCount") or 0)
            batch = (((body or {}).get("referenceList") or {}).get("reference")) or []
            references.extend(parse_reference_entry(entry) for entry in batch)
            if not batch or len(references) >= (hit_count or 0):
                break
        return references, {"hit_count": hit_count or 0, "pages": page}, ""
