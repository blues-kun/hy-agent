"""金标语料工具链的离线测试（Europe PMC / Crossref refs / 合并去重）。

全部用 mock session：不发真实请求、不产生真实等待。
真实联网构建由 scripts/build_gold_pool.py 单独执行。
"""
from __future__ import annotations

from tools.literature.crossref_refs import CrossrefRefsClient, parse_crossref_reference
from tools.literature.epmc_client import (
    EpmcClient,
    EpmcTransport,
    parse_reference_entry,
)
from tools.literature.pool_builder import (
    REVIEW_PMIDS,
    dedupe_key,
    merge_references,
)

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None, bad_json=False):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._payload = {} if payload is None else payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("响应不是 JSON")
        return self._payload


class FakeSession:
    """responses 可为列表（依序弹出，元素可为异常）或 call -> FakeResponse 的可调用。"""

    def __init__(self, responses):
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []
        self._responses = responses

    def request(self, method, url, params=None, data=None, timeout=None):
        call = {"method": method, "url": url, "params": dict(params or {}), "timeout": timeout}
        self.calls.append(call)
        if callable(self._responses):
            item = self._responses(call)
        elif self._responses:
            item = self._responses.pop(0)
        else:
            item = FakeResponse(200, {})
        if isinstance(item, BaseException):
            raise item
        return item


def make_client(responses, page_size=100):
    session = FakeSession(responses)
    sleeps: list[float] = []
    transport = EpmcTransport(session=session, sleep_fn=sleeps.append)
    return EpmcClient(transport=transport, page_size=page_size), session, sleeps


# ---------------------------------------------------------------------------
# Europe PMC：元数据
# ---------------------------------------------------------------------------


SEARCH_HIT = {
    "version": "6.9",
    "hitCount": 1,
    "resultList": {
        "result": [
            {
                "id": "37762083",
                "pmid": "37762083",
                "pmcid": "PMC10530730",
                "doi": "10.3390/IJMS241813782",
                "title": "Mitochondrial Dynamics and Insulin Secretion",
                "pubYear": "2023",
                "isOpenAccess": "Y",
                "inEPMC": "Y",
                "journalInfo": {"journal": {"title": "Int J Mol Sci"}},
            }
        ]
    },
}


def test_metadata_parses_oa_flag_pmcid_and_lowercases_doi():
    client, session, _ = make_client([FakeResponse(200, SEARCH_HIT)])
    record, error = client.fetch_metadata("37762083")
    assert error == ""
    assert record["pmcid"] == "PMC10530730"
    assert record["is_open_access"] is True
    assert record["doi"] == "10.3390/ijms241813782"
    assert record["year"] == 2023
    assert record["journal"] == "Int J Mol Sci"
    assert client.api_version == "6.9"
    params = session.calls[0]["params"]
    assert params["query"] == "EXT_ID:37762083 AND SRC:MED"
    assert params["resultType"] == "core"


def test_metadata_miss_returns_none_with_reason():
    client, _, _ = make_client(
        [FakeResponse(200, {"version": "6.9", "hitCount": 0, "resultList": {"result": []}})]
    )
    record, error = client.fetch_metadata("99999999")
    assert record is None
    assert "未命中" in error


# ---------------------------------------------------------------------------
# Europe PMC：OA 全文（404 容错）
# ---------------------------------------------------------------------------


def test_fulltext_200_returns_xml_text():
    xml = '<?xml version="1.0"?><article><body>…</body></article>'
    client, session, _ = make_client([FakeResponse(200, text=xml)])
    text, error = client.fetch_fulltext_xml("PMC10530730")
    assert error == ""
    assert text == xml
    assert session.calls[0]["url"].endswith("/PMC10530730/fullTextXML")


def test_fulltext_404_is_tolerated_not_retried():
    """核验报告 2.1 第 6 篇：PMC 存在但非 OA，fullTextXML 实测 404——预期容错。"""
    client, session, _ = make_client([FakeResponse(404, text="not found")])
    text, error = client.fetch_fulltext_xml("PMC8275885")
    assert text is None
    assert "404" in error
    assert len(session.calls) == 1  # 4xx 不重试


# ---------------------------------------------------------------------------
# Europe PMC：参考文献分页
# ---------------------------------------------------------------------------


def _refs_page(hit_count, start, count):
    return FakeResponse(
        200,
        {
            "version": "6.9",
            "hitCount": hit_count,
            "referenceList": {
                "reference": [
                    {
                        "id": str(10_000_000 + start + i),
                        "source": "MED",
                        "title": f"Paper {start + i}",
                        "pubYear": "2020",
                        "citedOrder": start + i + 1,
                    }
                    for i in range(count)
                ]
            },
        },
    )


def test_references_paginate_until_hit_count():
    """hitCount=250、pageSize=100 → 3 页请求，250 条完整取回。"""
    client, session, _ = make_client(
        [_refs_page(250, 0, 100), _refs_page(250, 100, 100), _refs_page(250, 200, 50)]
    )
    refs, meta, error = client.fetch_references("37762083")
    assert error == ""
    assert len(refs) == 250
    assert meta == {"hit_count": 250, "pages": 3}
    assert [c["params"]["page"] for c in session.calls] == [1, 2, 3]
    assert all(c["params"]["pageSize"] == 100 for c in session.calls)


def test_references_stop_on_empty_page_even_if_hit_count_lies():
    client, session, _ = make_client(
        [
            _refs_page(999, 0, 100),
            FakeResponse(200, {"hitCount": 999, "referenceList": {"reference": []}}),
        ]
    )
    refs, meta, error = client.fetch_references("1")
    assert error == ""
    assert len(refs) == 100
    assert len(session.calls) == 2


def test_reference_entry_parsing_pmid_only_for_med_source():
    med = parse_reference_entry(
        {"id": "12345", "source": "MED", "doi": "10.1000/ABC", "pubYear": "2019"}
    )
    assert med["pmid"] == "12345"
    assert med["doi"] == "10.1000/abc"
    assert med["year"] == 2019
    other = parse_reference_entry({"id": "PPR54321", "source": "PPR", "title": "预印本"})
    assert other["pmid"] is None


def test_transport_enforces_min_interval_between_requests():
    session = FakeSession([FakeResponse(200, {}), FakeResponse(200, {})])
    sleeps: list[float] = []
    transport = EpmcTransport(session=session, sleep_fn=sleeps.append, min_interval=1.0)
    transport.get("https://example.org/a")
    transport.get("https://example.org/b")
    # 第二次请求距第一次远小于 1s，必须节流补足间隔。
    assert sleeps and max(sleeps) > 0.5


# ---------------------------------------------------------------------------
# Crossref reference 兜底
# ---------------------------------------------------------------------------


CROSSREF_WORK = {
    "message": {
        "DOI": "10.1089/ars.2024.0799",
        "reference-count": 3,
        "reference": [
            {"key": "r1", "DOI": "10.2337/DB16-0405", "article-title": "Glucose sensing",
             "year": "2016"},
            {"key": "r2", "unstructured": "Smith J. Some old book chapter. 1998."},
            {"key": "r3", "DOI": "10.1074/jbc.m117.000000", "year": "2017"},
        ],
    }
}


def test_crossref_refs_parse_and_polite_mailto():
    session = FakeSession([FakeResponse(200, CROSSREF_WORK)])
    client = CrossrefRefsClient(mailto="team@example.org", session=session, sleep_fn=lambda _: None)
    refs, meta, error = client.fetch_references("10.1089/ars.2024.0799")
    assert error == ""
    assert meta == {"reference_count": 3, "declared": 3}
    assert session.calls[0]["url"].endswith("/works/10.1089/ars.2024.0799")
    assert session.calls[0]["params"]["mailto"] == "team@example.org"
    assert refs[0]["doi"] == "10.2337/db16-0405"
    assert refs[0]["title"] == "Glucose sensing"
    assert refs[1]["doi"] is None
    assert refs[1]["unstructured"].startswith("Smith J.")
    assert refs[2]["year"] == 2017


def test_crossref_refs_missing_reference_field_is_reported():
    session = FakeSession([FakeResponse(200, {"message": {"DOI": "10.1/x"}})])
    client = CrossrefRefsClient(mailto="", session=session, sleep_fn=lambda _: None)
    refs, _, error = client.fetch_references("10.1/x")
    assert refs == []
    assert "reference" in error


def test_crossref_refs_404_reports_error():
    session = FakeSession([FakeResponse(404, {})])
    client = CrossrefRefsClient(mailto="", session=session, sleep_fn=lambda _: None)
    refs, _, error = client.fetch_references("10.9999/none")
    assert refs == []
    assert "404" in error


def test_parse_crossref_reference_prefers_title_over_unstructured():
    ref = parse_crossref_reference(
        {"DOI": "10.1/y", "article-title": "T", "unstructured": "T. et al."}
    )
    assert ref["title"] == "T"
    assert ref["unstructured"] is None


# ---------------------------------------------------------------------------
# 合并去重
# ---------------------------------------------------------------------------


def test_dedupe_key_normalizes_doi_and_pmid_variants():
    assert dedupe_key({"doi": "https://doi.org/10.1000/ABC"}) == dedupe_key({"doi": "10.1000/abc"})
    assert dedupe_key({"pmid": "PMID: 123"}) == dedupe_key({"pmid": "123"})
    assert dedupe_key({"doi": "10.1000/x", "pmid": "123"}) == "doi:10.1000/x"  # DOI 优先
    assert dedupe_key({"title": "只有题名"}) is None


def test_merge_dedupes_across_reviews_and_tracks_cited_by():
    per_review = {
        "A": [
            {"doi": "10.1000/ABC", "title": "Shared paper", "year": 2020, "source": "epmc"},
            {"pmid": "111", "title": "Only in A", "year": 2019, "source": "epmc"},
        ],
        "B": [
            # 同一 DOI 不同大小写 + 补全缺失年份的能力（本条无 year）
            {"doi": "10.1000/abc", "title": None, "year": None, "source": "crossref"},
            {"pmid": "222", "title": "Only in B", "year": 2021, "source": "epmc"},
        ],
    }
    candidates, stats = merge_references(per_review)
    assert stats["total_raw"] == 4
    assert stats["unique"] == 3
    assert stats["duplicates_removed"] == 1
    assert stats["dedup_rate"] == round(1 - 3 / 4, 4)
    shared = next(c for c in candidates if c["doi"] == "10.1000/abc")
    assert shared["cited_by"] == ["A", "B"]
    assert sorted(shared["sources"]) == ["crossref", "epmc"]
    assert shared["title"] == "Shared paper"  # 先到者提供题名
    assert stats["n_cited_by_multiple"] == 1


def test_merge_backfills_fields_from_later_duplicates():
    per_review = {
        "A": [{"doi": "10.1000/z", "title": None, "year": None, "source": "crossref"}],
        "B": [{"doi": "10.1000/Z", "title": "Filled later", "year": 2018, "source": "epmc"}],
    }
    candidates, _ = merge_references(per_review)
    assert len(candidates) == 1
    assert candidates[0]["title"] == "Filled later"
    assert candidates[0]["year"] == 2018


def test_merge_groups_unidentified_by_title_and_keeps_blank_entries():
    per_review = {
        "A": [
            {"title": "An Untraceable Chapter", "source": "crossref"},
            {"unstructured": "An untraceable   CHAPTER", "source": "crossref"},  # 同题名归并
            {"source": "crossref"},  # 全空：原样保留
        ],
    }
    candidates, stats = merge_references(per_review)
    assert stats["total_raw"] == 3
    assert stats["n_unidentified"] == 3
    assert stats["unique"] == 2
    assert all(c["id_type"] == "none" for c in candidates)


def test_review_pmids_match_verification_report_2_1():
    assert len(REVIEW_PMIDS) == 12
    assert REVIEW_PMIDS[6] == "37762083"  # 第 7 篇（报告基准 136 条）
    assert REVIEW_PMIDS[8] == "39834189"  # 第 9 篇（报告基准 489 条，走 Crossref）
