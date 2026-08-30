"""D1 标识符核验的离线测试。

全部用 mock session，不发起任何真实请求、不产生真实等待：
HttpTransport 的 session 与 sleep_fn 均可注入。
真实联网冒烟由 scripts/verify_citations.py 单独执行。
"""
from __future__ import annotations

import pytest

from evaluator.rules.identifier_check import (
    CROSSREF_MAX_BATCH,
    NCBI_MAX_BATCH,
    NCBI_RPS_WITH_KEY,
    NCBI_RPS_WITHOUT_KEY,
    CitationCheckSummary,
    CrossrefClient,
    ExpectedMetadata,
    HttpTransport,
    MetadataMatchPolicy,
    NcbiESummaryClient,
    chunked,
    compare_metadata,
    first_author_matches,
    normalize_doi,
    normalize_identifier,
    normalize_pmcid,
    normalize_pmid,
    summarize_verifications,
    surname_candidates,
    title_similarity,
)
from evaluator.schemas import VerificationStatus

DOI_JUDGE = "10.1038/s41746-025-02005-2"
TITLE_JUDGE = "Evaluating clinical AI summaries with large language models as judges"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, bad_json=False):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = {} if payload is None else payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("响应不是 JSON")
        return self._payload


class FakeSession:
    """responses 可为 FakeResponse/异常的列表，或 call -> FakeResponse 的可调用对象。"""

    def __init__(self, responses):
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []
        self._responses = responses

    def request(self, method, url, params=None, data=None, timeout=None):
        call = {
            "method": method,
            "url": url,
            "params": dict(params or {}),
            "data": dict(data or {}),
            "timeout": timeout,
        }
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

    def requested_dois(self, index: int = 0) -> list[str]:
        raw = self.calls[index]["params"]["filter"]
        return [chunk.split("doi:", 1)[1] for chunk in raw.split(",")]


def crossref_echo(records: dict[str, dict], headers: dict | None = None):
    """回显请求 filter 中命中的 DOI；未收录的 DOI 不出现在 items 中。

    record 可带 "authors"：Crossref 风格的 [{"given":…, "family":…, "sequence":…}] 列表。
    """

    def responder(call):
        items = []
        for chunk in call["params"]["filter"].split(","):
            doi = chunk.split("doi:", 1)[1]
            record = records.get(doi)
            if record is None:
                continue
            item = {
                "DOI": record.get("returned_doi", doi),
                "title": [record["title"]],
                "container-title": [record["journal"]],
                "issued": {"date-parts": [[record["year"], 7, 10]]},
            }
            if record.get("authors"):
                item["author"] = record["authors"]
            items.append(item)
        return FakeResponse(200, {"status": "ok", "message": {"items": items}}, headers)

    return responder


def make_crossref(responses, sleeps=None, **kwargs) -> tuple[CrossrefClient, FakeSession, list]:
    recorded: list[float] = [] if sleeps is None else sleeps
    session = FakeSession(responses)
    client = CrossrefClient(
        mailto="ci@example.org", session=session, sleep_fn=recorded.append, **kwargs
    )
    return client, session, recorded


# ---------------------------------------------------------------------------
# DOI / PMID / PMCID 规范化
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (DOI_JUDGE, DOI_JUDGE),
        ("10.1038/S41746-025-02005-2", DOI_JUDGE),  # DOI 大小写不敏感
        ("https://doi.org/10.1038/s41746-025-02005-2", DOI_JUDGE),
        ("http://dx.doi.org/10.1038/s41746-025-02005-2", DOI_JUDGE),
        ("https://www.doi.org/10.2427/12267", "10.2427/12267"),
        ("doi:10.2427/12267", "10.2427/12267"),
        ("DOI: 10.1111/cts.13302", "10.1111/cts.13302"),
        ("  10.1038/s42256-020-00287-7  ", "10.1038/s42256-020-00287-7"),
        ("<10.1038/s41597-023-01960-3>", "10.1038/s41597-023-01960-3"),
        ("10.1348/000711006X126600.", "10.1348/000711006x126600"),
        ("doi.org/10.18653/v1/2026.acl-long.1161", "10.18653/v1/2026.acl-long.1161"),
    ],
)
def test_normalize_doi_accepts_common_formats(raw, expected):
    assert normalize_doi(raw).value == expected


@pytest.mark.parametrize("raw", ["", "   ", "not-a-doi", "11.1038/x", "10.abc/x", "10.1038", "PMC123"])
def test_normalize_doi_rejects_invalid_syntax(raw):
    result = normalize_doi(raw)
    assert result.value is None and result.reason


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42051080", "42051080"),
        ("PMID: 25659350", "25659350"),
        ("pmid28951827", "28951827"),
        ("https://pubmed.ncbi.nlm.nih.gov/32894309/", "32894309"),
    ],
)
def test_normalize_pmid(raw, expected):
    assert normalize_pmid(raw).value == expected


@pytest.mark.parametrize("raw", ["0123", "abc", "", "12.5", "1234567890"])
def test_normalize_pmid_rejects_invalid(raw):
    assert normalize_pmid(raw).value is None


@pytest.mark.parametrize(
    "raw,expected",
    [("PMC4404204", "PMC4404204"), ("pmc 11941261", "PMC11941261"), ("PMC10530730", "PMC10530730")],
)
def test_normalize_pmcid(raw, expected):
    assert normalize_pmcid(raw).value == expected


@pytest.mark.parametrize(
    "raw,id_type",
    [
        (DOI_JUDGE, "doi"),
        ("https://doi.org/10.2427/12267", "doi"),
        ("PMC4404204", "pmcid"),
        ("PMID: 25659350", "pmid"),
        ("42051080", "pmid"),
        ("总之没有标识符", "unknown"),
    ],
)
def test_normalize_identifier_detects_type(raw, id_type):
    assert normalize_identifier(raw).id_type == id_type


# ---------------------------------------------------------------------------
# 分片
# ---------------------------------------------------------------------------


def test_chunked_sizes():
    assert [len(c) for c in chunked(list(range(120)), 50)] == [50, 50, 20]
    assert [len(c) for c in chunked(list(range(50)), 50)] == [50]
    assert list(chunked([], 50)) == []


def test_chunked_rejects_non_positive_size():
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))


def test_120_dois_are_split_into_three_crossref_requests():
    """核验报告 3.3：单请求 50 个 DOI 为实测最优批量。"""
    dois = [f"10.1234/test{i:03d}" for i in range(120)]
    records = {d: {"title": f"Paper {d}", "journal": "J Test", "year": 2025} for d in dois}
    client, session, _ = make_crossref(crossref_echo(records))

    results = client.verify(dois)

    assert len(session.calls) == 3
    assert [len(session.requested_dois(i)) for i in range(3)] == [50, 50, 20]
    assert len(results) == 120
    assert all(r.status is VerificationStatus.VERIFIED for r in results.values())


def test_duplicate_dois_are_queried_once():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, session, _ = make_crossref(crossref_echo(records))

    results = client.verify([DOI_JUDGE, "https://doi.org/" + DOI_JUDGE, DOI_JUDGE.upper()])

    assert len(session.calls) == 1
    assert session.requested_dois(0) == [DOI_JUDGE]
    assert len(results) == 3
    assert all(r.status is VerificationStatus.VERIFIED for r in results.values())


def test_batch_size_above_measured_limit_is_rejected():
    with pytest.raises(ValueError, match="实测上限"):
        CrossrefClient(session=FakeSession([]), batch_size=CROSSREF_MAX_BATCH + 1)


# ---------------------------------------------------------------------------
# Crossref 请求形态
# ---------------------------------------------------------------------------


def test_request_uses_batch_filter_select_and_mailto():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, session, _ = make_crossref(crossref_echo(records))
    client.verify([DOI_JUDGE])

    params = session.calls[0]["params"]
    assert session.calls[0]["method"] == "GET"
    assert params["filter"] == f"doi:{DOI_JUDGE}"
    assert params["select"] == "DOI,title,container-title,issued,author"  # author 供 D1 作者要素
    assert params["rows"] >= CROSSREF_MAX_BATCH
    assert params["mailto"] == "ci@example.org"  # polite pool（核验报告 3.3）
    assert "mailto:ci@example.org" in session.headers["User-Agent"]


# ---------------------------------------------------------------------------
# 三态：verified
# ---------------------------------------------------------------------------


def test_verified_without_expected_metadata():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify([DOI_JUDGE])[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED
    assert record.title == TITLE_JUDGE
    assert record.journal == "npj Digital Medicine"
    assert record.year == 2025
    assert record.source == "crossref"
    assert record.reason == ""


def test_verified_with_matching_expected_metadata():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE], expected={DOI_JUDGE: ExpectedMetadata(title=TITLE_JUDGE, year=2025)}
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED


def test_year_within_tolerance_still_verified():
    """在线优先出版与正式卷期年份常差 1 年，容差内不判冲突。"""
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify([DOI_JUDGE], expected={DOI_JUDGE: ExpectedMetadata(year=2024)})[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED


def test_returned_doi_case_is_normalized_before_matching():
    records = {
        DOI_JUDGE: {
            "title": TITLE_JUDGE,
            "journal": "npj Digital Medicine",
            "year": 2025,
            "returned_doi": DOI_JUDGE.upper(),
        }
    }
    client, _, _ = make_crossref(crossref_echo(records))
    assert client.verify([DOI_JUDGE])[DOI_JUDGE].status is VerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# 三态：mismatch（伪造引用的触发前提）
# ---------------------------------------------------------------------------


def test_mismatch_when_title_conflicts():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE],
        expected={DOI_JUDGE: ExpectedMetadata(title="Mitochondrial fission drives beta cell failure")},
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.MISMATCH
    assert record.reason.startswith("metadata_explicit_conflict")
    assert "题名冲突" in record.reason


def test_mismatch_when_year_conflicts_beyond_tolerance():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE], expected={DOI_JUDGE: ExpectedMetadata(title=TITLE_JUDGE, year=2019)}
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.MISMATCH
    assert "年份冲突" in record.reason


def test_mismatch_when_identifier_not_found():
    """方案 9.3 第 1 类攻击：格式正确但不存在的 DOI。"""
    fake_doi = "10.9999/does.not.exist.2026"
    client, _, _ = make_crossref(crossref_echo({}))

    record = client.verify([fake_doi])[fake_doi]

    assert record.status is VerificationStatus.MISMATCH
    assert record.reason.startswith("identifier_not_found")


def test_mismatch_on_invalid_syntax_without_any_http_call():
    client, session, _ = make_crossref(crossref_echo({}))

    record = client.verify(["10.abc/not-a-doi"])["10.abc/not-a-doi"]

    assert record.status is VerificationStatus.MISMATCH
    assert record.reason.startswith("identifier_syntax_invalid")
    assert record.source == "syntax"
    assert session.calls == []


def test_mixed_batch_reports_each_doi_independently():
    real = DOI_JUDGE
    fake = "10.9999/fabricated"
    records = {real: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, session, _ = make_crossref(crossref_echo(records))

    results = client.verify([real, fake, "bad-doi"])

    assert results[real].status is VerificationStatus.VERIFIED
    assert results[fake].status is VerificationStatus.MISMATCH
    assert results["bad-doi"].status is VerificationStatus.MISMATCH
    assert len(session.calls) == 1  # 语法非法项不进请求


# ---------------------------------------------------------------------------
# 三态：unresolved（方案 8.3：不得判伪造）
# ---------------------------------------------------------------------------


def test_unresolved_on_transport_error():
    client, session, sleeps = make_crossref([ConnectionError("代理连接被拒")] * 4)

    record = client.verify([DOI_JUDGE])[DOI_JUDGE]

    assert record.status is VerificationStatus.UNRESOLVED
    assert record.reason.startswith("service_unavailable")
    assert "传输失败" in record.reason
    assert len(session.calls) == 4  # max_retries
    assert sleeps  # 退避确实发生


def test_unresolved_on_repeated_server_error():
    client, session, _ = make_crossref([FakeResponse(503) for _ in range(4)])

    record = client.verify([DOI_JUDGE])[DOI_JUDGE]

    assert record.status is VerificationStatus.UNRESOLVED
    assert len(session.calls) == 4


def test_unresolved_on_403_without_retry():
    """核验报告 3.3：403 = 人工封禁，重试无意义。"""
    client, session, _ = make_crossref([FakeResponse(403)])

    record = client.verify([DOI_JUDGE])[DOI_JUDGE]

    assert record.status is VerificationStatus.UNRESOLVED
    assert "403" in record.reason
    assert len(session.calls) == 1


def test_unresolved_on_non_json_body():
    client, _, _ = make_crossref([FakeResponse(200, bad_json=True)])
    assert client.verify([DOI_JUDGE])[DOI_JUDGE].status is VerificationStatus.UNRESOLVED


def test_network_failure_is_never_reported_as_forged():
    """方案 8.3 的硬规则：解析服务不可用只能标不可核验。"""
    client, _, _ = make_crossref([TimeoutError("read timeout")] * 4)
    record = client.verify([DOI_JUDGE])[DOI_JUDGE]
    assert record.status is VerificationStatus.UNRESOLVED
    assert "not_found" not in record.reason and "conflict" not in record.reason


# ---------------------------------------------------------------------------
# 限流：429 退避与响应头自适应
# ---------------------------------------------------------------------------


def test_429_then_success_backs_off_exponentially():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    ok = crossref_echo(records)
    responses = [FakeResponse(429), FakeResponse(429), None]
    sleeps: list[float] = []
    session = FakeSession(lambda call: responses.pop(0) or ok(call))
    client = CrossrefClient(mailto="ci@example.org", session=session, sleep_fn=sleeps.append)

    record = client.verify([DOI_JUDGE])[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED
    assert len(session.calls) == 3
    backoffs = [s for s in sleeps if s in (2.0, 4.0)]
    assert backoffs == [2.0, 4.0]  # 2^(attempt+1)


def test_429_honours_retry_after_header():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"})] * 4)
    client = CrossrefClient(session=session, sleep_fn=sleeps.append)

    client.verify([DOI_JUDGE])

    assert 7.0 in sleeps


def test_rate_limit_headers_are_read_and_clamped():
    """核验报告 3.3：按响应头自适应，但不突破 polite-array 实测的保守上限。"""
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    headers = {
        "x-rate-limit-limit": "50",
        "x-rate-limit-interval": "1s",
        "x-concurrency-limit": "5",
    }
    client, _, _ = make_crossref(crossref_echo(records, headers))

    client.verify([DOI_JUDGE])

    assert client.transport.observed_max_rps == 50.0
    assert client.transport.max_rps == 3.0  # 保守上限未被抬高
    assert client.transport.rate_limit_headers["x-concurrency-limit"] == "5"


def test_stricter_rate_limit_header_lowers_throughput():
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    headers = {"x-rate-limit-limit": "2", "x-rate-limit-interval": "1s"}
    client, _, _ = make_crossref(crossref_echo(records, headers))

    client.verify([DOI_JUDGE])

    assert client.transport.max_rps == 2.0


def test_interval_in_minutes_is_parsed():
    transport = HttpTransport(session=FakeSession([]), max_rps=10.0, sleep_fn=lambda _: None)
    transport._adapt({"x-rate-limit-limit": "60", "x-rate-limit-interval": "1m"})
    assert transport.observed_max_rps == 1.0
    assert transport.max_rps == 1.0


def test_transport_defaults_to_ignoring_env_proxies():
    """本机 http_proxy 指向已失效代理，默认必须直连。"""
    session = FakeSession([])
    session.trust_env = True
    HttpTransport(session=session, sleep_fn=lambda _: None)
    assert session.trust_env is True  # 注入的 session 由调用方掌控
    assert HttpTransport.__dataclass_fields__["trust_env"].default is False


# ---------------------------------------------------------------------------
# NCBI esummary
# ---------------------------------------------------------------------------


def ncbi_payload(entries: dict[str, dict]) -> dict:
    result: dict = {"uids": list(entries)}
    result.update(entries)
    return {"header": {"type": "esummary"}, "result": result}


def make_ncbi(responses, **kwargs) -> tuple[NcbiESummaryClient, FakeSession]:
    session = FakeSession(responses)
    client = NcbiESummaryClient(session=session, sleep_fn=lambda _: None, **kwargs)
    return client, session


def test_ncbi_uses_post_with_tool_email_and_api_key():
    """核验报告 3.2：tool + email 必填；>200 UID 用 POST。"""
    payload = ncbi_payload(
        {
            "25659350": {
                "uid": "25659350",
                "title": "Mitochondrial regulation of beta-cell function",
                "fulljournalname": "Molecular Aspects of Medicine",
                "pubdate": "2015 Apr",
            }
        }
    )
    client, session = make_ncbi(
        [FakeResponse(200, payload)], api_key="dummy-key", tool="mitoevidence-test", email="ci@example.org"
    )

    record = client.verify(["25659350"])["25659350"]

    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["data"]["db"] == "pubmed"
    assert call["data"]["id"] == "25659350"
    assert call["data"]["retmode"] == "json"
    assert call["data"]["tool"] == "mitoevidence-test"
    assert call["data"]["email"] == "ci@example.org"
    assert call["data"]["api_key"] == "dummy-key"
    assert record.status is VerificationStatus.VERIFIED
    assert record.year == 2015
    assert record.journal == "Molecular Aspects of Medicine"


def test_ncbi_rps_depends_on_api_key():
    with_key, _ = make_ncbi([], api_key="k", email="a@b.c")
    without_key, _ = make_ncbi([], api_key="", email="a@b.c")
    assert with_key.transport.max_rps == NCBI_RPS_WITH_KEY
    assert without_key.transport.max_rps == NCBI_RPS_WITHOUT_KEY


def test_ncbi_entry_with_error_is_mismatch():
    payload = ncbi_payload({"99999999": {"uid": "99999999", "error": "cannot get document summary"}})
    client, _ = make_ncbi([FakeResponse(200, payload)], api_key="", email="a@b.c")

    record = client.verify(["99999999"])["99999999"]

    assert record.status is VerificationStatus.MISMATCH
    assert record.reason.startswith("identifier_not_found")


def test_ncbi_title_conflict_is_mismatch():
    payload = ncbi_payload(
        {
            "25659350": {
                "uid": "25659350",
                "title": "Mitochondrial regulation of beta-cell function",
                "source": "Mol Aspects Med",
                "pubdate": "2015 Apr",
            }
        }
    )
    client, _ = make_ncbi([FakeResponse(200, payload)], api_key="", email="a@b.c")

    record = client.verify(
        ["25659350"], expected={"25659350": ExpectedMetadata(title="A totally different paper title")}
    )["25659350"]

    assert record.status is VerificationStatus.MISMATCH
    assert "题名冲突" in record.reason


def test_ncbi_transport_error_is_unresolved():
    client, _ = make_ncbi([ConnectionError("boom")] * 4, api_key="", email="a@b.c")
    assert client.verify(["25659350"])["25659350"].status is VerificationStatus.UNRESOLVED


def test_ncbi_invalid_pmid_syntax_is_mismatch_without_request():
    client, session = make_ncbi([], api_key="", email="a@b.c")
    record = client.verify(["PMID: abc"])["PMID: abc"]
    assert record.status is VerificationStatus.MISMATCH
    assert session.calls == []


def test_ncbi_batch_size_limit():
    with pytest.raises(ValueError):
        NcbiESummaryClient(session=FakeSession([]), batch_size=NCBI_MAX_BATCH + 1)


def test_ncbi_250_pmids_split_into_two_batches():
    pmids = [str(10_000_000 + i) for i in range(250)]

    def responder(call):
        ids = call["data"]["id"].split(",")
        return FakeResponse(
            200,
            ncbi_payload({p: {"uid": p, "title": f"T{p}", "source": "J", "pubdate": "2025"} for p in ids}),
        )

    client, session = make_ncbi(responder, api_key="", email="a@b.c")
    results = client.verify(pmids)

    assert len(session.calls) == 2
    assert [len(c["data"]["id"].split(",")) for c in session.calls] == [200, 50]
    assert len(results) == 250


# ---------------------------------------------------------------------------
# 作者要素：姓氏归一化（方案 8.2 D1「作者是否匹配」，口径见待澄清 G）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claimed,resolved",
    [
        ("Croxford E", "Croxford, Emma"),  # NCBI「姓+缩写」 vs Crossref「姓, 名」
        ("Emma Croxford", "Croxford"),  # 名前姓后的自由书写
        ("GARCÍA J", "Garcia, Juan"),  # 大小写 + 变音符
        ("Ångström K", "Angstrom, Kristina"),  # NFKD 可分解的变音符
        ("Møller H", "Moller, Hans"),  # NFKD 不可分解、需手工折叠的 ø
        ("Lopez-Garcia M", "López García, María"),  # 连字符复姓 vs 空格复姓
        ("Garcia M", "López-García, María"),  # 只写复姓的一段也命中
        ("O'Brien P", "OBrien, Patrick"),  # 撇号
        ("van de Schoot R", "van de Schoot, Rens"),  # 带介词的复合姓氏
        ("LI W", "Li, Wei"),  # 全大写缩写式中文罗马化姓名
        ("张伟", "张伟"),  # 中文：姓前名后
        ("伟张", "张伟"),  # 中文：名前姓后
        ("欧阳修", "修欧阳"),  # 中文复姓两种顺序
    ],
)
def test_first_author_surname_matches(claimed, resolved):
    assert first_author_matches(claimed, resolved) is True


@pytest.mark.parametrize(
    "claimed,resolved",
    [
        ("Li W", "Lin, Wei"),  # 前缀相近但姓氏不同
        ("Croxford E", "Smith, John"),
        ("Wang X", "Zhang, Xin"),  # 名首字母相同不足以通过
        ("张伟", "王强"),  # 中文姓名字均不同
        ("", "Croxford, Emma"),  # 空声称无候选 → 不算命中（由上层记 partial）
    ],
)
def test_first_author_surname_does_not_match(claimed, resolved):
    assert first_author_matches(claimed, resolved) is False


def test_surname_candidates_shapes():
    assert surname_candidates("Croxford E") == {"croxford"}
    assert surname_candidates("Croxford, Emma") == {"croxford"}
    assert surname_candidates("Lopez-García") == {"lopez", "garcia", "lopezgarcia"}
    assert {"张", "伟"} <= surname_candidates("张伟")
    assert "欧阳" in surname_candidates("欧阳修")


# ---------------------------------------------------------------------------
# 三要素组合判定：任一明确冲突才 mismatch，缺失只记 partial
# ---------------------------------------------------------------------------


RESOLVED_RECORD = {
    "title": TITLE_JUDGE,
    "year": 2025,
    "first_author": "Croxford, Emma",
    "first_author_surname": "Croxford",
}


def test_compare_metadata_all_three_elements_match():
    comparison = compare_metadata(
        ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Croxford E"),
        RESOLVED_RECORD,
    )
    assert comparison.fields == {"title": "match", "author": "match", "year": "match"}
    assert comparison.has_conflict is False
    assert comparison.detail == ""


def test_compare_metadata_author_conflict_alone_is_explicit_conflict():
    comparison = compare_metadata(
        ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Smith J"),
        RESOLVED_RECORD,
    )
    assert comparison.fields == {"title": "match", "author": "conflict", "year": "match"}
    assert comparison.has_conflict is True
    assert "第一作者姓氏冲突" in comparison.detail


@pytest.mark.parametrize(
    "expected,conflict_key",
    [
        (ExpectedMetadata(title="Another paper entirely different", first_author="Croxford E"), "题名冲突"),
        (ExpectedMetadata(year=2019, first_author="Croxford E"), "年份冲突"),
    ],
)
def test_compare_metadata_any_single_element_conflict_wins(expected, conflict_key):
    comparison = compare_metadata(expected, RESOLVED_RECORD)
    assert comparison.has_conflict is True
    assert conflict_key in comparison.detail


def test_compare_metadata_missing_fields_are_partial_not_conflict():
    """引用只声称了作者：题名与年份缺失 → partial，不判冲突。"""
    comparison = compare_metadata(ExpectedMetadata(first_author="Croxford E"), RESOLVED_RECORD)
    assert comparison.fields == {"title": "partial", "author": "match", "year": "partial"}
    assert comparison.has_conflict is False
    assert comparison.detail.startswith("metadata_partial")
    assert "题名（引用未声称）" in comparison.detail


def test_compare_metadata_source_missing_author_is_partial():
    record = {"title": TITLE_JUDGE, "year": 2025, "first_author": None}
    comparison = compare_metadata(
        ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Croxford E"), record
    )
    assert comparison.fields["author"] == "partial"
    assert comparison.has_conflict is False
    assert "数据源未返回作者" in comparison.detail


def test_compare_metadata_cross_script_names_are_partial_not_conflict():
    """声称中文姓名、数据源返回罗马化姓名：不可比 → partial，避免误判伪造。"""
    comparison = compare_metadata(
        ExpectedMetadata(first_author="张伟"),
        {"title": TITLE_JUDGE, "year": 2025, "first_author_surname": "Zhang"},
    )
    assert comparison.fields["author"] == "partial"
    assert comparison.has_conflict is False
    assert "字符集不可比" in comparison.detail


def test_compare_metadata_none_expected_returns_none():
    assert compare_metadata(None, RESOLVED_RECORD) is None


def test_match_policy_loads_from_rubric_config_single_truth_source():
    policy = MetadataMatchPolicy.from_config()
    assert policy.title_min_jaccard == 0.8
    assert policy.year_tolerance == 1
    assert policy.missing_field_is_conflict is False
    assert policy.author_enabled is True
    assert policy.author_strip_diacritics is True
    assert policy.author_strip_hyphens is True
    assert policy.author_cjk_both_orders is True
    assert policy.author_cross_script_is_conflict is False


def test_author_matching_can_be_disabled_by_policy():
    policy = MetadataMatchPolicy(author_enabled=False)
    comparison = compare_metadata(
        ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Smith J"),
        RESOLVED_RECORD,
        policy,
    )
    assert comparison.fields["author"] == "partial"
    assert comparison.has_conflict is False


# ---------------------------------------------------------------------------
# Crossref / NCBI mock 响应含作者字段
# ---------------------------------------------------------------------------

CROSSREF_AUTHORS = [
    {"given": "Emma", "family": "Croxford", "sequence": "first"},
    {"given": "John", "family": "Smith", "sequence": "additional"},
]


def test_crossref_verified_when_first_author_matches():
    records = {
        DOI_JUDGE: {
            "title": TITLE_JUDGE,
            "journal": "npj Digital Medicine",
            "year": 2025,
            "authors": CROSSREF_AUTHORS,
        }
    }
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE],
        expected={
            DOI_JUDGE: ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Croxford E")
        },
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED
    assert record.first_author == "Croxford, Emma"
    assert record.metadata_match == {"title": "match", "author": "match", "year": "match"}
    assert record.reason == ""


def test_crossref_mismatch_when_first_author_conflicts():
    """对抗用例：真实 DOI + 故意错的第一作者。"""
    records = {
        DOI_JUDGE: {
            "title": TITLE_JUDGE,
            "journal": "npj Digital Medicine",
            "year": 2025,
            "authors": CROSSREF_AUTHORS,
        }
    }
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE],
        expected={
            DOI_JUDGE: ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Vaswani A")
        },
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.MISMATCH
    assert record.metadata_match == {"title": "match", "author": "conflict", "year": "match"}
    assert record.reason.startswith("metadata_explicit_conflict")
    assert "第一作者姓氏冲突" in record.reason


def test_crossref_picks_sequence_first_not_list_order():
    records = {
        DOI_JUDGE: {
            "title": TITLE_JUDGE,
            "journal": "npj Digital Medicine",
            "year": 2025,
            "authors": [
                {"given": "John", "family": "Smith", "sequence": "additional"},
                {"given": "Emma", "family": "Croxford", "sequence": "first"},
            ],
        }
    }
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE], expected={DOI_JUDGE: ExpectedMetadata(first_author="Croxford E")}
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED
    assert record.first_author == "Croxford, Emma"
    assert record.metadata_match["author"] == "match"


def test_crossref_without_author_field_stays_verified_with_partial_note():
    """数据源未返回作者 → 缺失不算冲突，记 partial 并写进 detail。"""
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    client, _, _ = make_crossref(crossref_echo(records))

    record = client.verify(
        [DOI_JUDGE],
        expected={
            DOI_JUDGE: ExpectedMetadata(title=TITLE_JUDGE, year=2025, first_author="Croxford E")
        },
    )[DOI_JUDGE]

    assert record.status is VerificationStatus.VERIFIED
    assert record.metadata_match["author"] == "partial"
    assert record.reason.startswith("metadata_partial")
    assert "数据源未返回作者" in record.reason


def test_ncbi_verified_when_first_author_matches():
    payload = ncbi_payload(
        {
            "25659350": {
                "uid": "25659350",
                "title": "Mitochondrial regulation of beta-cell function",
                "fulljournalname": "Molecular Aspects of Medicine",
                "pubdate": "2015 Apr",
                "authors": [
                    {"name": "Wollheim CB", "authtype": "Author"},
                    {"name": "Maechler P", "authtype": "Author"},
                ],
                "sortfirstauthor": "Wollheim CB",
            }
        }
    )
    client, _ = make_ncbi([FakeResponse(200, payload)], api_key="", email="a@b.c")

    record = client.verify(
        ["25659350"],
        expected={"25659350": ExpectedMetadata(year=2015, first_author="C. B. Wollheim")},
    )["25659350"]

    assert record.status is VerificationStatus.VERIFIED
    assert record.first_author == "Wollheim CB"
    assert record.metadata_match["author"] == "match"


def test_ncbi_mismatch_when_first_author_conflicts():
    payload = ncbi_payload(
        {
            "25659350": {
                "uid": "25659350",
                "title": "Mitochondrial regulation of beta-cell function",
                "fulljournalname": "Molecular Aspects of Medicine",
                "pubdate": "2015 Apr",
                "authors": [{"name": "Wollheim CB", "authtype": "Author"}],
            }
        }
    )
    client, _ = make_ncbi([FakeResponse(200, payload)], api_key="", email="a@b.c")

    record = client.verify(
        ["25659350"], expected={"25659350": ExpectedMetadata(first_author="Hinton G")}
    )["25659350"]

    assert record.status is VerificationStatus.MISMATCH
    assert "第一作者姓氏冲突" in record.reason


def test_ncbi_collective_name_is_skipped_for_first_author():
    payload = ncbi_payload(
        {
            "32894309": {
                "uid": "32894309",
                "title": "A consortium paper",
                "source": "J Test",
                "pubdate": "2020",
                "authors": [
                    {"name": "GBD 2019 Collaborators", "authtype": "CollectiveName"},
                    {"name": "Murray CJL", "authtype": "Author"},
                ],
            }
        }
    )
    client, _ = make_ncbi([FakeResponse(200, payload)], api_key="", email="a@b.c")

    record = client.verify(
        ["32894309"], expected={"32894309": ExpectedMetadata(first_author="Murray C")}
    )["32894309"]

    assert record.status is VerificationStatus.VERIFIED
    assert record.first_author == "Murray CJL"
    assert record.metadata_match["author"] == "match"


# ---------------------------------------------------------------------------
# 题名相似度与 D1 汇总
# ---------------------------------------------------------------------------


def test_title_similarity_is_case_and_markup_insensitive():
    assert title_similarity(TITLE_JUDGE, TITLE_JUDGE.upper()) == 1.0
    assert title_similarity("Mitochondrial <i>dynamics</i> and insulin secretion", "mitochondrial dynamics and insulin secretion") == 1.0
    assert title_similarity(TITLE_JUDGE, "Mitochondrial homeostasis in beta cells") < 0.5


def test_summary_excludes_unresolved_from_the_denominator():
    """方案 8.3：unresolved 不参与 p 的计算，只触发人工复核。"""
    records = {DOI_JUDGE: {"title": TITLE_JUDGE, "journal": "npj Digital Medicine", "year": 2025}}
    good = DOI_JUDGE
    fake = "10.9999/fabricated"

    def responder(call):
        dois = [c.split("doi:", 1)[1] for c in call["params"]["filter"].split(",")]
        if "10.5555/flaky" in dois:
            return FakeResponse(503)
        return crossref_echo(records)(call)

    client = CrossrefClient(
        session=FakeSession(responder), batch_size=1, sleep_fn=lambda _: None, mailto="ci@example.org"
    )
    results = client.verify([good, fake, "10.5555/flaky"])
    summary = summarize_verifications(results.values())

    assert (summary.total, summary.verified, summary.mismatch, summary.unresolved) == (3, 1, 1, 1)
    assert summary.metadata_match_rate == pytest.approx(0.5)  # 1 / (1 + 1)
    assert summary.nonexistent_identifier_count == 1
    assert summary.has_unresolved is True
    assert summary.unresolved_ids == ["10.5555/flaky"]
    assert summary.mismatch_ids == [fake]


def test_summary_is_not_scorable_when_everything_is_unresolved():
    """全部不可核验时 p 无定义，D1 应整体记 NA，而不是当作满分。"""
    client, _, _ = make_crossref([ConnectionError("down")] * 4)
    summary = summarize_verifications(client.verify([DOI_JUDGE]).values())
    assert summary.metadata_match_rate is None
    assert summary.is_scorable is False


def test_summary_of_all_verified_is_one():
    dois = [f"10.1234/ok{i}" for i in range(5)]
    records = {d: {"title": f"T{d}", "journal": "J", "year": 2025} for d in dois}
    client, _, _ = make_crossref(crossref_echo(records))
    summary = summarize_verifications(client.verify(dois).values())
    assert summary.metadata_match_rate == 1.0
    assert summary.nonexistent_identifier_count == 0
    assert isinstance(summary, CitationCheckSummary)
