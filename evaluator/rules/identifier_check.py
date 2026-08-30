"""D1 引用真实性：标识符规范化与元数据核验。

三态结论（方案 8.3）：
  - verified    标识符解析成功，且（若给出期望元数据）题名与年份一致；
  - mismatch    标识符语法非法、经核验不存在，或与元数据明确冲突
                —— 这两种情形才是「伪造引用」的触发前提；
  - unresolved  网络失败、超时、限流退避耗尽、5xx/403
                —— 标记为「不可核验」，不得判伪造。

限流依据（核验报告 3.2 / 3.3，2026-08-28 实测）：
  - Crossref 已按请求类型分池，polite-array（filter 列表查询）实测仅 3 rps；
    官方文档表格写的 10 rps 已滞后。因此按响应头 x-rate-limit-limit /
    x-rate-limit-interval / x-concurrency-limit 自适应，并保留保守上限；
  - Crossref 批量最优解：/works?filter=doi:A,doi:B,… 单请求 50 个 DOI；
  - NCBI E-utilities 无 key 3 rps、有 key 10 rps；>200 UID 用 POST；
    tool 与 email 参数必填。

网络环境：本机 http_proxy/https_proxy 可能指向已失效的本地代理，因此默认
session.trust_env=False 直连（与 scripts/hy3_smoke_test.py 的 --no-proxy 同策）。
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from evaluator.schemas import IdentifierVerification, StrictModel, VerificationStatus

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CROSSREF_API = "https://api.crossref.org/works"
NCBI_ESUMMARY_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# 核验报告 3.3：单请求 50 个 DOI 实测全中（URL 1,760B）。
CROSSREF_MAX_BATCH = 50
# 核验报告 3.2：>200 UID 用 POST。
NCBI_MAX_BATCH = 200

# 核验报告 3.3：polite-array 池实测 3 rps；响应头给出更高值时也不突破该保守上限。
CROSSREF_DEFAULT_MAX_RPS = 3.0
# 核验报告 3.2：无 key 3 rps，有 key 10 rps。
NCBI_RPS_WITHOUT_KEY = 3.0
NCBI_RPS_WITH_KEY = 10.0

USER_AGENT_BASE = "MitoEvidence-Hy3-evaluator/0.1"

# 题名与年份的匹配容差：方案 8.2 D1 只写「题名、作者、年份是否匹配」，
# 未定义字符串匹配容差。以下阈值为实现选择，见 eval/rubric.md「待澄清 G」。
TITLE_MATCH_MIN_JACCARD = 0.8
YEAR_TOLERANCE = 1

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "https://www.doi.org/",
    "doi.org/",
    "doi:",
    "doi ",
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s]+$")
_PMID_RE = re.compile(r"^[1-9]\d{0,8}$")
_PMCID_RE = re.compile(r"^PMC\d{1,9}$")
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INTERVAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$")


# ---------------------------------------------------------------------------
# 规范化与语法校验
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedIdentifier:
    """规范化结果。id_type 为 doi / pmid / pmcid / unknown。"""

    raw: str
    id_type: str
    value: str | None
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.value is not None


def normalize_doi(raw: str) -> NormalizedIdentifier:
    """DOI 规范化：剥离 URL/`doi:` 前缀、去空白、统一小写。

    DOI 语法上大小写不敏感（DOI Handbook 2.4），Crossref 亦以小写作为规范形式，
    因此统一小写后比较。
    """
    text = (raw or "").strip().strip("<>").rstrip(".,;")
    lowered = text.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            lowered = text.lower()
            break
    text = text.strip().strip("<>").rstrip(".,;")
    if not text:
        return NormalizedIdentifier(raw, "doi", None, "空字符串")
    candidate = text.lower()
    if not _DOI_RE.match(candidate):
        return NormalizedIdentifier(raw, "doi", None, f"DOI 语法非法：{candidate!r}")
    return NormalizedIdentifier(raw, "doi", candidate)


def normalize_pmid(raw: str) -> NormalizedIdentifier:
    """PMID 规范化：剥离 `PMID:` 前缀与 PubMed URL，只保留纯数字。"""
    text = (raw or "").strip()
    lowered = text.lower()
    if lowered.startswith("pmid"):
        text = text[4:].lstrip(": ").strip()
    match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    text = text.strip().strip("/")
    if not _PMID_RE.match(text):
        return NormalizedIdentifier(raw, "pmid", None, f"PMID 语法非法：{text!r}")
    return NormalizedIdentifier(raw, "pmid", text)


def normalize_pmcid(raw: str) -> NormalizedIdentifier:
    """PMCID 规范化：统一为 `PMC` + 数字。"""
    text = (raw or "").strip()
    match = re.search(r"(PMC\s*\d+)", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"\s+", "", text).upper()
    if text.isdigit():
        text = f"PMC{text}"
    if not _PMCID_RE.match(text):
        return NormalizedIdentifier(raw, "pmcid", None, f"PMCID 语法非法：{text!r}")
    return NormalizedIdentifier(raw, "pmcid", text)


def normalize_identifier(raw: str) -> NormalizedIdentifier:
    """按字面形态自动判别标识符类型并规范化。"""
    text = (raw or "").strip()
    if not text:
        return NormalizedIdentifier(raw, "unknown", None, "空字符串")
    lowered = text.lower()
    if ("10." in lowered and "/" in lowered) or lowered.startswith("doi"):
        result = normalize_doi(text)
        if result.is_valid or lowered.startswith("doi") or "doi.org" in lowered:
            return result
    if "pmc" in lowered:
        return normalize_pmcid(text)
    if lowered.startswith("pmid") or "pubmed.ncbi" in lowered or text.isdigit():
        return normalize_pmid(text)
    return NormalizedIdentifier(raw, "unknown", None, f"无法判别标识符类型：{text!r}")


def chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """按固定大小分片；批量核验的分批依据。"""
    if size <= 0:
        raise ValueError("分片大小必须为正整数")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# ---------------------------------------------------------------------------
# 元数据比对
# ---------------------------------------------------------------------------


class ExpectedMetadata(StrictModel):
    """引用中声称的元数据；为 None 的字段不参与比对（记 partial，不算冲突）。"""

    title: str | None = None
    year: int | None = None
    first_author: str | None = Field(
        default=None,
        description="引用中声称的第一作者。可写 'Croxford E'、'Emma Croxford'、"
        "'Croxford, Emma' 或中文姓名；只有姓氏参与比对（方案 8.2 D1 的作者要素）",
    )


@dataclass(frozen=True)
class MetadataMatchPolicy:
    """题名/作者/年份三要素的比对口径。

    数值与开关的真值源是 configs/rubric_v0_1.yaml 的 `dimensions.D1.metadata_matching`；
    本类的字段默认值只是该节缺失时的兜底。用 `from_config()` 装载。
    """

    title_min_jaccard: float = TITLE_MATCH_MIN_JACCARD
    year_tolerance: int = YEAR_TOLERANCE
    missing_field_is_conflict: bool = False
    author_enabled: bool = True
    author_strip_diacritics: bool = True
    author_strip_hyphens: bool = True
    author_cjk_both_orders: bool = True
    author_cross_script_is_conflict: bool = False

    @classmethod
    def from_config(cls, config: Any = None) -> "MetadataMatchPolicy":
        from evaluator.rubric import default_rubric

        cfg = config or default_rubric()
        spec = cfg.dimensions["D1"].get("metadata_matching") or {}
        author = spec.get("author") or {}
        return cls(
            title_min_jaccard=float(spec.get("title_min_jaccard", TITLE_MATCH_MIN_JACCARD)),
            year_tolerance=int(spec.get("year_tolerance", YEAR_TOLERANCE)),
            missing_field_is_conflict=bool(spec.get("missing_field_is_conflict", False)),
            author_enabled=bool(author.get("enabled", True)),
            author_strip_diacritics=bool(author.get("strip_diacritics", True)),
            author_strip_hyphens=bool(author.get("strip_hyphens", True)),
            author_cjk_both_orders=bool(author.get("cjk_both_orders", True)),
            author_cross_script_is_conflict=bool(author.get("cross_script_is_conflict", False)),
        )


# -- 题名 -------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    text = _TAG_RE.sub(" ", title or "").lower()
    return " ".join(_TOKEN_RE.findall(text))


def title_similarity(a: str, b: str) -> float:
    """题名相似度（token 集合 Jaccard）。任一侧无可比 token 时返回 0。"""
    tokens_a = set(_normalize_title(a).split())
    tokens_b = set(_normalize_title(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _title_is_comparable(a: str, b: str) -> bool:
    """两侧都能切出 token 才算可比；否则只能记 partial，不能判冲突。"""
    return bool(_normalize_title(a).split()) and bool(_normalize_title(b).split())


# -- 作者姓氏归一化 -----------------------------------------------------------

# NFKD 分解无法处理的「带笔画」拉丁字母，先手工折叠。
_LATIN_FOLD_MAP = str.maketrans(
    {
        "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L",
        "ı": "i", "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH",
        "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
    }
)
# 姓名内部分隔符：空白、连字符（含短破折号）、撇号（含弯撇号）、点号。
_NAME_PART_SEP_RE = re.compile(r"[\s\u00a0\-\u2010-\u2015'\u2019\u02bc.]+")
_NAME_TOKEN_SEP_RE = re.compile(r"[\s\u00a0]+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", re.UNICODE)


def _strip_diacritics(text: str) -> str:
    """Å→A、García→Garcia、Ångström→Angstrom、Straße→Strasse。"""
    decomposed = unicodedata.normalize("NFKD", text.translate(_LATIN_FOLD_MAP))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _has_cjk(text: str) -> bool:
    return any(
        "\u3400" <= ch <= "\u4dbf" or "\u4e00" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"
        for ch in text
    )


def _has_latin(text: str) -> bool:
    return any(ch.isalpha() and ch.isascii() for ch in _strip_diacritics(text))


def name_script(name: str) -> str:
    """返回 cjk / latin / mixed / unknown，用于判断两个姓名是否可比。"""
    cjk, latin = _has_cjk(name or ""), _has_latin(name or "")
    if cjk and latin:
        return "mixed"
    if cjk:
        return "cjk"
    if latin:
        return "latin"
    return "unknown"


def _fold_part(part: str, policy: MetadataMatchPolicy) -> str:
    text = _strip_diacritics(part) if policy.author_strip_diacritics else part
    return _NON_ALNUM_RE.sub("", text.casefold())


def _expand_part(part: str, policy: MetadataMatchPolicy) -> set[str]:
    """把一个姓氏片段展开为「各组成部分 + 紧凑拼接形式」。

    Lopez-García → {lopez, garcia, lopezgarcia}，于是「Lopez」与「Lopez-García」
    可以互相命中，而「Li」与「Lin」不会。
    """
    if policy.author_strip_hyphens:
        components = [_fold_part(c, policy) for c in _NAME_PART_SEP_RE.split(part)]
    else:
        components = [_fold_part(part, policy)]
    components = [c for c in components if c]
    if not components:
        return set()
    return set(components) | {"".join(components)}


def _cjk_candidates(name: str) -> set[str]:
    """中文姓名：姓前名后与名前姓后两种顺序、单姓与复姓都生成候选。"""
    compact = re.sub(r"\s+", "", name)
    out = {compact}
    if len(compact) >= 2:
        out |= {compact[0], compact[-1]}  # 单姓
    if len(compact) >= 3:
        out |= {compact[:2], compact[-2:]}  # 复姓（欧阳、司马）
    return {c for c in out if c}


def _looks_like_initials(token: str) -> bool:
    stripped = _NAME_PART_SEP_RE.sub("", token)
    if not stripped:
        return True
    return len(stripped) <= 3 and stripped.isascii() and stripped.isalpha() and stripped.isupper()


def surname_candidates(name: str, policy: MetadataMatchPolicy | None = None) -> set[str]:
    """返回该姓名字符串中所有可能作为姓氏的归一化候选。

    覆盖的书写形式：
      - "Croxford E" / "Croxford EA"      NCBI esummary 的「姓 + 缩写」→ {croxford}
      - "Emma Croxford"                   给定名在前，姓氏位置不定 → {emma, croxford}
      - "Croxford, Emma"                  逗号形式姓氏无歧义 → {croxford}
      - "van de Schoot R"                 带介词的复合姓氏 → {van, schoot, vandeschoot}
      - "Lopez-García"                    连字符复姓 → {lopez, garcia, lopezgarcia}
      - "张伟" / "欧阳修"                  中文两种顺序、单姓与复姓
    """
    p = policy or MetadataMatchPolicy()
    raw = (name or "").strip()
    if not raw:
        return set()

    if "," in raw:  # "Family, Given"：逗号前就是姓氏，无需两端猜测
        head = raw.split(",", 1)[0].strip()
        if head:
            return _expand_part(head, p)

    if name_script(raw) == "cjk":
        return _cjk_candidates(raw) if p.author_cjk_both_orders else {re.sub(r"\s+", "", raw)}

    tokens = [t for t in _NAME_TOKEN_SEP_RE.split(raw) if t]
    substantive = [t for t in tokens if not _looks_like_initials(t)]
    if not substantive:  # 形如 "LI W"：全部像缩写时退回全部 token
        substantive = tokens
    if not substantive:
        return set()

    out = _expand_part(substantive[0], p) | _expand_part(substantive[-1], p)
    if len(substantive) > 1:  # "van de Schoot" 整体拼接
        joined = "".join(_fold_part(t, p) for t in substantive)
        if joined:
            out.add(joined)
    return {c for c in out if c}


def first_author_matches(
    claimed: str, resolved: str, policy: MetadataMatchPolicy | None = None
) -> bool:
    """第一作者姓氏是否匹配（任一候选姓氏命中即通过）。"""
    p = policy or MetadataMatchPolicy()
    claimed_set = surname_candidates(claimed, p)
    resolved_set = surname_candidates(resolved, p)
    return bool(claimed_set and resolved_set and (claimed_set & resolved_set))


# -- 三要素比对 ---------------------------------------------------------------


class MetadataComparison(StrictModel):
    """题名/作者/年份三要素的比对结果。

    方案 8.3 只有「与元数据明确冲突」才触发伪造规则，因此：
      - match     两侧都有值且一致；
      - conflict  两侧都有值且明确不一致 —— 唯一能把状态判成 mismatch 的结果；
      - partial   任一侧缺失、或两侧不可比（如跨字符集姓名），不算冲突。
    """

    title: str = "partial"
    author: str = "partial"
    year: str = "partial"
    conflicts: list[str] = Field(default_factory=list)
    partials: list[str] = Field(default_factory=list)

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def fields(self) -> dict[str, str]:
        return {"title": self.title, "author": self.author, "year": self.year}

    @property
    def detail(self) -> str:
        """写入 IdentifierVerification.reason 的说明文本。"""
        if self.has_conflict:
            text = "metadata_explicit_conflict: " + "；".join(self.conflicts)
            if self.partials:
                text += f"；未比对：{'、'.join(self.partials)}"
            return text
        if self.partials:
            return "metadata_partial: 未比对 " + "、".join(self.partials)
        return ""


def compare_metadata(
    expected: ExpectedMetadata | None,
    record: Mapping[str, Any],
    policy: MetadataMatchPolicy | None = None,
) -> MetadataComparison | None:
    """比对声称元数据与解析到的元数据。未给出期望元数据时返回 None（无从比对）。"""
    if expected is None:
        return None
    p = policy or MetadataMatchPolicy()
    result = MetadataComparison()

    # 题名
    title = record.get("title")
    if not expected.title:
        result.partials.append("题名（引用未声称）")
    elif not title:
        result.partials.append("题名（数据源未返回）")
    elif not _title_is_comparable(expected.title, title):
        result.partials.append("题名（两侧均无可比 token）")
    else:
        similarity = title_similarity(expected.title, title)
        if similarity < p.title_min_jaccard:
            result.title = "conflict"
            result.conflicts.append(
                f"题名冲突（Jaccard {similarity:.2f} < {p.title_min_jaccard}）："
                f"声称 {expected.title!r} vs 实际 {title!r}"
            )
        else:
            result.title = "match"

    # 作者：只比第一作者姓氏
    resolved_author = record.get("first_author_surname") or record.get("first_author")
    if not p.author_enabled:
        result.partials.append("作者（比对已在配置中关闭）")
    elif not expected.first_author:
        result.partials.append("作者（引用未声称第一作者）")
    elif not resolved_author:
        result.partials.append("作者（数据源未返回作者）")
    else:
        claimed_script = name_script(expected.first_author)
        resolved_script = name_script(str(resolved_author))
        incomparable_scripts = (
            claimed_script in ("cjk", "latin")
            and resolved_script in ("cjk", "latin")
            and claimed_script != resolved_script
        )
        if incomparable_scripts and not p.author_cross_script_is_conflict:
            result.partials.append(
                f"作者（字符集不可比：声称 {claimed_script}、实际 {resolved_script}，"
                f"疑为罗马化差异）"
            )
        elif first_author_matches(expected.first_author, str(resolved_author), p):
            result.author = "match"
        else:
            result.author = "conflict"
            result.conflicts.append(
                f"第一作者姓氏冲突：声称 {expected.first_author!r} vs 实际 {resolved_author!r}"
            )

    # 年份
    year = record.get("year")
    if not expected.year:
        result.partials.append("年份（引用未声称）")
    elif not year:
        result.partials.append("年份（数据源未返回）")
    elif abs(int(expected.year) - int(year)) > p.year_tolerance:
        result.year = "conflict"
        result.conflicts.append(f"年份冲突：声称 {expected.year} vs 实际 {year}")
    else:
        result.year = "match"

    if p.missing_field_is_conflict and result.partials:
        result.conflicts.extend(f"缺失字段按冲突处理：{item}" for item in result.partials)
    return result


# ---------------------------------------------------------------------------
# 限流与重试
# ---------------------------------------------------------------------------


def _parse_interval_seconds(value: str | None) -> float | None:
    if not value:
        return None
    match = _INTERVAL_RE.match(value)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    return amount * {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]


@dataclass
class HttpTransport:
    """带节流、响应头自适应与 429 指数退避的 HTTP 执行器。

    session 与 sleep_fn 均可注入：离线测试传入 mock，不产生真实请求与真实等待。
    """

    session: Any = None
    max_rps: float = CROSSREF_DEFAULT_MAX_RPS
    max_retries: int = 4
    timeout: float = 30.0
    sleep_fn: Callable[[float], None] = time.sleep
    trust_env: bool = False
    user_agent: str = USER_AGENT_BASE
    _last_call: float = field(default=0.0, init=False)
    rate_limit_headers: dict[str, str] = field(default_factory=dict, init=False)
    observed_max_rps: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.session is None:
            import requests

            self.session = requests.Session()
            # 本机 http_proxy 可能指向已失效代理：默认忽略环境变量直连。
            self.session.trust_env = self.trust_env
        headers = getattr(self.session, "headers", None)
        if hasattr(headers, "update"):
            headers.update({"User-Agent": self.user_agent})

    # -- 节流 ---------------------------------------------------------------

    def _throttle(self) -> None:
        interval = 1.0 / self.max_rps if self.max_rps > 0 else 0.0
        wait = interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self.sleep_fn(wait)
        self._last_call = time.monotonic()

    def _adapt(self, headers: Mapping[str, str]) -> None:
        """按响应头动态自适应限速（核验报告 3.3：不写死常量）。"""
        lowered = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
        for key in ("x-rate-limit-limit", "x-rate-limit-interval", "x-concurrency-limit"):
            if key in lowered:
                self.rate_limit_headers[key] = lowered[key]
        limit = lowered.get("x-rate-limit-limit")
        interval = _parse_interval_seconds(lowered.get("x-rate-limit-interval")) or 1.0
        if limit and limit.replace(".", "", 1).isdigit() and interval > 0:
            header_rps = float(limit) / interval
            self.observed_max_rps = header_rps
            # 响应头给出上限，实测池更严：取两者较小值。
            self.max_rps = min(self.max_rps, header_rps)

    # -- 请求 ---------------------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> tuple[int, Any, str]:
        """返回 (status_code, 解析后的 JSON 或 None, 错误说明)。

        错误说明非空即表示本次请求无法给出可信结论（应记 unresolved）。
        """
        last_error = "未发起请求"
        for attempt in range(self.max_retries):
            is_last = attempt == self.max_retries - 1
            self._throttle()
            try:
                response = self.session.request(
                    method, url, params=params, data=data, timeout=self.timeout
                )
            except Exception as exc:  # noqa: BLE001 —— 传输层异常（网络/DNS/代理/超时）
                # 必须宽catch：方案 8.3 要求任何解析失败都归入「不可核验」而非「伪造」，
                # 漏掉一种异常类型就会把网络故障误判成伪造引用。
                last_error = f"传输失败（{type(exc).__name__}: {exc}）"
                if not is_last:
                    self.sleep_fn(min(2.0**attempt, 8.0))
                continue

            status = int(getattr(response, "status_code", 0))
            self._adapt(getattr(response, "headers", {}) or {})

            if status == 429:
                retry_after = (getattr(response, "headers", {}) or {}).get("Retry-After")
                delay = float(retry_after) if str(retry_after or "").isdigit() else 2.0 ** (
                    attempt + 1
                )
                last_error = f"HTTP 429 限流，退避 {delay:.1f}s"
                if not is_last:
                    self.sleep_fn(delay)
                continue
            if status == 403:
                # 核验报告 3.3：403 = 人工封禁，重试无意义。
                return status, None, "HTTP 403（疑似人工封禁），不重试"
            if 500 <= status < 600:
                last_error = f"HTTP {status} 服务端错误"
                if not is_last:
                    self.sleep_fn(min(2.0**attempt, 8.0))
                continue
            try:
                return status, response.json(), ""
            except Exception as exc:  # noqa: BLE001 —— 同上：解析失败也只能记不可核验
                return status, None, f"响应非 JSON（{type(exc).__name__}: {exc}）"
        return -1, None, f"重试 {self.max_retries} 次后仍失败：{last_error}"


# ---------------------------------------------------------------------------
# Crossref 客户端
# ---------------------------------------------------------------------------


class CrossrefClient:
    """Crossref 批量 DOI 核验。

    请求形态（核验报告 3.3 实测最优解，author 字段为方案 8.2 D1 作者要素所需）：
        /works?filter=doi:A,doi:B,…&select=DOI,title,container-title,issued,author
               &rows=100&mailto=<env>
    """

    SELECT_FIELDS = "DOI,title,container-title,issued,author"

    def __init__(
        self,
        mailto: str | None = None,
        session: Any = None,
        transport: HttpTransport | None = None,
        batch_size: int = CROSSREF_MAX_BATCH,
        rows: int = 100,
        max_rps: float = CROSSREF_DEFAULT_MAX_RPS,
        sleep_fn: Callable[[float], None] = time.sleep,
        match_policy: MetadataMatchPolicy | None = None,
    ):
        if batch_size < 1 or batch_size > CROSSREF_MAX_BATCH:
            raise ValueError(f"batch_size 必须在 1..{CROSSREF_MAX_BATCH} 之间（实测上限）")
        self.match_policy = match_policy or MetadataMatchPolicy.from_config()
        self.mailto = mailto if mailto is not None else os.environ.get("CROSSREF_MAILTO", "")
        self.batch_size = batch_size
        self.rows = max(rows, batch_size)
        # polite pool：User-Agent 带 mailto 与 query 参数 ?mailto= 实测均生效。
        user_agent = (
            f"{USER_AGENT_BASE} (mailto:{self.mailto})" if self.mailto else USER_AGENT_BASE
        )
        self.transport = transport or HttpTransport(
            session=session, max_rps=max_rps, sleep_fn=sleep_fn, user_agent=user_agent
        )

    def _params(self, dois: Sequence[str]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "filter": ",".join(f"doi:{doi}" for doi in dois),
            "select": self.SELECT_FIELDS,
            "rows": self.rows,
        }
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    @staticmethod
    def _parse_first_author(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
        """返回 (展示用姓名, 权威姓氏)。优先取 sequence=first 的条目。"""
        authors = [a for a in (item.get("author") or []) if isinstance(a, Mapping)]
        if not authors:
            return None, None
        entry = next(
            (a for a in authors if str(a.get("sequence") or "").lower() == "first"),
            authors[0],
        )
        family = str(entry.get("family") or "").strip() or None
        given = str(entry.get("given") or "").strip()
        # 机构作者（consortium）只有 name 字段
        name = str(entry.get("name") or "").strip() or None
        if family:
            display = f"{family}, {given}" if given else family
        else:
            display = name or (given or None)
        return display, family

    @classmethod
    def _parse_item(cls, item: Mapping[str, Any]) -> dict[str, Any]:
        titles = item.get("title") or []
        journals = item.get("container-title") or []
        parts = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
        year = int(parts[0]) if parts and str(parts[0]).isdigit() else None
        first_author, first_author_surname = cls._parse_first_author(item)
        return {
            "doi": str(item.get("DOI", "")).lower(),
            "title": titles[0] if titles else None,
            "journal": journals[0] if journals else None,
            "year": year,
            "first_author": first_author,
            "first_author_surname": first_author_surname,
        }

    def fetch_batch(self, dois: Sequence[str]) -> tuple[dict[str, dict[str, Any]], str]:
        """取一批 DOI 的元数据。返回 (已解析映射, 错误说明)。"""
        if len(dois) > CROSSREF_MAX_BATCH:
            raise ValueError(f"单批 DOI 不得超过 {CROSSREF_MAX_BATCH} 个")
        status, body, error = self.transport.request("GET", CROSSREF_API, params=self._params(dois))
        if error:
            return {}, error
        if status != 200:
            return {}, f"HTTP {status}"
        items = ((body or {}).get("message") or {}).get("items") or []
        parsed = {}
        for item in items:
            record = self._parse_item(item)
            if record["doi"]:
                parsed[record["doi"]] = record
        return parsed, ""

    def verify(
        self,
        raw_dois: Iterable[str],
        expected: Mapping[str, ExpectedMetadata] | None = None,
    ) -> dict[str, IdentifierVerification]:
        """批量核验 DOI，返回以输入原串为键的三态结论。"""
        raw_list = list(raw_dois)
        expected = expected or {}
        results: dict[str, IdentifierVerification] = {}
        pending: dict[str, str] = {}  # normalized -> raw（同一 DOI 重复出现时取首个）

        for raw in raw_list:
            norm = normalize_doi(raw)
            if not norm.is_valid:
                results[raw] = IdentifierVerification(
                    input_id=raw,
                    id_type="doi",
                    status=VerificationStatus.MISMATCH,
                    reason=f"identifier_syntax_invalid: {norm.reason}",
                    source="syntax",
                )
                continue
            pending.setdefault(norm.value, raw)

        for batch in chunked(list(pending), self.batch_size):
            parsed, error = self.fetch_batch(batch)
            for doi in batch:
                raw = pending[doi]
                if error:
                    results[raw] = IdentifierVerification(
                        input_id=raw,
                        normalized_id=doi,
                        id_type="doi",
                        status=VerificationStatus.UNRESOLVED,
                        reason=f"service_unavailable: {error}",
                        source="crossref",
                    )
                    continue
                record = parsed.get(doi)
                if record is None:
                    results[raw] = IdentifierVerification(
                        input_id=raw,
                        normalized_id=doi,
                        id_type="doi",
                        status=VerificationStatus.MISMATCH,
                        reason="identifier_not_found: Crossref 批量查询未返回该 DOI",
                        source="crossref",
                    )
                    continue
                comparison = compare_metadata(
                    expected.get(raw) or expected.get(doi), record, self.match_policy
                )
                has_conflict = comparison is not None and comparison.has_conflict
                results[raw] = IdentifierVerification(
                    input_id=raw,
                    normalized_id=doi,
                    id_type="doi",
                    status=(
                        VerificationStatus.MISMATCH
                        if has_conflict
                        else VerificationStatus.VERIFIED
                    ),
                    title=record["title"],
                    journal=record["journal"],
                    year=record["year"],
                    first_author=record.get("first_author"),
                    metadata_match=comparison.fields if comparison else {},
                    reason=comparison.detail if comparison else "",
                    source="crossref",
                )

        # 保持与输入等长：重复 DOI 的后续出现复用首个结论。
        for raw in raw_list:
            if raw not in results:
                norm = normalize_doi(raw)
                first_raw = pending.get(norm.value or "")
                if first_raw and first_raw in results:
                    results[raw] = results[first_raw].model_copy(update={"input_id": raw})
        return results


# ---------------------------------------------------------------------------
# NCBI E-utilities 客户端
# ---------------------------------------------------------------------------


class NcbiESummaryClient:
    """NCBI esummary 批量核验（POST）。

    核验报告 3.2：tool 与 email 参数必填且需另发邮件到 eutilities@ncbi.nlm.nih.gov
    登记；无 key 3 rps、有 key 10 rps；>200 UID 用 POST。
    """

    def __init__(
        self,
        api_key: str | None = None,
        tool: str | None = None,
        email: str | None = None,
        db: str = "pubmed",
        session: Any = None,
        transport: HttpTransport | None = None,
        batch_size: int = NCBI_MAX_BATCH,
        sleep_fn: Callable[[float], None] = time.sleep,
        match_policy: MetadataMatchPolicy | None = None,
    ):
        if batch_size < 1 or batch_size > NCBI_MAX_BATCH:
            raise ValueError(f"batch_size 必须在 1..{NCBI_MAX_BATCH} 之间")
        self.match_policy = match_policy or MetadataMatchPolicy.from_config()
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY", "")
        self.tool = tool if tool is not None else os.environ.get("NCBI_TOOL", "mitoevidence-hy3")
        self.email = email if email is not None else os.environ.get("NCBI_EMAIL", "")
        self.db = db
        self.batch_size = batch_size
        max_rps = NCBI_RPS_WITH_KEY if self.api_key else NCBI_RPS_WITHOUT_KEY
        self.transport = transport or HttpTransport(
            session=session, max_rps=max_rps, sleep_fn=sleep_fn
        )

    def _payload(self, pmids: Sequence[str]) -> dict[str, str]:
        payload = {
            "db": self.db,
            "id": ",".join(pmids),
            "retmode": "json",
            "tool": self.tool,
        }
        if self.email:
            payload["email"] = self.email
        if self.api_key:
            payload["api_key"] = self.api_key
        return payload

    @staticmethod
    def _parse_year(entry: Mapping[str, Any]) -> int | None:
        for key in ("pubdate", "epubdate", "sortpubdate"):
            match = re.search(r"(19|20|21)\d{2}", str(entry.get(key) or ""))
            if match:
                return int(match.group(0))
        return None

    @staticmethod
    def _parse_first_author(entry: Mapping[str, Any]) -> str | None:
        """esummary 的 authors[].name 形如 'Croxford E'（姓在前、缩写在后）。"""
        for author in entry.get("authors") or []:
            if isinstance(author, Mapping):
                if str(author.get("authtype") or "Author").lower() != "author":
                    continue  # 跳过 CollectiveName 等非个人作者
                name = str(author.get("name") or "").strip()
            else:
                name = str(author).strip()
            if name:
                return name
        fallback = str(entry.get("sortfirstauthor") or "").strip()
        return fallback or None

    def fetch_batch(self, pmids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], str]:
        if len(pmids) > NCBI_MAX_BATCH:
            raise ValueError(f"单批 UID 不得超过 {NCBI_MAX_BATCH} 个")
        status, body, error = self.transport.request(
            "POST", NCBI_ESUMMARY_API, data=self._payload(pmids)
        )
        if error:
            return {}, error
        if status != 200:
            return {}, f"HTTP {status}"
        if isinstance(body, Mapping) and body.get("error") and not body.get("result"):
            return {}, f"NCBI 返回错误：{body['error']}"
        result = (body or {}).get("result") or {}
        parsed: dict[str, dict[str, Any]] = {}
        for uid in result.get("uids") or []:
            entry = result.get(str(uid)) or {}
            if entry.get("error"):
                continue
            parsed[str(uid)] = {
                "title": entry.get("title"),
                "journal": entry.get("fulljournalname") or entry.get("source"),
                "year": self._parse_year(entry),
                "first_author": self._parse_first_author(entry),
            }
        return parsed, ""

    def verify(
        self,
        raw_pmids: Iterable[str],
        expected: Mapping[str, ExpectedMetadata] | None = None,
    ) -> dict[str, IdentifierVerification]:
        raw_list = list(raw_pmids)
        expected = expected or {}
        results: dict[str, IdentifierVerification] = {}
        pending: dict[str, str] = {}

        for raw in raw_list:
            norm = normalize_pmid(raw)
            if not norm.is_valid:
                results[raw] = IdentifierVerification(
                    input_id=raw,
                    id_type="pmid",
                    status=VerificationStatus.MISMATCH,
                    reason=f"identifier_syntax_invalid: {norm.reason}",
                    source="syntax",
                )
                continue
            pending.setdefault(norm.value, raw)

        for batch in chunked(list(pending), self.batch_size):
            parsed, error = self.fetch_batch(batch)
            for pmid in batch:
                raw = pending[pmid]
                if error:
                    results[raw] = IdentifierVerification(
                        input_id=raw,
                        normalized_id=pmid,
                        id_type="pmid",
                        status=VerificationStatus.UNRESOLVED,
                        reason=f"service_unavailable: {error}",
                        source="ncbi_esummary",
                    )
                    continue
                record = parsed.get(pmid)
                if record is None:
                    results[raw] = IdentifierVerification(
                        input_id=raw,
                        normalized_id=pmid,
                        id_type="pmid",
                        status=VerificationStatus.MISMATCH,
                        reason="identifier_not_found: esummary 未返回该 UID 或返回 error",
                        source="ncbi_esummary",
                    )
                    continue
                comparison = compare_metadata(
                    expected.get(raw) or expected.get(pmid), record, self.match_policy
                )
                has_conflict = comparison is not None and comparison.has_conflict
                results[raw] = IdentifierVerification(
                    input_id=raw,
                    normalized_id=pmid,
                    id_type="pmid",
                    status=(
                        VerificationStatus.MISMATCH
                        if has_conflict
                        else VerificationStatus.VERIFIED
                    ),
                    title=record["title"],
                    journal=record["journal"],
                    year=record["year"],
                    first_author=record.get("first_author"),
                    metadata_match=comparison.fields if comparison else {},
                    reason=comparison.detail if comparison else "",
                    source="ncbi_esummary",
                )

        for raw in raw_list:
            if raw not in results:
                norm = normalize_pmid(raw)
                first_raw = pending.get(norm.value or "")
                if first_raw and first_raw in results:
                    results[raw] = results[first_raw].model_copy(update={"input_id": raw})
        return results


# ---------------------------------------------------------------------------
# D1 指标汇总
# ---------------------------------------------------------------------------


class CitationCheckSummary(StrictModel):
    """把三态核验结论汇总为 D1 的输入。"""

    total: int
    verified: int
    mismatch: int
    unresolved: int
    metadata_match_rate: float | None = Field(
        description="D1 的 p；分母只含可核验的引用（verified + mismatch），unresolved 不参与。"
        "无可核验引用时为 None，此时 D1 应整体记 NA 而不是当作满分。"
        "见 eval/rubric.md「待澄清 H」"
    )
    nonexistent_identifier_count: int = Field(
        description="经核验不存在或与元数据明确冲突的标识符数；D1 0 分事件上限的输入"
    )
    unresolved_ids: list[str] = Field(default_factory=list)
    mismatch_ids: list[str] = Field(default_factory=list)

    @property
    def has_unresolved(self) -> bool:
        return self.unresolved > 0

    @property
    def is_scorable(self) -> bool:
        return self.metadata_match_rate is not None


def summarize_verifications(
    verifications: Iterable[IdentifierVerification],
) -> CitationCheckSummary:
    """汇总核验结论。

    方案 8.3：unresolved 是「不可核验」，不得判伪造，因此不计入 p 的分子也不计入
    分母；它单独触发「必须人工复核」。
    """
    records = list(verifications)
    verified = [r for r in records if r.status is VerificationStatus.VERIFIED]
    mismatch = [r for r in records if r.status is VerificationStatus.MISMATCH]
    unresolved = [r for r in records if r.status is VerificationStatus.UNRESOLVED]
    checkable = len(verified) + len(mismatch)
    return CitationCheckSummary(
        total=len(records),
        verified=len(verified),
        mismatch=len(mismatch),
        unresolved=len(unresolved),
        metadata_match_rate=(len(verified) / checkable) if checkable else None,
        nonexistent_identifier_count=len(mismatch),
        unresolved_ids=[r.input_id for r in unresolved],
        mismatch_ids=[r.input_id for r in mismatch],
    )
