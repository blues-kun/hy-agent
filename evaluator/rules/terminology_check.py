"""D8 术语的离线、可审计初筛。

本模块只核对一个**本地、版本化**词表。它既不联网，也不宣称拥有完整的
MeSH/GO 真值：

* ``verified``：字面量命中该版本词表中的首选词或显式别名；
* ``rejected``：字面量命中该版本词表明确列出的禁用/损坏写法；
* ``unknown``：其余情况，进入人工或外部 MeSH/GO 核验队列，绝不因查不到而判错。

因此结果不能直接替代 D8 专家金标；尤其 ``verified`` 只表示
``verified_against=local_versioned_vocabulary``，不表示已被 MeSH/GO 认证。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from evaluator.schemas import StrictModel


DEFAULT_VOCABULARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "data"
    / "terminology"
    / "project_biomedical_terms_v0_1.json"
)


class TerminologyStatus(str, Enum):
    """本地词表初筛的严格三态。"""

    VERIFIED = "verified"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class MatchKind(str, Enum):
    """字面量如何命中本地词表。"""

    EXACT_PREFERRED = "exact_preferred"
    EXACT_ALIAS = "exact_alias"
    NORMALIZED_PREFERRED = "normalized_preferred"
    NORMALIZED_ALIAS = "normalized_alias"
    EXACT_DISABLED = "exact_disabled"
    NORMALIZED_DISABLED = "normalized_disabled"
    NONE = "none"


class VocabularyProvenance(StrictModel):
    """词表来源声明；必须明确是否经过外部权威库核验。"""

    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    curated_by: str = Field(min_length=1)
    reviewed_at: str | None = None
    externally_authority_verified: bool = False
    notes: str = Field(min_length=1)


class TermEntry(StrictModel):
    """一条可接受的首选词与显式别名。"""

    term_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
    preferred_label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    definition: str = Field(min_length=1)
    provenance: VocabularyProvenance
    external_identifiers: dict[str, str] = Field(default_factory=dict)


class DisabledTermEntry(StrictModel):
    """明确禁用的写法；只有命中本表才可输出 rejected。"""

    form: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    suggested_label: str | None = None
    provenance: VocabularyProvenance


class LocalTerminologyVocabulary(StrictModel):
    """本地词表文件的严格数据契约。"""

    schema_version: Literal["terminology-v1"]
    vocabulary_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str = Field(min_length=1)
    language: str = Field(min_length=2)
    scope: str = Field(min_length=1)
    authority_disclaimer: str = Field(min_length=1)
    provenance: VocabularyProvenance
    terms: list[TermEntry]
    disabled_terms: list[DisabledTermEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids_and_forms(self) -> "LocalTerminologyVocabulary":
        ids: set[str] = set()
        accepted: dict[str, str] = {}
        for entry in self.terms:
            if entry.term_id in ids:
                raise ValueError(f"重复 term_id：{entry.term_id}")
            ids.add(entry.term_id)
            forms = [entry.preferred_label, *entry.aliases]
            for form in forms:
                key = normalize_term(form)
                if not key:
                    raise ValueError(f"{entry.term_id} 含空白术语形式")
                owner = accepted.get(key)
                if owner is not None and owner != entry.term_id:
                    raise ValueError(f"术语规范化冲突：{form!r} 同时属于 {owner} 与 {entry.term_id}")
                accepted[key] = entry.term_id

        disabled: set[str] = set()
        for entry in self.disabled_terms:
            key = normalize_term(entry.form)
            if not key:
                raise ValueError("disabled_terms 含空白术语形式")
            if key in accepted:
                raise ValueError(f"禁用词与可接受词冲突：{entry.form!r}")
            if key in disabled:
                raise ValueError(f"重复禁用词：{entry.form!r}")
            disabled.add(key)
        return self


class TerminologyCheckItem(StrictModel):
    """一项待核术语；context 仅供复核，不参与字符串命中。"""

    item_id: str = Field(min_length=1)
    claimed_term: str = Field(min_length=1)
    context: str | None = None
    is_key: bool = False
    requested_authority: Literal["local", "MeSH", "GO", "human"] = "local"


class TerminologyCheckResult(StrictModel):
    """一项术语的可追溯三态结果。"""

    item_id: str
    claimed_term: str
    normalized_term: str
    status: TerminologyStatus
    match_kind: MatchKind
    matched_term_id: str | None = None
    preferred_label: str | None = None
    matched_form: str | None = None
    reason: str
    requested_authority: Literal["local", "MeSH", "GO", "human"]
    verified_against: Literal["local_versioned_vocabulary"]
    vocabulary_id: str
    vocabulary_version: str
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_authority_verified: bool
    review_required: bool
    is_key: bool
    context: str | None = None


class TerminologyReviewQueueItem(StrictModel):
    """unknown 项的外部/人工复核任务。"""

    item_id: str
    claimed_term: str
    normalized_term: str
    context: str | None = None
    requested_authority: Literal["local", "MeSH", "GO", "human"]
    reason: str
    suggested_next_step: Literal[
        "human_review", "verify_in_mesh", "verify_in_go", "human_then_external"
    ]
    vocabulary_id: str
    vocabulary_version: str
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TerminologyCheckSummary(StrictModel):
    """批量初筛报告；不计算或冒充 D8 准确率。"""

    schema_version: Literal["terminology-check-v1"] = "terminology-check-v1"
    vocabulary_id: str
    vocabulary_version: str
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total: int = Field(ge=0)
    verified: int = Field(ge=0)
    rejected: int = Field(ge=0)
    unknown: int = Field(ge=0)
    d8_accuracy_ready: Literal[False] = False
    boundary: str
    results: list[TerminologyCheckResult]
    review_queue: list[TerminologyReviewQueueItem]

    @model_validator(mode="after")
    def _counts_match(self) -> "TerminologyCheckSummary":
        if self.verified + self.rejected + self.unknown != self.total:
            raise ValueError("三态计数之和必须等于 total")
        if len(self.results) != self.total:
            raise ValueError("results 数量必须等于 total")
        if len(self.review_queue) != self.unknown:
            raise ValueError("review_queue 只应包含全部 unknown 项")
        return self


_SPACE_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")


def normalize_term(value: str) -> str:
    """保守规范化：NFKC、大小写、Unicode 横线及空白。

    不做词干化、模糊匹配、缩写推断或语义近邻，以避免把不同生物医学概念误合并。
    """

    text = unicodedata.normalize("NFKC", value or "")
    text = _DASH_RE.sub("-", text)
    return _SPACE_RE.sub(" ", text).strip().casefold()


def load_vocabulary(path: str | Path = DEFAULT_VOCABULARY_PATH) -> tuple[LocalTerminologyVocabulary, str]:
    """加载并严格校验本地 JSON 词表，同时返回原始文件 SHA-256。"""

    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"术语表不是合法 UTF-8 JSON：{source}") from exc
    vocabulary = LocalTerminologyVocabulary.model_validate(payload)
    return vocabulary, hashlib.sha256(raw).hexdigest()


class TerminologyChecker:
    """预构建索引的确定性本地术语初筛器。"""

    def __init__(self, vocabulary: LocalTerminologyVocabulary, vocabulary_sha256: str):
        self.vocabulary = vocabulary
        self.vocabulary_sha256 = vocabulary_sha256
        self._exact: dict[str, tuple[TermEntry, str, MatchKind]] = {}
        self._normalized: dict[str, tuple[TermEntry, str, MatchKind]] = {}
        self._disabled_exact: dict[str, DisabledTermEntry] = {}
        self._disabled_normalized: dict[str, DisabledTermEntry] = {}

        for entry in vocabulary.terms:
            self._exact[entry.preferred_label] = (
                entry,
                entry.preferred_label,
                MatchKind.EXACT_PREFERRED,
            )
            self._normalized[normalize_term(entry.preferred_label)] = (
                entry,
                entry.preferred_label,
                MatchKind.NORMALIZED_PREFERRED,
            )
            for alias in entry.aliases:
                self._exact[alias] = (entry, alias, MatchKind.EXACT_ALIAS)
                self._normalized[normalize_term(alias)] = (
                    entry,
                    alias,
                    MatchKind.NORMALIZED_ALIAS,
                )
        for entry in vocabulary.disabled_terms:
            self._disabled_exact[entry.form] = entry
            self._disabled_normalized[normalize_term(entry.form)] = entry

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_VOCABULARY_PATH) -> "TerminologyChecker":
        vocabulary, digest = load_vocabulary(path)
        return cls(vocabulary, digest)

    def check(self, item: TerminologyCheckItem) -> TerminologyCheckResult:
        raw = item.claimed_term.strip()
        normalized = normalize_term(raw)

        disabled = self._disabled_exact.get(raw)
        disabled_kind = MatchKind.EXACT_DISABLED
        if disabled is None:
            disabled = self._disabled_normalized.get(normalized)
            disabled_kind = MatchKind.NORMALIZED_DISABLED
        if disabled is not None:
            return self._result(
                item,
                normalized,
                TerminologyStatus.REJECTED,
                disabled_kind,
                preferred_label=disabled.suggested_label,
                matched_form=disabled.form,
                reason=f"命中本地禁用表：{disabled.reason}",
                external_verified=disabled.provenance.externally_authority_verified,
            )

        accepted = self._exact.get(raw)
        if accepted is None:
            accepted = self._normalized.get(normalized)
        if accepted is not None:
            entry, matched_form, kind = accepted
            return self._result(
                item,
                normalized,
                TerminologyStatus.VERIFIED,
                kind,
                matched_term_id=entry.term_id,
                preferred_label=entry.preferred_label,
                matched_form=matched_form,
                reason="命中本地版本化词表的首选词或显式别名；不代表已通过 MeSH/GO 核验",
                external_verified=entry.provenance.externally_authority_verified,
            )

        return self._result(
            item,
            normalized,
            TerminologyStatus.UNKNOWN,
            MatchKind.NONE,
            reason="本地词表无此字面量；unknown 不等于术语错误",
            external_verified=False,
        )

    def _result(
        self,
        item: TerminologyCheckItem,
        normalized: str,
        status: TerminologyStatus,
        match_kind: MatchKind,
        *,
        matched_term_id: str | None = None,
        preferred_label: str | None = None,
        matched_form: str | None = None,
        reason: str,
        external_verified: bool,
    ) -> TerminologyCheckResult:
        needs_review = status is TerminologyStatus.UNKNOWN or (
            item.requested_authority != "local" and not external_verified
        )
        return TerminologyCheckResult(
            item_id=item.item_id,
            claimed_term=item.claimed_term,
            normalized_term=normalized,
            status=status,
            match_kind=match_kind,
            matched_term_id=matched_term_id,
            preferred_label=preferred_label,
            matched_form=matched_form,
            reason=reason,
            requested_authority=item.requested_authority,
            verified_against="local_versioned_vocabulary",
            vocabulary_id=self.vocabulary.vocabulary_id,
            vocabulary_version=self.vocabulary.version,
            vocabulary_sha256=self.vocabulary_sha256,
            external_authority_verified=external_verified,
            review_required=needs_review,
            is_key=item.is_key,
            context=item.context,
        )

    def check_many(self, items: list[TerminologyCheckItem]) -> TerminologyCheckSummary:
        results = [self.check(item) for item in items]
        queue = [self._to_queue(result) for result in results if result.status is TerminologyStatus.UNKNOWN]
        counts = {
            status: sum(result.status is status for result in results)
            for status in TerminologyStatus
        }
        return TerminologyCheckSummary(
            vocabulary_id=self.vocabulary.vocabulary_id,
            vocabulary_version=self.vocabulary.version,
            vocabulary_sha256=self.vocabulary_sha256,
            total=len(results),
            verified=counts[TerminologyStatus.VERIFIED],
            rejected=counts[TerminologyStatus.REJECTED],
            unknown=counts[TerminologyStatus.UNKNOWN],
            boundary=(
                "本报告仅依据本地版本化术语表生成，不是完整 MeSH/GO 真值，也不直接产生 "
                "D8 准确率；unknown 必须由人工或外部权威库继续核验。"
            ),
            results=results,
            review_queue=queue,
        )

    def _to_queue(self, result: TerminologyCheckResult) -> TerminologyReviewQueueItem:
        next_step = {
            "local": "human_then_external",
            "human": "human_review",
            "MeSH": "verify_in_mesh",
            "GO": "verify_in_go",
        }[result.requested_authority]
        return TerminologyReviewQueueItem(
            item_id=result.item_id,
            claimed_term=result.claimed_term,
            normalized_term=result.normalized_term,
            context=result.context,
            requested_authority=result.requested_authority,
            reason=result.reason,
            suggested_next_step=next_step,
            vocabulary_id=result.vocabulary_id,
            vocabulary_version=result.vocabulary_version,
            vocabulary_sha256=result.vocabulary_sha256,
        )
