"""应用层结构化契约。

所有模型输出都先经过本地 Pydantic 校验。这里刻意不复用“金标”命名：应用运行
产物是待评回答，只有双人盲标和裁决后的 ``QuestionGold`` 才能称为金标准。
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from evaluator.schemas import Answerability, ExperimentConditions, StrictModel


class RunKind(str, Enum):
    HY3 = "hy3"
    OFFLINE_SMOKE = "offline_smoke"


class ReviewRequest(StrictModel):
    """一次用户研究问题。source_pmids 是检索约束，不是答案证据。"""

    question_id: str
    question: str
    scope: str = ""
    source_pmids: list[str] = Field(default_factory=list)
    answerability_hint: Answerability | None = None
    prohibited_inferences: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _question_not_blank(self) -> "ReviewRequest":
        if not self.question.strip():
            raise ValueError("question 不能为空")
        return self


class SearchPlan(StrictModel):
    """Hy3 生成的检索计划；查询词要求为可检索的英文生物医学短语。"""

    queries: list[str] = Field(min_length=1, max_length=6)
    source_pmids: list[str] = Field(default_factory=list)
    rationale: str
    answerability_hint: Answerability

    @model_validator(mode="after")
    def _queries_nonblank(self) -> "SearchPlan":
        cleaned = [q.strip() for q in self.queries if q.strip()]
        if not cleaned:
            raise ValueError("SearchPlan.queries 不能全为空")
        # StrictModel 开启了 validate_assignment；在 after validator 内通过普通赋值会
        # 再次触发本 validator 并无限递归。这里仅写回已经完成校验的规范化值。
        object.__setattr__(self, "queries", list(dict.fromkeys(cleaned)))
        object.__setattr__(
            self,
            "source_pmids",
            list(dict.fromkeys(p.strip() for p in self.source_pmids if p.strip())),
        )
        return self


class CorpusPassage(StrictModel):
    """冻结 OA 综述 XML 中的可重定位段落。"""

    passage_id: str
    paper_id: str
    pmid: str
    pmcid: str
    title: str | None = None
    section: str | None = None
    text: str
    anchor_exact: str = Field(
        default="",
        description="用于TextQuoteSelector重定位的段内原文；为空时回退到完整text",
    )
    prefix: str = ""
    postfix: str = ""
    score: float = 0.0
    source_path: str
    source_sha256: str


class GeneratedClaim(StrictModel):
    """应用输出中的一个原子主张及其证据段落引用。"""

    claim_id: str
    text: str
    is_core: bool = False
    conditions: ExperimentConditions = Field(default_factory=ExperimentConditions)
    evidence_passage_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _claim_not_blank(self) -> "GeneratedClaim":
        if not self.text.strip():
            raise ValueError("GeneratedClaim.text 不能为空")
        object.__setattr__(
            self,
            "evidence_passage_ids",
            list(dict.fromkeys(self.evidence_passage_ids)),
        )
        return self


class GeneratedReview(StrictModel):
    """Hy3 的结构化综述输出。"""

    answerability: Answerability
    answer: str
    claims: list[GeneratedClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answer_not_blank(self) -> "GeneratedReview":
        if not self.answer.strip():
            raise ValueError("GeneratedReview.answer 不能为空")
        if self.answerability in (Answerability.INSUFFICIENT, Answerability.OUT_OF_SCOPE):
            # 拒答可以没有科学主张；若给出主张，仍会在流水线里强制检查证据引用。
            return self
        if not self.claims:
            raise ValueError("answerable/partial 输出至少需要一个原子主张")
        return self


class ModelCallAudit(StrictModel):
    stage: str
    provider: str = ""
    model: str = ""
    endpoint_origin: str = ""
    prompt_sha256: str = ""
    schema_sha256: str = ""
    config_sha256: str = ""
    response_sha256: str = ""
    temperature: float | None = None
    reasoning_effort: str = ""
    max_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    parse_source: str = ""


class AnchorCheck(StrictModel):
    """一次 EvidenceSpan→冻结 XML 的实际重定位结果。"""

    span_id: str
    source_path: str
    source_sha256: str
    status: Literal["found", "ambiguous", "not_found", "error"]
    candidate_count: int = 0
    readable_position: str | None = None
    reason: str = ""


class ReviewRunArtifact(StrictModel):
    application_version: str = "mitoevidence-hy3-v0.3.0"
    evidence_manifest_path: str
    evidence_manifest_sha256: str
    request: ReviewRequest
    plan: SearchPlan
    passages: list[CorpusPassage]
    review: GeneratedReview
    model_calls: list[ModelCallAudit] = Field(default_factory=list)
    anchor_checks: list[AnchorCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_kind: RunKind
    formal_status: str = "engineering_run_pending_human_gold"
