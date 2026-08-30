"""MitoEvidence-Hy3 评估器数据契约。

与方案对齐关系：
  - 8.1 评估对象：原子主张 / 核心主张 / 五值判定 / source_access 规则；
  - 6.1 条件化 Claim 证据图：实验条件槽位与 effect_direction 取值域；
  - 9.2 金标准构建：QuestionGold 的字段清单；
  - 8.2 / 8.3：EvaluationResult 的维度分、致命错误与发布决策字段。

证据定位不使用字符 offset：Europe PMC Annotations API 无字符偏移，定位只能靠
W3C TextQuoteSelector 式的 prefix/exact/postfix 三段文本锚点加 section 标签
（核验报告 3.1 与 3.4 第 2 条）。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """所有契约的公共基类。

    方案 5.3：模型侧结构约束不视为安全边界，一切结构化输出由本地 Pydantic
    重新校验；因此这里禁止未声明字段，不合规输出直接拒绝而不是静默吞掉。
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class SourceAccess(str, Enum):
    """来源可用性（方案 6.1 字段 source_access、8.1 来源可用性规则）。"""

    FULLTEXT = "fulltext"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"


class SupportVerdict(str, Enum):
    """方案 8.1 冻结的五值判定。"""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class Answerability(str, Enum):
    """方案 9.2 / 8.2 D6 决策表标签。"""

    ANSWERABLE = "answerable"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    OUT_OF_SCOPE = "out_of_scope"


class EffectDirection(str, Enum):
    """方案 6.1 目标字段 effect_direction 的取值域。"""

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_EFFECT = "no_effect"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ConditionSlot(str, Enum):
    """方案 8.2 D4 逐项检查的实验条件槽位。"""

    SPECIES = "species"
    CELL_TYPE = "cell_type"
    PERTURBATION = "perturbation"
    DOSE = "dose"
    TIME = "time"
    METHOD = "method"
    OUTCOME = "outcome"
    EFFECT_DIRECTION = "effect_direction"


class ReleaseDecision(str, Enum):
    """方案 8.3 发布决策三档。"""

    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class VerificationStatus(str, Enum):
    """标识符核验三态（方案 8.3：网络失败只能标不可核验，不得判伪造）。"""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# 证据定位
# ---------------------------------------------------------------------------


class TextAnchor(StrictModel):
    """W3C TextQuoteSelector 式的三段文本锚点。

    核验报告 3.1：Europe PMC Annotations API 无字符 offset，定位只能靠
    prefix/exact/postfix 在全文 XML 中重定位。
    """

    prefix: str = Field(default="", description="exact 之前的上下文文本，用于消歧")
    exact: str = Field(description="被引用的原文片段本身")
    postfix: str = Field(default="", description="exact 之后的上下文文本，用于消歧")

    @model_validator(mode="after")
    def _exact_not_blank(self) -> TextAnchor:
        if not self.exact.strip():
            raise ValueError("TextAnchor.exact 不能为空白：证据定位必须给出原文片段")
        return self


class EvidenceSpan(StrictModel):
    """一条可核验的原文证据片段。"""

    span_id: str = Field(description="片段稳定标识，供 JudgeVerdict.evidence_span_refs 引用")
    paper_id: str = Field(description="本地论文标识")
    doi_or_pmid: str = Field(description="稳定外部标识：DOI、PMID 或 PMCID")
    section: str | None = Field(
        default=None,
        description="Europe PMC 受控 section 标签，如 Results / Introduction；"
        "用于区分结果段原创断言与引言背景转述（核验报告 3.1）",
    )
    page_or_figure: str | None = Field(default=None, description="页码或图表号，如 p.5 / Fig. 3B")
    anchor: TextAnchor | None = Field(
        default=None, description="文本锚点；metadata_only 时可为 None"
    )
    source_access: SourceAccess = Field(description="来源可用性，决定该片段能支撑什么级别的主张")

    @model_validator(mode="after")
    def _anchor_required_unless_metadata_only(self) -> EvidenceSpan:
        if self.source_access is SourceAccess.METADATA_ONLY:
            return self
        if self.anchor is None:
            raise ValueError(
                f"source_access={self.source_access.value} 的证据片段必须提供 anchor 文本锚点"
            )
        return self

    def can_support_scientific_claim(self) -> bool:
        """方案 8.1：metadata_only 只能证明论文存在，不能支撑科学主张。"""
        return self.source_access is not SourceAccess.METADATA_ONLY

    def can_support_slot(self, slot: ConditionSlot) -> bool:
        """方案 8.1：abstract_only 不能补写摘要未报告的剂量、时间或方法细节。"""
        if self.source_access is SourceAccess.METADATA_ONLY:
            return False
        if self.source_access is SourceAccess.ABSTRACT_ONLY:
            return slot not in (ConditionSlot.DOSE, ConditionSlot.TIME, ConditionSlot.METHOD)
        return True


# ---------------------------------------------------------------------------
# 主张
# ---------------------------------------------------------------------------


class ExperimentConditions(StrictModel):
    """方案 8.2 D4 的八个条件槽位；未报告或题目不要求时为 None（记 NA）。"""

    species: str | None = None
    cell_type: str | None = None
    perturbation: str | None = None
    dose: str | None = None
    time: str | None = None
    method: str | None = None
    outcome: str | None = None
    effect_direction: EffectDirection | None = None

    def filled_slots(self) -> dict[ConditionSlot, Any]:
        """返回已填写的槽位；None 表示 NA，不进入 D4 分母。"""
        return {
            slot: value
            for slot in ConditionSlot
            if (value := getattr(self, slot.value)) is not None
        }


class Citation(StrictModel):
    """主张携带的一条引用。"""

    doi_or_pmid: str = Field(description="稳定标识；D1 核验的输入")
    paper_id: str | None = Field(default=None, description="本地论文标识")
    evidence_span_ids: list[str] = Field(
        default_factory=list, description="支撑该引用的 EvidenceSpan.span_id 列表"
    )


class AtomicClaim(StrictModel):
    """原子主张（方案 8.1）。

    定义约束：只允许一个主体、一个关系或结局、一个效应方向及一组实验条件。
    并列结局、不同方向或不同条件必须拆分。本类不校验「原子性」——拆分器在校准后
    冻结并由人工抽检，不由被测系统自报（方案 8.1）。
    """

    claim_id: str
    text: str
    is_core: bool = Field(
        default=False,
        description="核心主张：直接回答研究问题、出现在摘要结论，或对应金标准 "
        "required_claims；D2 中权重为次要主张的 2 倍（方案 8.1 / 8.2 D2）",
    )
    conditions: ExperimentConditions = Field(default_factory=ExperimentConditions)
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _text_not_blank(self) -> AtomicClaim:
        if not self.text.strip():
            raise ValueError("AtomicClaim.text 不能为空")
        return self


class JudgeVerdict(StrictModel):
    """单个原子主张的判定结果（方案 8.4 Judge 约束：只接收问题、一个主张和候选原文）。"""

    claim_id: str
    verdict: SupportVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="判定理由；低置信与方向冲突项进入人工复核")
    evidence_span_refs: list[str] = Field(
        default_factory=list, description="判定所依据的 EvidenceSpan.span_id 列表"
    )

    @model_validator(mode="after")
    def _unknown_needs_no_evidence(self) -> JudgeVerdict:
        needs_evidence = self.verdict in (
            SupportVerdict.FULLY_SUPPORTED,
            SupportVerdict.PARTIALLY_SUPPORTED,
            SupportVerdict.REFUTED,
        )
        if needs_evidence and not self.evidence_span_refs:
            raise ValueError(
                f"verdict={self.verdict.value} 必须给出 evidence_span_refs；"
                "无可定位证据时应判 unknown（方案 8.1「未知」定义）"
            )
        return self


# ---------------------------------------------------------------------------
# 金标准（方案 9.2）
# ---------------------------------------------------------------------------


class EvidencePaper(StrictModel):
    """金标准证据池中的一篇论文。"""

    paper_id: str
    doi_or_pmid: str
    title: str | None = None
    year: int | None = None
    source_access: SourceAccess = SourceAccess.METADATA_ONLY
    is_key_evidence: bool = Field(
        default=False, description="是否属于池化关键证据（D3 Recall 的分母）"
    )
    is_conflict_or_negative: bool = Field(
        default=False, description="是否为冲突/阴性研究（D3 4 分附加条件与 D5 适用项判定）"
    )


class QuestionGold(StrictModel):
    """方案 9.2 的金标记录。字段清单与方案 JSON 一致，不增删。"""

    question_id: str
    question: str
    scope: str = Field(default="", description="题目范围与纳排边界")
    answerability: Answerability
    required_claims: list[AtomicClaim] = Field(default_factory=list)
    optional_claims: list[AtomicClaim] = Field(default_factory=list)
    evidence_papers: list[EvidencePaper] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    required_context_slots: list[ConditionSlot] = Field(
        default_factory=list, description="题目要求作答必须写明的条件槽位（D4 分母的「题目要求」部分）"
    )
    known_conflicts: list[str] = Field(
        default_factory=list, description="已知冲突/异质性描述；为空表示 D3/D5 对应检查记 NA"
    )
    prohibited_inferences: list[str] = Field(
        default_factory=list, description="禁止的外推与越界推断；D6 与致命错误判定依据"
    )

    @model_validator(mode="after")
    def _required_claims_are_core(self) -> QuestionGold:
        bad = [c.claim_id for c in self.required_claims if not c.is_core]
        if bad:
            raise ValueError(
                f"required_claims 必须标记 is_core=True（方案 8.1 核心主张定义）：{bad}"
            )
        return self


# ---------------------------------------------------------------------------
# 核验与评估结果
# ---------------------------------------------------------------------------


class IdentifierVerification(StrictModel):
    """单条标识符的核验结论（方案 8.3 三态规则）。"""

    input_id: str = Field(description="原始输入的标识符字符串")
    normalized_id: str | None = Field(default=None, description="规范化后的 DOI/PMID/PMCID")
    id_type: str = Field(description="doi / pmid / pmcid / unknown")
    status: VerificationStatus
    title: str | None = None
    journal: str | None = None
    year: int | None = None
    first_author: str | None = Field(
        default=None, description="解析到的第一作者（D1 作者要素比对的实际值）"
    )
    metadata_match: dict[str, str] = Field(
        default_factory=dict,
        description="题名/作者/年份三要素的逐项比对结果，取值 match|conflict|partial；"
        "未给出期望元数据时为空字典（方案 8.2 D1 的三要素口径）",
    )
    reason: str = Field(default="", description="mismatch/unresolved 的具体原因")
    source: str = Field(default="", description="核验数据源：crossref / ncbi_esummary / syntax")


class FatalErrorRecord(StrictModel):
    """一条已触发的致命错误（方案 8.3）。"""

    key: str
    label_zh: str
    score_cap: int
    evidence: str = Field(default="", description="触发依据，写入审计包")


class DimensionScore(StrictModel):
    """单维评分结果，同时保留连续指标与 0—4 档位（方案 8.3 末段要求）。"""

    dimension: str
    name_zh: str
    weight: int
    is_na: bool = False
    level: int | None = Field(default=None, ge=0, le=4)
    metric_name: str | None = None
    metric_value: float | None = Field(default=None, description="分档前的原始连续指标")
    level_before_event_caps: int | None = Field(default=None, ge=0, le=4)
    event_caps_applied: list[str] = Field(
        default_factory=list, description="已生效的事件上限 flag 名称"
    )
    notes: str = ""

    @model_validator(mode="after")
    def _na_has_no_level(self) -> DimensionScore:
        if self.is_na and self.level is not None:
            raise ValueError(f"{self.dimension} 记 NA 时不得同时给出 level")
        if not self.is_na and self.level is None:
            raise ValueError(f"{self.dimension} 非 NA 时必须给出 level")
        return self


class EvaluationResult(StrictModel):
    """单份输出的完整评估结果。

    方案 8.3 末段：结果同时保留原始连续指标、0—4 档位、致命错误类型和发布决策，
    避免只看一个总分。
    """

    question_id: str
    output_id: str | None = Field(default=None, description="被评输出的匿名编号（方案 10.2）")
    dimension_scores: dict[str, DimensionScore]
    raw_score: float = Field(ge=0.0, le=100.0, description="九维加权折算分，未施加致命错误上限")
    final_score: float = Field(ge=0.0, le=100.0, description="min(RawScore, 所有已触发上限)")
    applied_score_cap: int | None = Field(
        default=None, description="实际生效的最低上限；无致命错误时为 None"
    )
    fatal_errors: list[FatalErrorRecord] = Field(default_factory=list)
    unresolved_unverifiable: bool = Field(
        default=False, description="存在未解决的「不可核验」项（方案 8.3 复核触发条件）"
    )
    decision: ReleaseDecision
    decision_reasons: list[str] = Field(default_factory=list)
    na_dimensions: list[str] = Field(default_factory=list)
    effective_weight_sum: int = Field(description="重归一后参与计分的权重之和")
    evaluator_version: str
    rubric_version: str
    rubric_config_sha256: str = Field(description="量表配置文件哈希，写入 run manifest")
