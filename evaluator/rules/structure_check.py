"""D7 流程必需项与 D9 清单的可配置检查器。

方案 8.2 只给出两组条目的名称：
  D7：检索数据库/检索式/日期、纳排记录、原文定位、稳定标识、运行版本、证据快照；
  D9：直接回答、结构固定、提供证据矩阵、引文紧邻主张、解释必要术语、无无关扩写。
方案没有给出把条目名称落成程序判据的细则，因此本模块的结构：

  1. ChecklistChecker —— 唯一的计分入口。条目清单从 configs/rubric_v0_1.yaml 读取，
     调用方逐条给出 True/False/None(NA)，检查器只负责校验键名、计数与分档输入。
     这一层不含任何领域判断，是「量表条目」到「档位指标」的纯映射。
  2. DEFAULT_D7_RULES / DEFAULT_D9_RULES —— 默认的程序化判据。它们是对条目名称的
     一种操作化实现（方案未规定），标注为待校准项，见 eval/rubric.md「待澄清 I」。
     可整体替换而不影响 ChecklistChecker。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field

from evaluator.rubric import RubricConfig, default_rubric
from evaluator.schemas import StrictModel


class ChecklistResult(StrictModel):
    """一组清单条目的检查结果。"""

    dimension: str
    satisfied: list[str] = Field(default_factory=list)
    unsatisfied: list[str] = Field(default_factory=list)
    not_applicable: list[str] = Field(default_factory=list)

    @property
    def satisfied_count(self) -> int:
        return len(self.satisfied)

    @property
    def applicable_count(self) -> int:
        return len(self.satisfied) + len(self.unsatisfied)

    @property
    def metric_count(self) -> int:
        """D7/D9 的分档指标：满足的条目数（方案 8.2 按 n/6 分档）。"""
        return self.satisfied_count

    @property
    def satisfaction_rate(self) -> float | None:
        """满足比例；全部条目 NA 时为 None（该维应整体记 NA）。"""
        if self.applicable_count == 0:
            return None
        return self.satisfied_count / self.applicable_count


class ChecklistChecker:
    """按配置清单计数的检查器；不含领域判断。"""

    def __init__(self, dimension: str, config: RubricConfig | None = None):
        cfg = config or default_rubric()
        spec = cfg.dimensions[dimension]
        checklist = spec.get("checklist")
        if not checklist:
            raise ValueError(f"{dimension} 在配置中没有 checklist，无法用清单检查器计分")
        self.dimension = dimension
        self.config = cfg
        self.item_keys: tuple[str, ...] = tuple(str(item["key"]) for item in checklist)
        self.labels: dict[str, str] = {
            str(item["key"]): str(item.get("label_zh", item["key"])) for item in checklist
        }
        denominator = spec.get("denominator")
        self.denominator: int | None = None if denominator is None else int(denominator)

    def check(self, observations: Mapping[str, bool | None]) -> ChecklistResult:
        """observations：条目键 → True 满足 / False 不满足 / None 不适用。

        缺失的条目视为「不满足」而不是「不适用」：方案 10.3 要求缺失输出记为失败，
        不得从分母删除。
        """
        unknown = [k for k in observations if k not in self.item_keys]
        if unknown:
            raise ValueError(
                f"{self.dimension} 出现未知清单条目 {unknown}；合法键：{list(self.item_keys)}"
            )
        result = ChecklistResult(dimension=self.dimension)
        for key in self.item_keys:
            value = observations.get(key, False)
            if value is None:
                result.not_applicable.append(key)
            elif value:
                result.satisfied.append(key)
            else:
                result.unsatisfied.append(key)
        return result


# ---------------------------------------------------------------------------
# D7 可追溯性：结构化运行记录
# ---------------------------------------------------------------------------


class TraceabilityArtifacts(StrictModel):
    """一次运行产生的可追溯性凭证（方案 6.3 目标可追溯运行记录的子集）。"""

    search_databases: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    search_date: str | None = None
    inclusion_exclusion_records: int = Field(
        default=0, description="带纳入/排除原因的文献记录条数"
    )
    core_claims_total: int = 0
    core_claims_localized: int = Field(
        default=0, description="能回到原文（章节/页码/图表 + 文本锚点）的核心主张数"
    )
    citations_total: int = 0
    citations_with_stable_id: int = Field(
        default=0, description="带 DOI/PMID/PMCID 等稳定标识的引用数"
    )
    run_version: str | None = Field(default=None, description="模型、提示词与工具版本指纹")
    evidence_snapshot_hash: str | None = Field(default=None, description="证据快照哈希")

    @property
    def core_claim_traceability_rate(self) -> float | None:
        if self.core_claims_total == 0:
            return None
        return self.core_claims_localized / self.core_claims_total


# 每条判据都是对 D7 条目名称的一种操作化实现，方案未规定细则（待澄清 I）。
DEFAULT_D7_RULES: dict[str, Callable[[TraceabilityArtifacts], bool | None]] = {
    # 「检索数据库/检索式/日期」三者齐备才算满足。
    "search_db_query_date": lambda a: bool(a.search_databases and a.search_queries and a.search_date),
    # 「纳排记录」：至少有一条带原因的纳入/排除记录。
    "inclusion_exclusion_record": lambda a: a.inclusion_exclusion_records > 0,
    # 「原文定位」：全部核心主张都能回到原文；核心主张为 0 时无从判断，记 NA。
    "source_localization": lambda a: (
        None if a.core_claims_total == 0 else a.core_claims_localized == a.core_claims_total
    ),
    # 「稳定标识」：全部引用都带稳定标识；无引用时记 NA。
    "stable_identifiers": lambda a: (
        None if a.citations_total == 0 else a.citations_with_stable_id == a.citations_total
    ),
    "run_version": lambda a: bool(a.run_version),
    "evidence_snapshot": lambda a: bool(a.evidence_snapshot_hash),
}


def check_d7(
    artifacts: TraceabilityArtifacts,
    rules: Mapping[str, Callable[[TraceabilityArtifacts], bool | None]] | None = None,
    config: RubricConfig | None = None,
) -> tuple[ChecklistResult, dict[str, Any]]:
    """返回 (清单结果, D7 事件上限 flag)。

    事件 flag：core_claim_traceability_below_threshold 对应方案 8.2 D7 的
    4 分附加条件「至少 95% 核心主张可回到原文」；阈值来自配置。
    """
    cfg = config or default_rubric()
    checker = ChecklistChecker("D7", cfg)
    rule_set = dict(rules or DEFAULT_D7_RULES)
    result = checker.check({key: rule(artifacts) for key, rule in rule_set.items()})

    threshold = float(cfg.dimensions["D7"]["core_claim_traceability_threshold"])
    rate = artifacts.core_claim_traceability_rate
    flags = {
        "core_claim_traceability_below_threshold": rate is not None and rate < threshold,
    }
    return result, flags


# ---------------------------------------------------------------------------
# D9 可理解性与格式：结构化答案凭证
# ---------------------------------------------------------------------------

# 方案 3.2 / 8.2 D9 所要求的固定结构；作为默认判据的参数，可按题型覆盖。
DEFAULT_REQUIRED_SECTIONS = ("结论", "证据", "冲突", "局限")


class PresentationArtifacts(StrictModel):
    """一份综述输出在格式与可理解性上的可程序校验特征。"""

    has_direct_answer: bool = False
    present_sections: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=lambda: list(DEFAULT_REQUIRED_SECTIONS))
    evidence_matrix_rows: int = 0
    claims_total: int = 0
    claims_with_adjacent_citation: int = 0
    jargon_terms_total: int = 0
    jargon_terms_explained: int = 0
    off_topic_sections: int = Field(default=0, description="与研究问题无关的扩写段落数")

    @property
    def missing_sections(self) -> list[str]:
        present = set(self.present_sections)
        return [s for s in self.required_sections if s not in present]


# 每条判据都是对 D9 条目名称的一种操作化实现，方案未规定细则（待澄清 I）。
DEFAULT_D9_RULES: dict[str, Callable[[PresentationArtifacts], bool | None]] = {
    "direct_answer": lambda a: a.has_direct_answer,
    "fixed_structure": lambda a: not a.missing_sections,
    "evidence_matrix_provided": lambda a: a.evidence_matrix_rows > 0,
    # 「引文紧邻主张」：全部主张都有紧邻引文；无主张时记 NA。
    "citation_adjacent_to_claim": lambda a: (
        None if a.claims_total == 0 else a.claims_with_adjacent_citation == a.claims_total
    ),
    # 「解释必要术语」：无需解释的术语为 0 时记 NA。
    "terms_explained": lambda a: (
        None if a.jargon_terms_total == 0 else a.jargon_terms_explained == a.jargon_terms_total
    ),
    "no_irrelevant_padding": lambda a: a.off_topic_sections == 0,
}


def check_d9(
    artifacts: PresentationArtifacts,
    rules: Mapping[str, Callable[[PresentationArtifacts], bool | None]] | None = None,
    config: RubricConfig | None = None,
) -> ChecklistResult:
    """D9 六项清单的程序校验（方案维度对照表：D9 走程序校验，不打印象分）。"""
    checker = ChecklistChecker("D9", config or default_rubric())
    rule_set = dict(rules or DEFAULT_D9_RULES)
    return checker.check({key: rule(artifacts) for key, rule in rule_set.items()})
