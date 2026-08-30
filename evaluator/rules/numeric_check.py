"""D8 数字与单位的抽取与比对辅助。

方案 8.2 D8：只评价答案实际写出或题目要求写出的术语、数字和单位；分母沿用 D4 的
「适用或题目要求」；原文未报告且题目不要求记 NA；出现数量级或单位错误直接判 0 分。
方案 9.3 第 9 类对抗攻击即「修改剂量、时间、样本量、单位或统计方向」，因此比对必须
同时看数值、单位量纲与比较符方向。

本模块只提供数字与单位的确定性辅助：抽取候选量、单位换算、逐对比对与 D8 输入汇总。
术语初筛由同目录 `terminology_check.py` 提供；它仅使用本地版本化词表，完整 MeSH/GO
对齐仍须外部或人工核验。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import Field

from evaluator.schemas import StrictModel

# 相对容差：同量纲数值在此容差内视为一致。方案未规定数值容差，属实现选择
# （见 eval/rubric.md「待澄清 J」）。
DEFAULT_REL_TOL = 1e-6
# 数量级判据：换算到同一基准单位后比值达到 10 倍即判数量级错误。
MAGNITUDE_FACTOR = 10.0

# 单位表：unit -> (量纲, 相对基准单位的换算系数)。
# μ(U+03BC) 与 µ(U+00B5) 统一折叠为 u。
_UNIT_TABLE: dict[str, tuple[str, float]] = {
    # 摩尔浓度，基准 M
    "M": ("concentration", 1.0),
    "mM": ("concentration", 1e-3),
    "uM": ("concentration", 1e-6),
    "nM": ("concentration", 1e-9),
    "pM": ("concentration", 1e-12),
    # 时间，基准 s
    "s": ("time", 1.0),
    "sec": ("time", 1.0),
    "second": ("time", 1.0),
    "seconds": ("time", 1.0),
    "min": ("time", 60.0),
    "minute": ("time", 60.0),
    "minutes": ("time", 60.0),
    "h": ("time", 3600.0),
    "hr": ("time", 3600.0),
    "hour": ("time", 3600.0),
    "hours": ("time", 3600.0),
    "d": ("time", 86400.0),
    "day": ("time", 86400.0),
    "days": ("time", 86400.0),
    "wk": ("time", 604800.0),
    "week": ("time", 604800.0),
    "weeks": ("time", 604800.0),
    # 质量，基准 g
    "kg": ("mass", 1e3),
    "g": ("mass", 1.0),
    "mg": ("mass", 1e-3),
    "ug": ("mass", 1e-6),
    "ng": ("mass", 1e-9),
    "pg": ("mass", 1e-12),
    # 体积，基准 L
    "L": ("volume", 1.0),
    "mL": ("volume", 1e-3),
    "uL": ("volume", 1e-6),
    "nL": ("volume", 1e-9),
    # 长度，基准 m
    "m": ("length", 1.0),
    "cm": ("length", 1e-2),
    "mm": ("length", 1e-3),
    "um": ("length", 1e-6),
    "nm": ("length", 1e-9),
    # 剂量/体重，基准 mg/kg
    "g/kg": ("dose_per_mass", 1e3),
    "mg/kg": ("dose_per_mass", 1.0),
    "ug/kg": ("dose_per_mass", 1e-3),
    # 质量浓度，基准 mg/mL
    "mg/mL": ("mass_concentration", 1.0),
    "ug/mL": ("mass_concentration", 1e-3),
    "ng/mL": ("mass_concentration", 1e-6),
    "pg/mL": ("mass_concentration", 1e-9),
    # 其他
    "%": ("percent", 1.0),
    "degC": ("temperature", 1.0),
}

_MICRO_CHARS = ("\u03bc", "\u00b5")  # μ 与 µ
_DEGREE_C = ("\u2103", "\u00b0C", "\u00b0 C")

# 单位字面量按长度降序，保证 "mg/kg" 先于 "mg" 匹配。
_UNIT_ALTERNATION = "|".join(
    re.escape(u) for u in sorted(_UNIT_TABLE, key=len, reverse=True)
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCI = r"(?:\s*[x*\u00d7\u2715]\s*10\s*(?:\^|\*\*)?\s*(?P<exp>[-+]?\d+))?"
_COMPARATOR = r"(?P<cmp>[<>]=?|[\u2264\u2265\u2248=])?\s*"
# 前后守卫的作用：避免把 "INS-1E"、"Drp1" 这类品系/基因名里的数字当成量。
#   (?<![\w.\-])  数字（含符号）之前不能紧邻字母、数字、点或连字符；
#   (?![A-Za-z0-9]) 单位之后不能紧邻字母数字，防止 "1E"、"Fig3" 之类误配。
_QUANTITY_RE = re.compile(
    rf"(?:(?P<label>[a-zA-Z][a-zA-Z0-9_]{{0,7}})\s*)?{_COMPARATOR}"
    rf"(?<![\w.\-])(?P<value>{_NUMBER}){_SCI}"
    rf"\s*(?P<unit>{_UNIT_ALTERNATION})?(?![A-Za-z0-9])",
)

_COMPARATOR_CANON = {
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2248": "~=",
    "=": "=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
}


def canonicalize_unit_text(text: str) -> str:
    """把 μ/µ 折叠为 u，把 ℃/°C 折叠为 degC，便于统一匹配。"""
    out = text
    for ch in _MICRO_CHARS:
        out = out.replace(ch, "u")
    for ch in _DEGREE_C:
        out = out.replace(ch, "degC")
    return out


@dataclass(frozen=True)
class Quantity:
    """一个带单位的量。"""

    value: float
    unit: str | None
    raw: str
    label: str | None = None
    comparator: str | None = None

    @property
    def dimension(self) -> str:
        if self.unit is None:
            return "dimensionless"
        return _UNIT_TABLE[self.unit][0]

    @property
    def canonical_value(self) -> float:
        """换算到该量纲基准单位后的数值。"""
        if self.unit is None:
            return self.value
        return self.value * _UNIT_TABLE[self.unit][1]


def extract_quantities(text: str) -> list[Quantity]:
    """从自由文本中抽取带单位/比较符的候选量。

    覆盖科学计数法（1.5 × 10^-3、1.5e-3）、比较符（p < 0.01）与常见生物医学单位。
    抽取器是候选生成器，不做语义对齐：哪个量对应哪个槽位由调用方决定。
    """
    normalized = canonicalize_unit_text(text or "")
    results: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(normalized):
        raw_value = match.group("value")
        if raw_value is None:
            continue
        value = float(raw_value)
        if match.group("exp") is not None:
            value *= 10.0 ** int(match.group("exp"))
        comparator = match.group("cmp")
        label = match.group("label")
        # 形如 "10 mM" 时 label 会误吞前一个词，只在紧邻比较符时才认作标签。
        if label is not None and comparator is None:
            label = None
        results.append(
            Quantity(
                value=value,
                unit=match.group("unit"),
                raw=match.group(0).strip(),
                label=label,
                comparator=_COMPARATOR_CANON.get(comparator or "", None),
            )
        )
    return results


def parse_quantity(text: str) -> Quantity | None:
    """把单个字面量解析为 Quantity；解析不出时返回 None。"""
    found = extract_quantities(text)
    return found[0] if found else None


class QuantityComparison(StrictModel):
    """一对量的比对结论。"""

    claimed: str
    reference: str
    verdict: str = Field(
        description="match / value_mismatch / magnitude_error / unit_error / "
        "comparator_flip / unparsable"
    )
    detail: str = ""

    @property
    def is_match(self) -> bool:
        return self.verdict == "match"

    @property
    def is_magnitude_or_unit_error(self) -> bool:
        """方案 8.2 D8 0 分事件：出现数量级、单位错误。"""
        return self.verdict in ("magnitude_error", "unit_error")


def compare_quantities(
    claimed: str, reference: str, rel_tol: float = DEFAULT_REL_TOL
) -> QuantityComparison:
    """比对「答案写出的量」与「原文/金标的量」。

    判定顺序：
      1. 任一侧解析不出 → unparsable（不计入准确率分子，交由调用方决定是否记 NA）；
      2. 量纲不同 → unit_error；
      3. 换算后比值 ≥ 10 或 ≤ 0.1 → magnitude_error；
      4. 换算后相对差 ≤ rel_tol → 再看比较符方向是否一致：不一致记 comparator_flip；
      5. 其余 → value_mismatch。
    """
    a = parse_quantity(claimed)
    b = parse_quantity(reference)
    if a is None or b is None:
        return QuantityComparison(
            claimed=claimed,
            reference=reference,
            verdict="unparsable",
            detail=f"解析失败：claimed={a is not None}, reference={b is not None}",
        )
    if a.dimension != b.dimension:
        return QuantityComparison(
            claimed=claimed,
            reference=reference,
            verdict="unit_error",
            detail=f"量纲不同：{a.dimension}({a.unit}) vs {b.dimension}({b.unit})",
        )

    va, vb = a.canonical_value, b.canonical_value
    if vb == 0.0:
        matched = va == 0.0
        ratio = float("inf") if not matched else 1.0
    else:
        ratio = abs(va / vb) if va != 0.0 else 0.0
        matched = abs(va - vb) <= rel_tol * max(abs(va), abs(vb))

    base = f"{a.value:g}{a.unit or ''} vs {b.value:g}{b.unit or ''}（换算后 {va:g} vs {vb:g}）"
    if not matched and (ratio >= MAGNITUDE_FACTOR or ratio <= 1.0 / MAGNITUDE_FACTOR):
        return QuantityComparison(
            claimed=claimed,
            reference=reference,
            verdict="magnitude_error",
            detail=f"数量级偏差：{base}，比值 {ratio:g}",
        )
    if not matched:
        return QuantityComparison(
            claimed=claimed, reference=reference, verdict="value_mismatch", detail=f"数值不一致：{base}"
        )
    if a.comparator != b.comparator:
        return QuantityComparison(
            claimed=claimed,
            reference=reference,
            verdict="comparator_flip",
            detail=f"比较符方向不一致：{a.comparator} vs {b.comparator}",
        )
    return QuantityComparison(
        claimed=claimed, reference=reference, verdict="match", detail=f"{va:g} == {vb:g}"
    )


class NumericCheckItem(StrictModel):
    """一项待核数字/单位/术语。"""

    slot: str = Field(description="所属槽位或字段名，如 dose / time / sample_size / term")
    claimed: str
    reference: str | None = Field(
        default=None, description="原文/金标值；None 表示原文未报告且题目不要求 → 记 NA"
    )
    is_key: bool = Field(default=False, description="是否为关键数字/单位（D8 4 分附加条件）")


class NumericCheckSummary(StrictModel):
    """D8 输入汇总。"""

    applicable: int
    correct: int
    accuracy: float | None = Field(
        description="D8 的 p；无适用项时为 None，此时 D8 应整体记 NA"
    )
    comparisons: list[QuantityComparison] = Field(default_factory=list)
    key_number_or_unit_error: bool = False
    magnitude_or_unit_error: bool = False
    unparsable_slots: list[str] = Field(default_factory=list)

    def event_flags(self) -> dict[str, bool]:
        """转成 rubric.DimensionInput.event_flags 可直接使用的形式。"""
        return {
            "key_number_or_unit_error": self.key_number_or_unit_error,
            "magnitude_or_unit_error": self.magnitude_or_unit_error,
        }


def check_numeric_items(
    items: Iterable[NumericCheckItem], rel_tol: float = DEFAULT_REL_TOL
) -> NumericCheckSummary:
    """逐项比对并汇总为 D8 输入。

    reference 为 None 的项记 NA（不进入分母）；解析失败的项计入分母但不计入分子，
    并登记到 unparsable_slots 供人工复核（方案 8.4：低置信项进入人工复核）。
    """
    comparisons: list[QuantityComparison] = []
    unparsable: list[str] = []
    applicable = 0
    correct = 0
    key_error = False
    magnitude_error = False

    for item in items:
        if item.reference is None:
            continue
        applicable += 1
        comparison = compare_quantities(item.claimed, item.reference, rel_tol)
        comparisons.append(comparison)
        if comparison.verdict == "unparsable":
            unparsable.append(item.slot)
            continue
        if comparison.is_match:
            correct += 1
            continue
        if item.is_key:
            key_error = True
        if comparison.is_magnitude_or_unit_error:
            magnitude_error = True

    return NumericCheckSummary(
        applicable=applicable,
        correct=correct,
        accuracy=(correct / applicable) if applicable else None,
        comparisons=comparisons,
        key_number_or_unit_error=key_error,
        magnitude_or_unit_error=magnitude_error,
        unparsable_slots=unparsable,
    )


def compare_number_sets(
    claimed_text: str, reference_text: str, rel_tol: float = DEFAULT_REL_TOL
) -> tuple[list[Quantity], list[Quantity], list[Quantity]]:
    """粗筛辅助：返回 (仅出现在答案中的量, 仅出现在原文中的量, 双方一致的量)。

    用于快速定位「答案写出了原文没有的数字」这类问题；不替代逐槽位比对，
    结果只能作为人工复核的候选清单。
    """
    claimed = extract_quantities(claimed_text)
    reference = extract_quantities(reference_text)
    remaining = list(reference)
    matched: list[Quantity] = []
    only_claimed: list[Quantity] = []

    for q in claimed:
        hit = None
        for candidate in remaining:
            if candidate.dimension != q.dimension:
                continue
            va, vb = q.canonical_value, candidate.canonical_value
            if abs(va - vb) <= rel_tol * max(abs(va), abs(vb), 1e-30):
                hit = candidate
                break
        if hit is None:
            only_claimed.append(q)
        else:
            remaining.remove(hit)
            matched.append(q)
    return only_claimed, remaining, matched


def numeric_items_from_pairs(
    pairs: Sequence[tuple[str, str, str | None]], key_slots: Sequence[str] = ()
) -> list[NumericCheckItem]:
    """便捷构造：(slot, claimed, reference) 三元组列表 → NumericCheckItem 列表。"""
    key_set = set(key_slots)
    return [
        NumericCheckItem(
            slot=slot, claimed=claimed, reference=reference, is_key=slot in key_set
        )
        for slot, claimed, reference in pairs
    ]
