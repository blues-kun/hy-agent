"""Europe PMC fullTextXML 上的文本锚点重定位（纯离线）。

``EvidenceSpan`` 使用 W3C TextQuoteSelector 式的 ``prefix / exact /
postfix``，而不保存字符偏移。本模块将锚点重新定位到一份已冻结的
Europe PMC JATS XML 快照：

* 不解析外部 DTD，拒绝内部 DTD/实体声明，并设置资源上限；
* 将内联标记展平后规范化 Unicode 和空白，但不改写标点或科学符号；
* 以段落为搜索单位，保留 section 路径、段落 ID 和人类可读位置；
* ``exact`` 唯一命中时直接定位；多命中时用前文、后文和期望
  section 评分消歧，无法拉开分差时返回 ``ambiguous``；
* 只返回 ``found / ambiguous / not_found`` 三态，不暴露或声称任何
  字符 offset 在 XML 版本之间稳定。

该模块不发网络请求。上层应将 XML 快照与 SHA-256/PMCID 一起冻结；
若来源 XML 变化，应重新定位并复核，而不复用旧的可读位置。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from xml.etree import ElementTree as ET

from evaluator.schemas import EvidenceSpan, TextAnchor

MAX_XML_BYTES = 64 * 1024 * 1024
MAX_XML_ELEMENTS = 1_000_000
MAX_XML_DEPTH = 256
MAX_PARAGRAPHS = 200_000

_DOCTYPE_BYTES = re.compile(br"<!\s*DOCTYPE\b", re.IGNORECASE)
_DOCTYPE_TEXT = re.compile(r"<!\s*DOCTYPE\b", re.IGNORECASE)
_ENTITY_BYTES = re.compile(br"<!\s*ENTITY\b", re.IGNORECASE)
_ENTITY_TEXT = re.compile(r"<!\s*ENTITY\b", re.IGNORECASE)
_SPACE_TRANSLATION = str.maketrans({"\u00a0": " ", "\u2007": " ", "\u202f": " "})


class XmlAnchorError(ValueError):
    """全文 XML 无法安全解析时的公共异常。"""


class UnsafeXmlError(XmlAnchorError):
    """XML 含内部 DTD/实体声明或超出资源上限。"""


class MalformedXmlError(XmlAnchorError):
    """XML 语法错误。"""


class AnchorStatus(str, Enum):
    """锚点重定位的三态结果。"""

    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class ParagraphLocation:
    """段落级可读位置；这些索引只对当前 XML 快照有意义。"""

    section_path: tuple[str, ...]
    document_paragraph_index: int
    section_paragraph_index: int
    paragraph_id: str | None
    readable_position: str

    @property
    def section(self) -> str | None:
        """返回最内层 section 标题。"""
        return self.section_path[-1] if self.section_path else None


@dataclass(frozen=True)
class XmlParagraph:
    """从 JATS XML 展平的一个规范化段落。"""

    text: str
    location: ParagraphLocation


@dataclass(frozen=True)
class AnchorCandidate:
    """一个 ``exact`` 命中及其消歧评分。

    故意不保存开始/结束字符偏移；``occurrence_in_paragraph`` 只表示该段
    中第几次命中，便于人工阅读。
    """

    location: ParagraphLocation
    occurrence_in_paragraph: int
    matched_text: str
    paragraph_text: str
    excerpt: str
    prefix_score: float | None
    postfix_score: float | None
    section_score: float | None
    total_score: float


@dataclass(frozen=True)
class AnchorRelocationResult:
    """重定位结果。``ambiguous`` 时 ``selected`` 必须为 ``None``。"""

    status: AnchorStatus
    normalized_exact: str
    candidate_count: int
    selected: AnchorCandidate | None
    candidates: tuple[AnchorCandidate, ...]
    reason: str


def normalize_anchor_text(text: str) -> str:
    """NFC 归一 + 特殊空格归一 + 连续空白折叠。"""

    normalized = unicodedata.normalize("NFC", str(text)).translate(_SPACE_TRANSLATION)
    return " ".join(normalized.split())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _direct_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return normalize_anchor_text("".join(child.itertext()))
    return ""


def _strip_external_doctype(xml_source: str | bytes) -> str | bytes:
    """剔除 Europe PMC 常见的外部 JATS DOCTYPE，但绝不解析该 DTD。

    带 ``[...]`` 内部子集或任何 ``<!ENTITY>`` 声明一律拒绝。扫描时
    识别引号，避免把 SYSTEM/PUBLIC 标识符内容误当成声明结束。
    """

    is_bytes = isinstance(xml_source, bytes)
    doctype_re = _DOCTYPE_BYTES if is_bytes else _DOCTYPE_TEXT
    entity_re = _ENTITY_BYTES if is_bytes else _ENTITY_TEXT
    if entity_re.search(xml_source):
        raise UnsafeXmlError("XML 含实体声明，为防止实体展开已拒绝解析")
    matches = list(doctype_re.finditer(xml_source))
    if not matches:
        return xml_source
    if len(matches) != 1:
        raise UnsafeXmlError("XML 含多个 DOCTYPE 声明")

    start = matches[0].start()
    quote: int | str | None = None
    end: int | None = None
    open_bracket = ord("[") if is_bytes else "["
    close_declaration = ord(">") if is_bytes else ">"
    single_quote = ord("'") if is_bytes else "'"
    double_quote = ord('"') if is_bytes else '"'
    for index in range(matches[0].end(), len(xml_source)):
        char = xml_source[index]
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in (single_quote, double_quote):
            quote = char
        elif char == open_bracket:
            raise UnsafeXmlError("XML DOCTYPE 含内部子集，已拒绝解析")
        elif char == close_declaration:
            end = index + 1
            break
    if end is None:
        raise MalformedXmlError("XML DOCTYPE 声明未闭合")
    return xml_source[:start] + xml_source[end:]


def _validate_source(xml_source: str | bytes, max_xml_bytes: int) -> str | bytes:
    if not isinstance(xml_source, (str, bytes)):
        raise TypeError("xml_source 必须是 str 或 bytes")
    if isinstance(xml_source, bytes) and b"\x00" in xml_source:
        raise UnsafeXmlError(
            "XML bytes 含 NUL；不解析无法安全检查声明的宽字节编码"
        )
    source_size = (
        len(xml_source) if isinstance(xml_source, bytes) else len(xml_source.encode("utf-8"))
    )
    if source_size > max_xml_bytes:
        raise UnsafeXmlError(
            f"XML 大小 {source_size} bytes 超过安全上限 {max_xml_bytes} bytes"
        )
    return _strip_external_doctype(xml_source)


def _validate_tree(root: ET.Element) -> None:
    count = 0
    stack: list[tuple[ET.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > MAX_XML_ELEMENTS:
            raise UnsafeXmlError(f"XML 元素数超过安全上限 {MAX_XML_ELEMENTS}")
        if depth > MAX_XML_DEPTH:
            raise UnsafeXmlError(f"XML 嵌套深度超过安全上限 {MAX_XML_DEPTH}")
        stack.extend((child, depth + 1) for child in element)


def _section_path(element: ET.Element, parent_map: dict[ET.Element, ET.Element]) -> tuple[str, ...]:
    ancestors: list[ET.Element] = []
    current = parent_map.get(element)
    while current is not None:
        ancestors.append(current)
        current = parent_map.get(current)
    ancestors.reverse()

    path: list[str] = []
    for ancestor in ancestors:
        name = _local_name(ancestor.tag)
        label = ""
        if name == "sec":
            label = _direct_child_text(ancestor, "title")
        elif name == "abstract":
            label = _direct_child_text(ancestor, "title") or "Abstract"
        elif name in {"ack", "acknowledgments"}:
            label = _direct_child_text(ancestor, "title") or "Acknowledgements"
        elif name == "fig":
            figure_label = _direct_child_text(ancestor, "label")
            label = f"Figure {figure_label}" if figure_label else "Figure"
        elif name == "table-wrap":
            table_label = _direct_child_text(ancestor, "label")
            label = f"Table {table_label}" if table_label else "Table"
        if label and (not path or path[-1].casefold() != label.casefold()):
            path.append(label)
    return tuple(path)


def _readable_position(
    section_path: tuple[str, ...],
    document_index: int,
    section_index: int,
    paragraph_id: str | None,
) -> str:
    section_label = " > ".join(section_path) if section_path else "Article"
    id_label = f"，id={paragraph_id}" if paragraph_id else ""
    return (
        f"{section_label} · 第 {section_index} 段"
        f"（全文第 {document_index} 段{id_label}）"
    )


def parse_epmc_fulltext_xml(
    xml_source: str | bytes, *, max_xml_bytes: int = MAX_XML_BYTES
) -> "EpmcXmlDocument":
    """安全解析 Europe PMC JATS XML 并提取段落。

    不处理 XInclude，不发网络请求，不保存字符偏移。
    """

    source = _validate_source(xml_source, max_xml_bytes)
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise MalformedXmlError(f"XML 解析失败：{exc}") from exc
    _validate_tree(root)

    parent_map = {child: parent for parent in root.iter() for child in parent}
    paragraphs: list[XmlParagraph] = []
    section_counts: dict[tuple[str, ...], int] = {}
    for element in root.iter():
        if _local_name(element.tag) != "p":
            continue
        # 非法/非典型嵌套 <p> 不重复收录内层文本。
        parent = parent_map.get(element)
        nested = False
        while parent is not None:
            if _local_name(parent.tag) == "p":
                nested = True
                break
            parent = parent_map.get(parent)
        if nested:
            continue
        text = normalize_anchor_text("".join(element.itertext()))
        if not text:
            continue
        if len(paragraphs) >= MAX_PARAGRAPHS:
            raise UnsafeXmlError(f"XML 段落数超过安全上限 {MAX_PARAGRAPHS}")
        path = _section_path(element, parent_map)
        section_counts[path] = section_counts.get(path, 0) + 1
        document_index = len(paragraphs) + 1
        section_index = section_counts[path]
        paragraph_id = element.attrib.get("id") or None
        location = ParagraphLocation(
            section_path=path,
            document_paragraph_index=document_index,
            section_paragraph_index=section_index,
            paragraph_id=paragraph_id,
            readable_position=_readable_position(
                path, document_index, section_index, paragraph_id
            ),
        )
        paragraphs.append(XmlParagraph(text=text, location=location))
    return EpmcXmlDocument(paragraphs=tuple(paragraphs))


def _context_score(expected: str, actual: str) -> float | None:
    expected_norm = normalize_anchor_text(expected)
    if not expected_norm:
        return None
    actual_norm = normalize_anchor_text(actual)
    return round(
        SequenceMatcher(None, expected_norm.casefold(), actual_norm.casefold()).ratio(),
        6,
    )


def _section_score(expected_section: str | None, section_path: tuple[str, ...]) -> float | None:
    expected = normalize_anchor_text(expected_section or "").casefold()
    if not expected:
        return None
    labels = [normalize_anchor_text(label).casefold() for label in section_path]
    if not labels:
        return 0.0
    if expected in labels:
        return 1.0
    if any(expected in label or label in expected for label in labels):
        return 0.8
    # section 是受控的辅助信号；不用模糊标题相似度独自选中证据段。
    return 0.0


def _weighted_score(
    prefix_score: float | None,
    postfix_score: float | None,
    section_score: float | None,
) -> float:
    weighted: list[tuple[float, float]] = []
    if prefix_score is not None:
        weighted.append((prefix_score, 1.0))
    if postfix_score is not None:
        weighted.append((postfix_score, 1.0))
    if section_score is not None:
        weighted.append((section_score, 0.75))
    if not weighted:
        return 0.0
    return round(sum(score * weight for score, weight in weighted) / sum(w for _, w in weighted), 6)


def _excerpt(paragraph: str, start: int, exact_length: int, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(paragraph), start + exact_length + radius)
    prefix = "…" if left else ""
    postfix = "…" if right < len(paragraph) else ""
    return f"{prefix}{paragraph[left:right]}{postfix}"


@dataclass(frozen=True)
class EpmcXmlDocument:
    """一份已解析的 XML 快照，可批量重定位多个 EvidenceSpan。"""

    paragraphs: tuple[XmlParagraph, ...]

    @classmethod
    def from_xml(
        cls, xml_source: str | bytes, *, max_xml_bytes: int = MAX_XML_BYTES
    ) -> "EpmcXmlDocument":
        return parse_epmc_fulltext_xml(xml_source, max_xml_bytes=max_xml_bytes)

    def relocate(
        self,
        anchor: TextAnchor,
        *,
        expected_section: str | None = None,
        min_disambiguation_score: float = 0.75,
        min_score_margin: float = 0.15,
    ) -> AnchorRelocationResult:
        """在段落中重定位锚点，返回 found/ambiguous/not_found。"""

        if not 0.0 <= min_disambiguation_score <= 1.0:
            raise ValueError("min_disambiguation_score 必须在 [0, 1] 内")
        if not 0.0 <= min_score_margin <= 1.0:
            raise ValueError("min_score_margin 必须在 [0, 1] 内")
        exact = normalize_anchor_text(anchor.exact)
        if not exact:
            raise ValueError("anchor.exact 不能为空")
        prefix = normalize_anchor_text(anchor.prefix)
        postfix = normalize_anchor_text(anchor.postfix)

        candidates: list[AnchorCandidate] = []
        for paragraph in self.paragraphs:
            start = 0
            occurrence = 0
            while True:
                match_start = paragraph.text.find(exact, start)
                if match_start < 0:
                    break
                occurrence += 1
                match_end = match_start + len(exact)
                left_context = paragraph.text[:match_start].rstrip()
                right_context = paragraph.text[match_end:].lstrip()
                actual_prefix = left_context[-len(prefix) :] if prefix else ""
                actual_postfix = right_context[: len(postfix)] if postfix else ""
                prefix_score = _context_score(prefix, actual_prefix)
                postfix_score = _context_score(postfix, actual_postfix)
                section_score = _section_score(expected_section, paragraph.location.section_path)
                total_score = _weighted_score(prefix_score, postfix_score, section_score)
                candidates.append(
                    AnchorCandidate(
                        location=paragraph.location,
                        occurrence_in_paragraph=occurrence,
                        matched_text=paragraph.text[match_start:match_end],
                        paragraph_text=paragraph.text,
                        excerpt=_excerpt(paragraph.text, match_start, len(exact)),
                        prefix_score=prefix_score,
                        postfix_score=postfix_score,
                        section_score=section_score,
                        total_score=total_score,
                    )
                )
                start = match_end

        if not candidates:
            return AnchorRelocationResult(
                status=AnchorStatus.NOT_FOUND,
                normalized_exact=exact,
                candidate_count=0,
                selected=None,
                candidates=(),
                reason="exact 在所有规范化段落中均未命中",
            )

        ranked = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.total_score,
                    item.location.document_paragraph_index,
                    item.occurrence_in_paragraph,
                ),
            )
        )
        if len(ranked) == 1:
            return AnchorRelocationResult(
                status=AnchorStatus.FOUND,
                normalized_exact=exact,
                candidate_count=1,
                selected=ranked[0],
                candidates=ranked,
                reason="exact 在全文段落中唯一命中",
            )

        has_disambiguation_signal = bool(
            prefix or postfix or normalize_anchor_text(expected_section or "")
        )
        best = ranked[0]
        runner_up = ranked[1]
        margin = best.total_score - runner_up.total_score
        if (
            has_disambiguation_signal
            and best.total_score >= min_disambiguation_score
            and margin >= min_score_margin
        ):
            return AnchorRelocationResult(
                status=AnchorStatus.FOUND,
                normalized_exact=exact,
                candidate_count=len(ranked),
                selected=best,
                candidates=ranked,
                reason=(
                    f"exact 有 {len(ranked)} 个命中；前后文/section 评分消歧"
                    f"（top={best.total_score:.3f}, margin={margin:.3f}）"
                ),
            )
        return AnchorRelocationResult(
            status=AnchorStatus.AMBIGUOUS,
            normalized_exact=exact,
            candidate_count=len(ranked),
            selected=None,
            candidates=ranked,
            reason=(
                f"exact 有 {len(ranked)} 个命中，但前后文/section 不足以唯一消歧"
                f"（top={best.total_score:.3f}, margin={margin:.3f}）"
            ),
        )

    def relocate_evidence_span(
        self,
        span: EvidenceSpan,
        *,
        min_disambiguation_score: float = 0.75,
        min_score_margin: float = 0.15,
    ) -> AnchorRelocationResult:
        """用 ``EvidenceSpan.section`` 作为额外消歧信号。"""

        if span.anchor is None:
            return AnchorRelocationResult(
                status=AnchorStatus.NOT_FOUND,
                normalized_exact="",
                candidate_count=0,
                selected=None,
                candidates=(),
                reason="EvidenceSpan 无文本锚点（metadata_only 不能定位科学证据）",
            )
        return self.relocate(
            span.anchor,
            expected_section=span.section,
            min_disambiguation_score=min_disambiguation_score,
            min_score_margin=min_score_margin,
        )


def relocate_text_anchor(
    xml_source: str | bytes,
    anchor: TextAnchor,
    *,
    expected_section: str | None = None,
    min_disambiguation_score: float = 0.75,
    min_score_margin: float = 0.15,
) -> AnchorRelocationResult:
    """单锚点便捷函数；批量处理时应复用 ``EpmcXmlDocument``。"""

    document = parse_epmc_fulltext_xml(xml_source)
    return document.relocate(
        anchor,
        expected_section=expected_section,
        min_disambiguation_score=min_disambiguation_score,
        min_score_margin=min_score_margin,
    )


def relocate_evidence_span(
    xml_source: str | bytes,
    span: EvidenceSpan,
    *,
    min_disambiguation_score: float = 0.75,
    min_score_margin: float = 0.15,
) -> AnchorRelocationResult:
    """单 EvidenceSpan 便捷函数。"""

    document = parse_epmc_fulltext_xml(xml_source)
    return document.relocate_evidence_span(
        span,
        min_disambiguation_score=min_disambiguation_score,
        min_score_margin=min_score_margin,
    )


__all__ = [
    "AnchorCandidate",
    "AnchorRelocationResult",
    "AnchorStatus",
    "EpmcXmlDocument",
    "MalformedXmlError",
    "ParagraphLocation",
    "UnsafeXmlError",
    "XmlAnchorError",
    "XmlParagraph",
    "normalize_anchor_text",
    "parse_epmc_fulltext_xml",
    "relocate_evidence_span",
    "relocate_text_anchor",
]
