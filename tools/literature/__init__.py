"""文献数据源客户端与金标语料摄取管线（方案 9.2 证据池构建的第一步）。

  - epmc_client.py    Europe PMC REST：元数据 / OA 全文 XML / 参考文献列表（分页）；
  - crossref_refs.py  Crossref /works/{doi} 的 reference 字段（非 OA 综述引文兜底）；
  - pool_builder.py   引文合并去重、证据池候选与构建 manifest（纯函数，离线可测）。
  - frozen_fetch.py   只获取并写入与冻结 manifest SHA-256 完全一致的 OA XML。
  - xml_anchor.py     EvidenceSpan 在 Europe PMC fullTextXML 上的离线重定位与消歧。
"""

from tools.literature.xml_anchor import (
    AnchorCandidate,
    AnchorRelocationResult,
    AnchorStatus,
    EpmcXmlDocument,
    MalformedXmlError,
    ParagraphLocation,
    UnsafeXmlError,
    XmlAnchorError,
    XmlParagraph,
    normalize_anchor_text,
    parse_epmc_fulltext_xml,
    relocate_evidence_span,
    relocate_text_anchor,
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
