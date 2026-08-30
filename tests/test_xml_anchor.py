"""EvidenceSpan 在 Europe PMC fullTextXML 上重定位的离线测试。"""
from __future__ import annotations

from dataclasses import fields

import pytest

from evaluator.schemas import EvidenceSpan, SourceAccess, TextAnchor
from tools.literature.xml_anchor import (
    AnchorCandidate,
    AnchorRelocationResult,
    AnchorStatus,
    EpmcXmlDocument,
    MalformedXmlError,
    ParagraphLocation,
    UnsafeXmlError,
    normalize_anchor_text,
    parse_epmc_fulltext_xml,
    relocate_evidence_span,
    relocate_text_anchor,
)


JATS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <abstract id="abs1">
        <title>Abstract</title>
        <p id="a1">High glucose increased ATP in beta cells.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec id="s-intro">
      <title>Introduction</title>
      <p id="i1">Earlier work reported that high glucose increased ATP in beta cells.</p>
    </sec>
    <sec id="s-results">
      <title>Results</title>
      <sec id="s-mito">
        <title>Mitochondrial responses</title>
        <p id="r1">In INS-1E cells, high <italic>glucose</italic>\n
          increased ATP in beta cells after 30 min.</p>
        <p id="r2">In mouse islets, high glucose increased ATP in beta cells after 60 min.</p>
      </sec>
    </sec>
    <sec id="s-discussion">
      <title>Discussion</title>
      <p id="d1">High glucose increased ATP in beta cells, consistent with prior work.</p>
    </sec>
  </body>
</article>
"""


def _span(anchor: TextAnchor, section: str | None = "Mitochondrial responses") -> EvidenceSpan:
    return EvidenceSpan(
        span_id="s1",
        paper_id="p1",
        doi_or_pmid="PMC0000001",
        section=section,
        anchor=anchor,
        source_access=SourceAccess.FULLTEXT,
    )


def test_normalize_whitespace_preserves_scientific_symbols():
    assert normalize_anchor_text("  Ca²⁺\u00a0  rose\n\tby 5 µM ") == "Ca²⁺ rose by 5 µM"


def test_parse_flattens_inline_markup_and_preserves_readable_location():
    document = parse_epmc_fulltext_xml(JATS_XML)
    paragraph = next(item for item in document.paragraphs if item.location.paragraph_id == "r1")
    assert "high glucose increased ATP" in paragraph.text
    assert paragraph.location.section_path == ("Results", "Mitochondrial responses")
    assert paragraph.location.section == "Mitochondrial responses"
    assert paragraph.location.section_paragraph_index == 1
    assert "Results > Mitochondrial responses" in paragraph.location.readable_position
    assert "id=r1" in paragraph.location.readable_position


def test_unique_exact_is_found_even_without_context():
    result = relocate_text_anchor(JATS_XML, TextAnchor(exact="after 60 min"))
    assert result.status is AnchorStatus.FOUND
    assert result.candidate_count == 1
    assert result.selected is not None
    assert result.selected.location.paragraph_id == "r2"
    assert "唯一命中" in result.reason


def test_multiple_exact_matches_use_prefix_and_postfix_to_disambiguate():
    anchor = TextAnchor(
        prefix="In INS-1E cells,",
        exact="high glucose increased ATP in beta cells",
        postfix="after 30 min.",
    )
    result = relocate_text_anchor(JATS_XML, anchor)
    assert result.status is AnchorStatus.FOUND
    assert result.candidate_count == 3
    assert result.selected is not None
    assert result.selected.location.paragraph_id == "r1"
    assert result.selected.prefix_score == 1.0
    assert result.selected.postfix_score == 1.0
    assert result.candidates[0].total_score > result.candidates[1].total_score


def test_expected_section_disambiguates_when_context_is_absent():
    anchor = TextAnchor(exact="High glucose increased ATP in beta cells")
    result = relocate_text_anchor(JATS_XML, anchor, expected_section="Discussion")
    assert result.status is AnchorStatus.FOUND
    assert result.selected is not None
    assert result.selected.location.paragraph_id == "d1"
    assert result.selected.section_score == 1.0


def test_tied_multiple_matches_are_ambiguous_and_not_silently_selected():
    xml = """<article><body><sec><title>Results</title>
      <p id="p1">Before. ATP increased. After.</p>
      <p id="p2">Before. ATP increased. After.</p>
    </sec></body></article>"""
    anchor = TextAnchor(prefix="Before.", exact="ATP increased.", postfix="After.")
    result = relocate_text_anchor(xml, anchor, expected_section="Results")
    assert result.status is AnchorStatus.AMBIGUOUS
    assert result.selected is None
    assert result.candidate_count == 2
    assert result.candidates[0].total_score == result.candidates[1].total_score == 1.0


def test_multiple_matches_without_context_are_ambiguous():
    result = relocate_text_anchor(JATS_XML, TextAnchor(exact="beta cells"))
    assert result.status is AnchorStatus.AMBIGUOUS
    assert result.selected is None
    assert result.candidate_count == 5


def test_missing_exact_returns_not_found():
    result = relocate_text_anchor(JATS_XML, TextAnchor(exact="Drp1 was knocked out"))
    assert result.status is AnchorStatus.NOT_FOUND
    assert result.selected is None
    assert result.candidates == ()


def test_evidence_span_wrapper_uses_span_section_for_disambiguation():
    span = _span(TextAnchor(exact="high glucose increased ATP in beta cells"))
    result = relocate_evidence_span(JATS_XML, span)
    # 同一 section 内仍有两个相同命中，仅 section 不应胡乱二选一。
    assert result.status is AnchorStatus.AMBIGUOUS
    assert result.selected is None


def test_document_can_be_reused_for_multiple_spans_and_metadata_has_no_anchor():
    document = EpmcXmlDocument.from_xml(JATS_XML)
    first = document.relocate(TextAnchor(exact="after 30 min"))
    second = document.relocate(TextAnchor(exact="after 60 min"))
    assert first.status is second.status is AnchorStatus.FOUND

    metadata_span = EvidenceSpan(
        span_id="meta",
        paper_id="p1",
        doi_or_pmid="PMC0000001",
        source_access=SourceAccess.METADATA_ONLY,
    )
    metadata_result = document.relocate_evidence_span(metadata_span)
    assert metadata_result.status is AnchorStatus.NOT_FOUND
    assert "metadata_only" in metadata_result.reason


def test_multiple_occurrences_in_one_paragraph_are_separate_candidates():
    xml = "<article><body><p id='p'>ATP rose; ATP rose again.</p></body></article>"
    result = relocate_text_anchor(xml, TextAnchor(exact="ATP rose"))
    assert result.status is AnchorStatus.AMBIGUOUS
    assert [item.occurrence_in_paragraph for item in result.candidates] == [1, 2]


def test_public_result_contract_contains_no_character_offset_fields():
    names = {
        field.name
        for model in (ParagraphLocation, AnchorCandidate, AnchorRelocationResult)
        for field in fields(model)
    }
    assert not {name for name in names if "offset" in name.casefold()}


@pytest.mark.parametrize(
    "xml",
    [
        "<!DOCTYPE article [<!ENTITY x 'boom'>]><article><p>&x;</p></article>",
        "<!ENTITY x 'boom'><article><p>text</p></article>",
    ],
)
def test_dtd_and_entity_declarations_are_rejected(xml):
    with pytest.raises(UnsafeXmlError, match="DTD|实体"):
        parse_epmc_fulltext_xml(xml)


def test_external_jats_doctype_is_stripped_without_resolving_it():
    xml = """<?xml version="1.0"?>
    <!DOCTYPE article
      PUBLIC "-//NLM//DTD JATS Journal Archiving DTD v1.4//EN"
      "JATS-archivearticle1-4.dtd">
    <article><body><sec><title>Results</title><p id="p1">ATP rose.</p></sec></body></article>
    """
    result = relocate_text_anchor(xml, TextAnchor(exact="ATP rose."))
    assert result.status is AnchorStatus.FOUND
    assert result.selected is not None
    assert result.selected.location.paragraph_id == "p1"


def test_malformed_xml_is_rejected_with_typed_error():
    with pytest.raises(MalformedXmlError, match="解析失败"):
        parse_epmc_fulltext_xml("<article><p>broken</article>")


def test_size_limit_is_enforced_before_parsing():
    with pytest.raises(UnsafeXmlError, match="大小"):
        parse_epmc_fulltext_xml("<article/>", max_xml_bytes=4)


def test_wide_byte_encoding_is_rejected_before_declaration_scan():
    utf16_xml = "<!DOCTYPE article><article/>".encode("utf-16")
    with pytest.raises(UnsafeXmlError, match="NUL|宽字节"):
        parse_epmc_fulltext_xml(utf16_xml)


def test_bytes_and_default_namespace_are_supported():
    xml = (
        b"<article xmlns='urn:jats'><body><sec><title>Results</title>"
        b"<p id='p1'>ATP rose.</p></sec></body></article>"
    )
    result = relocate_text_anchor(xml, TextAnchor(exact="ATP rose."))
    assert result.status is AnchorStatus.FOUND
    assert result.selected is not None
    assert result.selected.location.section == "Results"
