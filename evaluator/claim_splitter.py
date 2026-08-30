"""Auditable candidate splitter for open-ended biomedical answers.

This module deliberately produces *candidate* atomic claims.  It is independent
from a tested system's self-reported claim list and never decides the formal
evaluation denominator.  Sentence and semicolon boundaries are observable text
boundaries; ambiguous coordination, condition scope and effect direction are
flagged for human review instead of being resolved with invented semantics.

The splitter is an engineering draft until it has been calibrated on at least
20 independently reviewed outputs and its candidates have been confirmed by a
human reviewer.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, model_validator

from evaluator.schemas import StrictModel, TextAnchor


SPLITTER_VERSION = "candidate-splitter-v0.1.0-unfrozen"
MINIMUM_CALIBRATION_OUTPUTS = 20


class SplitBoundary(str, Enum):
    """Observable boundary used to create a candidate."""

    SENTENCE = "sentence"
    SEMICOLON_CLAUSE = "semicolon_clause"
    LINE_OR_BULLET = "line_or_bullet"
    WHOLE_ANSWER = "whole_answer"


class ReviewRisk(str, Enum):
    """Reasons why a candidate must not be accepted as atomic automatically."""

    COORDINATED_PROPOSITIONS = "coordinated_propositions"
    MULTIPLE_EFFECT_DIRECTIONS = "multiple_effect_directions"
    MULTIPLE_CONDITIONS_OR_COMPARATORS = "multiple_conditions_or_comparators"
    NEGATION_SCOPE = "negation_scope"
    CONTRAST_OR_EXCEPTION = "contrast_or_exception"
    ANAPHORA_OR_CONTEXT_DEPENDENCE = "anaphora_or_context_dependence"
    VERY_LONG_CANDIDATE = "very_long_candidate"


class ClaimSplitRequest(StrictModel):
    """Independent splitter input; tested-system claim declarations are forbidden."""

    output_id: str
    question: str
    answer: str

    @model_validator(mode="after")
    def _required_text(self) -> "ClaimSplitRequest":
        if not self.output_id.strip():
            raise ValueError("output_id cannot be blank")
        if not self.question.strip():
            raise ValueError("question cannot be blank")
        if not self.answer.strip():
            raise ValueError("answer cannot be blank")
        return self


class SourceSentenceAnchor(StrictModel):
    """Character-exact source sentence and candidate offsets in the answer."""

    sentence_id: str
    sentence_index: int = Field(ge=0)
    sentence_start_char: int = Field(ge=0)
    sentence_end_char: int = Field(gt=0)
    sentence_exact: str
    candidate_start_char: int = Field(ge=0)
    candidate_end_char: int = Field(gt=0)
    text_quote: TextAnchor

    @model_validator(mode="after")
    def _ordered_offsets(self) -> "SourceSentenceAnchor":
        if self.sentence_end_char <= self.sentence_start_char:
            raise ValueError("sentence offsets are not ordered")
        if self.candidate_end_char <= self.candidate_start_char:
            raise ValueError("candidate offsets are not ordered")
        if not (
            self.sentence_start_char
            <= self.candidate_start_char
            < self.candidate_end_char
            <= self.sentence_end_char
        ):
            raise ValueError("candidate offsets must be inside the source sentence")
        return self


class CandidateAtomicClaim(StrictModel):
    """A text-grounded candidate, not a confirmed atomic claim."""

    candidate_id: str
    text: str
    source: SourceSentenceAnchor
    split_boundary: SplitBoundary
    split_reason: str
    review_risks: list[ReviewRisk] = Field(default_factory=list)
    requires_human_review: bool

    @model_validator(mode="after")
    def _review_consistency(self) -> "CandidateAtomicClaim":
        if not self.text.strip():
            raise ValueError("candidate text cannot be blank")
        if self.review_risks and not self.requires_human_review:
            raise ValueError("candidates with review risks must require human review")
        return self


class CandidateSplitResult(StrictModel):
    """Candidate inventory with an explicit non-gold denominator boundary."""

    output_id: str
    question: str
    answer_sha256: str
    splitter_version: str = SPLITTER_VERSION
    calibration_status: str = "unfrozen_requires_20_output_calibration"
    minimum_calibration_outputs: int = MINIMUM_CALIBRATION_OUTPUTS
    independent_of_tested_system_claims: bool = True
    candidates: list[CandidateAtomicClaim]
    candidate_count: int = Field(ge=0)
    human_confirmed_claim_count: int | None = None
    formal_denominator: int | None = None
    requires_human_review: bool
    contains_ambiguity_flags: bool
    warnings: list[str]

    @model_validator(mode="after")
    def _candidate_count_matches(self) -> "CandidateSplitResult":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must equal len(candidates)")
        if self.formal_denominator is not None or self.human_confirmed_claim_count is not None:
            raise ValueError(
                "unreviewed splitter output cannot set a formal denominator or confirmed count"
            )
        if not self.requires_human_review:
            raise ValueError("every unfrozen split result requires human review before formal use")
        return self


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    boundary: SplitBoundary


_ABBREVIATIONS = {
    "al.",
    "dr.",
    "e.g.",
    "et.",
    "fig.",
    "i.e.",
    "mr.",
    "mrs.",
    "prof.",
    "ref.",
    "vs.",
}

_COORDINATION_RE = re.compile(
    r"(?:\b(?:and|or|as well as)\b|以及|并且|同时|且|和|及|或)", re.IGNORECASE
)
_CONTRAST_RE = re.compile(
    r"(?:\b(?:but|whereas|while|however|except|although|yet)\b|但是|但|而|然而|相反|除外|尽管)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:\b(?:not|no|neither|nor|without|failed to)\b|不|未|无|并非|不能|没有)",
    re.IGNORECASE,
)
_ANAPHORA_RE = re.compile(
    r"^(?:this|that|these|those|it|they|其|该|这些|上述|前者|后者|这种|此)(?:\b|一|项|种|个|些)",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"(?:\b(?:under|after|before|during|at|in the presence of|compared with|versus|vs\.)\b|"
    r"在.{0,24}(?:条件|刺激|处理|干预|组)|与.{0,18}(?:相比|比较)|剂量|时间|浓度)",
    re.IGNORECASE,
)
_DOSE_TIME_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:nM|uM|µM|mM|M|mg|g|ng|pg|h|hr|hrs|min|s|day|days|周|天|小时|分钟)\b",
    re.IGNORECASE,
)
_DIRECTION_PATTERNS = {
    "increase": re.compile(
        r"(?:\b(?:increase[sd]?|elevat(?:e[sd]?|ion)|upregulat(?:e[sd]?|ion)|promot(?:e[sd]?|ion)|enhanc(?:e[sd]?|ement))\b|增加|升高|上调|促进|增强)",
        re.IGNORECASE,
    ),
    "decrease": re.compile(
        r"(?:\b(?:decrease[sd]?|reduc(?:e[sd]?|tion)|lower(?:ed|s)?|downregulat(?:e[sd]?|ion)|inhibit(?:s|ed|ion)?|impair(?:s|ed|ment)?)\b|降低|减少|下调|抑制|削弱|损害)",
        re.IGNORECASE,
    ),
    "no_effect": re.compile(
        r"(?:\b(?:no (?:significant )?(?:effect|change|difference)|unchanged)\b|无显著(?:影响|变化|差异)|没有显著(?:影响|变化|差异))",
        re.IGNORECASE,
    ),
}


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    # List markers are layout rather than scientific content.  Remove only a
    # leading marker while retaining offsets into the original answer.
    marker = re.match(r"(?:[-*•]\s+|\(?\d{1,3}[.)、]\s*)", text[start:end])
    if marker:
        start += marker.end()
        while start < end and text[start].isspace():
            start += 1
    return (start, end) if start < end else None


def _is_period_boundary(text: str, index: int) -> bool:
    if index + 1 < len(text) and text[index + 1].isdigit() and index > 0 and text[index - 1].isdigit():
        return False
    token_start = index
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start : index + 1].lower().strip("([{\"'")
    if token in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", token):
        return False
    next_index = index + 1
    while next_index < len(text) and text[next_index] in "\"')]}”’":
        next_index += 1
    return next_index >= len(text) or text[next_index].isspace()


def _sentence_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        boundary: SplitBoundary | None = None
        end = index + 1
        if char in "。！？!?":
            boundary = SplitBoundary.SENTENCE
        elif char == "." and _is_period_boundary(text, index):
            boundary = SplitBoundary.SENTENCE
        elif char == "\n":
            # A non-empty line is an observable item boundary.  A wrapped line
            # without terminal punctuation is kept as a separate reviewable
            # candidate instead of being silently concatenated.
            boundary = SplitBoundary.LINE_OR_BULLET
            end = index
        if boundary is not None:
            trimmed = _trim_span(text, start, end)
            if trimmed:
                spans.append(_Span(*trimmed, boundary))
            start = index + 1
        index += 1
    trimmed = _trim_span(text, start, len(text))
    if trimmed:
        fallback = SplitBoundary.WHOLE_ANSWER if not spans else SplitBoundary.SENTENCE
        spans.append(_Span(*trimmed, fallback))
    return spans


def _clause_spans(text: str, sentence: _Span) -> list[_Span]:
    boundaries = [match.start() for match in re.finditer(r"[;；]", text[sentence.start : sentence.end])]
    if not boundaries:
        return [sentence]
    result: list[_Span] = []
    start = sentence.start
    for relative in boundaries:
        separator = sentence.start + relative
        trimmed = _trim_span(text, start, separator)
        if trimmed:
            result.append(_Span(*trimmed, SplitBoundary.SEMICOLON_CLAUSE))
        start = separator + 1
    trimmed = _trim_span(text, start, sentence.end)
    if trimmed:
        result.append(_Span(*trimmed, SplitBoundary.SEMICOLON_CLAUSE))
    return result or [sentence]


def _review_risks(text: str) -> list[ReviewRisk]:
    risks: list[ReviewRisk] = []
    if _COORDINATION_RE.search(text):
        risks.append(ReviewRisk.COORDINATED_PROPOSITIONS)
    direction_groups = [name for name, pattern in _DIRECTION_PATTERNS.items() if pattern.search(text)]
    direction_hits = sum(len(pattern.findall(text)) for pattern in _DIRECTION_PATTERNS.values())
    if len(direction_groups) > 1 or direction_hits > 1:
        risks.append(ReviewRisk.MULTIPLE_EFFECT_DIRECTIONS)
    condition_markers = len(_CONDITION_RE.findall(text))
    dose_or_time_values = len(_DOSE_TIME_RE.findall(text))
    if condition_markers > 1 or dose_or_time_values > 1:
        risks.append(ReviewRisk.MULTIPLE_CONDITIONS_OR_COMPARATORS)
    if _NEGATION_RE.search(text):
        risks.append(ReviewRisk.NEGATION_SCOPE)
    if _CONTRAST_RE.search(text):
        risks.append(ReviewRisk.CONTRAST_OR_EXCEPTION)
    if _ANAPHORA_RE.search(text.strip()):
        risks.append(ReviewRisk.ANAPHORA_OR_CONTEXT_DEPENDENCE)
    if len(text) > 320:
        risks.append(ReviewRisk.VERY_LONG_CANDIDATE)
    return list(dict.fromkeys(risks))


def _stable_id(output_id: str, start: int, end: int, exact: str) -> str:
    digest = hashlib.sha256(f"{output_id}\0{start}\0{end}\0{exact}".encode("utf-8")).hexdigest()
    return f"CC-{digest[:16]}"


def split_claim_candidates(request: ClaimSplitRequest) -> CandidateSplitResult:
    """Generate auditable claim candidates without deciding scientific meaning."""

    answer = request.answer
    candidates: list[CandidateAtomicClaim] = []
    for sentence_index, sentence in enumerate(_sentence_spans(answer)):
        sentence_exact = answer[sentence.start : sentence.end]
        sentence_id = f"S-{sentence_index + 1:04d}"
        for clause in _clause_spans(answer, sentence):
            exact = answer[clause.start : clause.end]
            risks = _review_risks(exact)
            prefix = answer[max(sentence.start, clause.start - 48) : clause.start]
            postfix = answer[clause.end : min(sentence.end, clause.end + 48)]
            boundary = clause.boundary
            reason = {
                SplitBoundary.SENTENCE: "observable sentence boundary",
                SplitBoundary.SEMICOLON_CLAUSE: "observable semicolon clause boundary",
                SplitBoundary.LINE_OR_BULLET: "observable line or bullet boundary",
                SplitBoundary.WHOLE_ANSWER: "single non-empty answer span",
            }[boundary]
            source = SourceSentenceAnchor(
                sentence_id=sentence_id,
                sentence_index=sentence_index,
                sentence_start_char=sentence.start,
                sentence_end_char=sentence.end,
                sentence_exact=sentence_exact,
                candidate_start_char=clause.start,
                candidate_end_char=clause.end,
                text_quote=TextAnchor(prefix=prefix, exact=exact, postfix=postfix),
            )
            candidates.append(
                CandidateAtomicClaim(
                    candidate_id=_stable_id(request.output_id, clause.start, clause.end, exact),
                    text=exact,
                    source=source,
                    split_boundary=boundary,
                    split_reason=reason,
                    review_risks=risks,
                    requires_human_review=bool(risks),
                )
            )

    warnings = [
        "Candidate count is not the formal claim denominator until human confirmation.",
        "Tested-system self-reported claims are not accepted as splitter input.",
        "Splitter is unfrozen until at least 20 outputs are independently calibrated and reviewed.",
    ]
    return CandidateSplitResult(
        output_id=request.output_id,
        question=request.question,
        answer_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        candidates=candidates,
        candidate_count=len(candidates),
        # Even an apparently simple candidate must be confirmed before it can
        # enter the formal denominator.  The separate flag below tells reviewers
        # whether the conservative rules also detected a known ambiguity.
        requires_human_review=True,
        contains_ambiguity_flags=any(item.requires_human_review for item in candidates),
        warnings=warnings,
    )


__all__ = [
    "CandidateAtomicClaim",
    "CandidateSplitResult",
    "ClaimSplitRequest",
    "ReviewRisk",
    "SourceSentenceAnchor",
    "SplitBoundary",
    "split_claim_candidates",
]
