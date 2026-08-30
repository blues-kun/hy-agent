"""金标语料工具链（方案 9.1–9.3）。

职责：
  1. 装载并校验 questions.jsonl 的 QuestionGold 记录（Schema 即方案 9.2 字段清单）；
  2. 语料级完整性检查：question_id 唯一、span/paper/claim 的引用闭合、
     标识符语法合法（复用 D1 的规范化器）；
  3. 校准/盲测分集检查（方案 9.2：数据分 10 个校准问题与 30 个盲测问题，
     禁止同一论文的近似问题跨集合泄漏——程序可查的代理判据是
     「同一 DOI/PMID 不得同时出现在两个分集」，近似问题仍需人工判断）；
  4. 按方案 9.1 的口径输出构成统计（answerability 分布、难例信息由人工把关）。

纪律：本模块只做机械校验与统计，不裁决任何量表歧义；schema 之外的语义问题
（如 answerability 与主张列表是否自洽）只发 warning，不判 error。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import Field, ValidationError

from evaluator.rules.identifier_check import normalize_identifier
from evaluator.schemas import QuestionGold, StrictModel

VALID_SPLITS = ("calibration", "blind")


class GoldValidationReport(StrictModel):
    """一次语料校验的完整结论。errors 非空即语料不可用。"""

    n_records: int = 0
    errors: list[str] = Field(default_factory=list, description="必须修复才能使用的缺陷")
    warnings: list[str] = Field(default_factory=list, description="需人工确认但不阻断的问题")
    answerability_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    n_required_claims: int = 0
    n_evidence_papers: int = 0
    n_evidence_spans: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def load_gold_records(path: str | Path) -> tuple[list[QuestionGold], list[str]]:
    """逐行装载 JSONL。返回 (合规记录, 逐行错误)；单行失败不拖垮整个文件。"""
    records: list[QuestionGold] = []
    errors: list[str] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except ValueError as exc:
            errors.append(f"第 {lineno} 行：JSON 解析失败（{exc}）")
            continue
        try:
            records.append(QuestionGold.model_validate(payload))
        except ValidationError as exc:
            question_id = payload.get("question_id", "?") if isinstance(payload, dict) else "?"
            summary = "；".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]
            )
            errors.append(f"第 {lineno} 行（{question_id}）：Schema 校验失败（{summary}）")
    return records, errors


def _paper_keys(record: QuestionGold) -> set[str]:
    """一个问题触及的论文集合（规范化标识；供分集泄漏检查）。"""
    keys: set[str] = set()
    for paper in record.evidence_papers:
        norm = normalize_identifier(paper.doi_or_pmid)
        keys.add(norm.value or paper.doi_or_pmid.strip().lower())
    return keys


def check_record(record: QuestionGold) -> tuple[list[str], list[str]]:
    """单条记录的引用闭合与标识符语法检查。返回 (errors, warnings)。"""
    qid = record.question_id
    errors: list[str] = []
    warnings: list[str] = []

    paper_ids = [p.paper_id for p in record.evidence_papers]
    span_ids = [s.span_id for s in record.evidence_spans]
    claim_ids = [c.claim_id for c in record.required_claims + record.optional_claims]
    for label, values in (("paper_id", paper_ids), ("span_id", span_ids), ("claim_id", claim_ids)):
        duplicates = sorted({v for v, n in Counter(values).items() if n > 1})
        if duplicates:
            errors.append(f"{qid}: {label} 重复：{duplicates}")

    # 引用闭合：span → paper，claim.citation → span。
    paper_id_set, span_id_set = set(paper_ids), set(span_ids)
    for span in record.evidence_spans:
        if span.paper_id not in paper_id_set:
            errors.append(f"{qid}: 证据片段 {span.span_id} 的 paper_id={span.paper_id!r} 不在 evidence_papers 中")
    for claim in record.required_claims + record.optional_claims:
        for citation in claim.citations:
            missing = [s for s in citation.evidence_span_ids if s not in span_id_set]
            if missing:
                errors.append(f"{qid}: 主张 {claim.claim_id} 引用了不存在的 span_id：{missing}")

    # 标识符语法（复用 D1 规范化器；这里只查语法，不联网核验）。
    for owner, identifier in (
        [(f"论文 {p.paper_id}", p.doi_or_pmid) for p in record.evidence_papers]
        + [(f"片段 {s.span_id}", s.doi_or_pmid) for s in record.evidence_spans]
        + [
            (f"主张 {c.claim_id} 的引用", cit.doi_or_pmid)
            for c in record.required_claims + record.optional_claims
            for cit in c.citations
        ]
    ):
        norm = normalize_identifier(identifier)
        if not norm.is_valid:
            errors.append(f"{qid}: {owner} 的标识符语法非法：{identifier!r}（{norm.reason}）")

    # 语义自洽性只发 warning（不裁决方案未定义的口径）。
    has_conflict_paper = any(p.is_conflict_or_negative for p in record.evidence_papers)
    if record.known_conflicts and not has_conflict_paper:
        warnings.append(
            f"{qid}: known_conflicts 非空但没有任何论文标记 is_conflict_or_negative——请人工确认"
        )
    if has_conflict_paper and not record.known_conflicts:
        warnings.append(
            f"{qid}: 有论文标记 is_conflict_or_negative 但 known_conflicts 为空——请人工确认"
        )
    if record.answerability.value in ("insufficient", "out_of_scope") and record.required_claims:
        warnings.append(
            f"{qid}: answerability={record.answerability.value} 却给出了 required_claims——请人工确认"
        )
    return errors, warnings


def load_split_map(path: str | Path) -> tuple[dict[str, str], list[str]]:
    """分集文件：JSON 对象 {question_id: "calibration"|"blind"}。"""
    errors: list[str] = []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        return {}, [f"分集文件 JSON 解析失败：{exc}"]
    if not isinstance(raw, dict):
        return {}, ["分集文件必须是 {question_id: split} 的 JSON 对象"]
    split_map: dict[str, str] = {}
    for question_id, split in raw.items():
        if split not in VALID_SPLITS:
            errors.append(f"分集 {question_id!r} 的取值 {split!r} 非法（只允许 {VALID_SPLITS}）")
            continue
        split_map[str(question_id)] = str(split)
    return split_map, errors


def check_split_leakage(
    records: list[QuestionGold], split_map: dict[str, str]
) -> tuple[list[str], list[str]]:
    """方案 9.2：禁止同一论文的近似问题跨（校准/盲测）集合泄漏。

    程序可查的代理判据：同一论文（规范化 DOI/PMID）出现在两个分集 → error。
    「近似问题」的语义判断超出程序能力，仍需人工复核（warning 提示）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    unassigned = [r.question_id for r in records if r.question_id not in split_map]
    if unassigned:
        warnings.append(f"未分配分集的问题：{unassigned}")
    papers_by_split: dict[str, dict[str, list[str]]] = {s: {} for s in VALID_SPLITS}
    for record in records:
        split = split_map.get(record.question_id)
        if split is None:
            continue
        for key in _paper_keys(record):
            papers_by_split[split].setdefault(key, []).append(record.question_id)
    leaked = sorted(set(papers_by_split["calibration"]) & set(papers_by_split["blind"]))
    for key in leaked:
        errors.append(
            f"论文 {key} 跨分集泄漏：校准 {papers_by_split['calibration'][key]} "
            f"vs 盲测 {papers_by_split['blind'][key]}（方案 9.2 禁止）"
        )
    return errors, warnings


def validate_corpus(
    records: list[QuestionGold],
    line_errors: list[str] | None = None,
    split_map: dict[str, str] | None = None,
) -> GoldValidationReport:
    """语料级校验入口。"""
    errors = list(line_errors or [])
    warnings: list[str] = []

    duplicate_ids = sorted(
        {qid for qid, n in Counter(r.question_id for r in records).items() if n > 1}
    )
    if duplicate_ids:
        errors.append(f"question_id 重复：{duplicate_ids}")

    for record in records:
        record_errors, record_warnings = check_record(record)
        errors.extend(record_errors)
        warnings.extend(record_warnings)

    split_counts: dict[str, int] = {}
    if split_map is not None:
        leak_errors, leak_warnings = check_split_leakage(records, split_map)
        errors.extend(leak_errors)
        warnings.extend(leak_warnings)
        split_counts = dict(
            Counter(split_map[r.question_id] for r in records if r.question_id in split_map)
        )

    return GoldValidationReport(
        n_records=len(records),
        errors=errors,
        warnings=warnings,
        answerability_counts=dict(Counter(r.answerability.value for r in records)),
        split_counts=split_counts,
        n_required_claims=sum(len(r.required_claims) for r in records),
        n_evidence_papers=sum(len(r.evidence_papers) for r in records),
        n_evidence_spans=sum(len(r.evidence_spans) for r in records),
    )
