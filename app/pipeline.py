"""端到端编排：问题 → Hy3计划 → 冻结语料检索 → 综述 → Judge输入与审计包。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.corpus import FrozenReviewCorpus
from app.schemas import (
    AnchorCheck,
    CorpusPassage,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    ReviewRunArtifact,
    RunKind,
    SearchPlan,
)
from evaluator.schemas import AtomicClaim, Citation, EvidenceSpan, SourceAccess, TextAnchor
from tools.literature.xml_anchor import EpmcXmlDocument


class ReviewModel(Protocol):
    def plan(self, request: ReviewRequest) -> tuple[SearchPlan, ModelCallAudit]: ...

    def synthesize(
        self, request: ReviewRequest, passages: list[CorpusPassage]
    ) -> tuple[GeneratedReview, ModelCallAudit]: ...


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"非法 run_id：{value!r}")
    return cleaned


def load_pilot_request(path: str | Path, question_id: str) -> ReviewRequest:
    """从预标文件构造中性应用输入。

    只读取题目与范围；绝不把 AI 预标的 answerability、source_reviews、
    prohibited_inferences 或 required_claims 喂给被测应用，否则会造成标签泄漏。
    """
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("question_id") != question_id:
            continue
        return ReviewRequest(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            scope=str(row.get("scope") or ""),
        )
    raise ValueError(f"{path} 中找不到 question_id={question_id!r}")


class ReviewRunner:
    def __init__(
        self,
        *,
        model: ReviewModel,
        corpus: FrozenReviewCorpus,
        run_kind: RunKind = RunKind.HY3,
    ):
        self.model = model
        self.corpus = corpus
        self.run_kind = run_kind

    def run(self, request: ReviewRequest, *, top_k: int = 12) -> ReviewRunArtifact:
        plan, plan_audit = self.model.plan(request)
        requested_pmids = plan.source_pmids or request.source_pmids
        unavailable = sorted(set(requested_pmids) - self.corpus.available_pmids)
        warnings: list[str] = []
        if unavailable:
            warnings.append(
                "以下约束综述没有本地OA XML，仅能在后续联网/人工阶段补证据："
                + ", ".join(f"PMID:{p}" for p in unavailable)
            )
        # 越界问题不检索科学证据，避免用检索结果诱导出临床建议。
        effective_answerability = request.answerability_hint or plan.answerability_hint
        is_out_of_scope = effective_answerability.value == "out_of_scope"
        passages = (
            []
            if is_out_of_scope
            else self.corpus.search(
                plan.queries,
                source_pmids=requested_pmids,
                top_k=top_k,
            )
        )
        if not passages and not is_out_of_scope:
            warnings.append("冻结语料未召回全文段落；应用必须降级为证据不足，不能凭模型记忆补写。")
        # 这里只把模型自己的检索计划判断传给综合阶段，并不读取专家金标。
        # 它使综合阶段能遵守越界拒答约束，同时保持被测输入与金标隔离。
        synthesis_request = request.model_copy(
            update={"answerability_hint": effective_answerability}
        )
        review, synthesis_audit = self.model.synthesize(synthesis_request, passages)
        if is_out_of_scope and review.answerability.value != "out_of_scope":
            raise ValueError("显式越界问题未被模型拒答；为防止临床建议泄漏，整次运行失败并进入人工复核")
        if is_out_of_scope and review.claims:
            raise ValueError("越界拒答不得附带科学或诊疗主张；请删除 claims 后重试")
        if (
            not passages
            and not is_out_of_scope
            and review.answerability.value in {"answerable", "partial"}
        ):
            raise ValueError("冻结语料无可用证据，但模型仍尝试作答；禁止凭模型记忆生成科学结论")
        known = {passage.passage_id for passage in passages}
        unknown = sorted(
            {
                ref
                for claim in review.claims
                for ref in claim.evidence_passage_ids
                if ref not in known
            }
        )
        if unknown:
            raise ValueError(f"模型引用了未提供的 passage_id（证据幻觉）：{unknown}")
        if review.answerability.value in {"answerable", "partial"}:
            ungrounded = [claim.claim_id for claim in review.claims if not claim.evidence_passage_ids]
            if ungrounded:
                warnings.append("存在无证据主张，必须进入人工复核：" + ", ".join(ungrounded))
        status = (
            "offline_engineering_smoke_not_model_result"
            if self.run_kind is RunKind.OFFLINE_SMOKE
            else "hy3_run_pending_expert_gold_scoring"
        )
        try:
            manifest_label = str(self.corpus.manifest_path.relative_to(self.corpus.repo_root))
        except ValueError:
            # 自定义测试 manifest 可能位于仓库根之外；只保留文件名，避免泄露主机路径。
            manifest_label = self.corpus.manifest_path.name
        artifact = ReviewRunArtifact(
            evidence_manifest_path=manifest_label,
            evidence_manifest_sha256=self.corpus.manifest_sha256,
            request=request,
            plan=plan,
            passages=passages,
            review=review,
            model_calls=[plan_audit, synthesis_audit],
            warnings=warnings,
            run_kind=self.run_kind,
            formal_status=status,
        )
        anchor_checks = self._validate_anchors(artifact)
        bad_anchors = [item for item in anchor_checks if item.status != "found"]
        if bad_anchors:
            warnings = list(artifact.warnings)
            warnings.append(
                "存在无法唯一回到冻结XML的证据锚点，必须人工复核："
                + ", ".join(f"{item.span_id}={item.status}" for item in bad_anchors)
            )
            artifact = artifact.model_copy(update={"warnings": warnings})
        return artifact.model_copy(update={"anchor_checks": anchor_checks})

    @staticmethod
    def _evidence_spans(artifact: ReviewRunArtifact) -> dict[str, EvidenceSpan]:
        used = {ref for claim in artifact.review.claims for ref in claim.evidence_passage_ids}
        output: dict[str, EvidenceSpan] = {}
        for passage in artifact.passages:
            if passage.passage_id not in used:
                continue
            output[passage.passage_id] = EvidenceSpan(
                span_id=passage.passage_id,
                paper_id=passage.paper_id,
                doi_or_pmid=f"PMID:{passage.pmid}",
                section=passage.section,
                anchor=TextAnchor(
                    prefix=passage.prefix,
                    exact=passage.anchor_exact or passage.text,
                    postfix=passage.postfix,
                ),
                source_access=SourceAccess.FULLTEXT,
            )
        return output

    def _validate_anchors(self, artifact: ReviewRunArtifact) -> list[AnchorCheck]:
        """把所有实际引用的锚点重新定位到 manifest 冻结的原始 XML。"""
        spans = self._evidence_spans(artifact)
        passage_by_id = {item.passage_id: item for item in artifact.passages}
        documents: dict[str, EpmcXmlDocument] = {}
        checks: list[AnchorCheck] = []
        for span_id, span in sorted(spans.items()):
            passage = passage_by_id[span_id]
            try:
                document = documents.get(passage.source_path)
                if document is None:
                    source = (self.corpus.repo_root / passage.source_path).resolve()
                    if self.corpus.repo_root not in source.parents:
                        raise ValueError(f"证据路径越界：{passage.source_path}")
                    document = EpmcXmlDocument.from_xml(source.read_bytes())
                    documents[passage.source_path] = document
                result = document.relocate_evidence_span(span)
                checks.append(
                    AnchorCheck(
                        span_id=span_id,
                        source_path=passage.source_path,
                        source_sha256=passage.source_sha256,
                        status=result.status.value,
                        candidate_count=result.candidate_count,
                        readable_position=(
                            result.selected.location.readable_position
                            if result.selected is not None
                            else None
                        ),
                        reason=result.reason,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 失败必须显式进入审计包
                checks.append(
                    AnchorCheck(
                        span_id=span_id,
                        source_path=passage.source_path,
                        source_sha256=passage.source_sha256,
                        status="error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
        return checks

    @classmethod
    def judge_inputs(cls, artifact: ReviewRunArtifact) -> list[dict]:
        spans = cls._evidence_spans(artifact)
        passages = {p.passage_id: p for p in artifact.passages}
        rows: list[dict] = []
        for generated in artifact.review.claims:
            by_paper: dict[str, list[str]] = {}
            for span_id in generated.evidence_passage_ids:
                passage = passages[span_id]
                by_paper.setdefault(passage.pmid, []).append(span_id)
            citations = [
                Citation(
                    doi_or_pmid=f"PMID:{pmid}",
                    paper_id=f"PMID:{pmid}",
                    evidence_span_ids=span_ids,
                )
                for pmid, span_ids in sorted(by_paper.items())
            ]
            claim = AtomicClaim(
                claim_id=generated.claim_id,
                text=generated.text,
                is_core=generated.is_core,
                conditions=generated.conditions,
                citations=citations,
            )
            rows.append(
                {
                    "question": artifact.request.question,
                    "claim": claim.model_dump(mode="json"),
                    "evidence_spans": [
                        spans[span_id].model_dump(mode="json")
                        for span_id in generated.evidence_passage_ids
                    ],
                }
            )
        return rows

    @staticmethod
    def _review_markdown(artifact: ReviewRunArtifact) -> str:
        lines = [
            f"# {artifact.request.question_id} · 可追溯快速证据综述",
            "",
            f"> 状态：`{artifact.formal_status}`。未经已确认专家金标比对和结果审核，不得作为正式科学结论。",
            "",
            "## 回答",
            "",
            artifact.review.answer,
            "",
            "## 原子主张—证据映射",
            "",
            "| Claim | 核心 | 原子主张 | Evidence passage IDs |",
            "|---|---|---|---|",
        ]
        for claim in artifact.review.claims:
            text = claim.text.replace("|", "\\|").replace("\n", " ")
            refs = ", ".join(f"`{r}`" for r in claim.evidence_passage_ids) or "**无证据**"
            lines.append(f"| {claim.claim_id} | {'是' if claim.is_core else '否'} | {text} | {refs} |")
        lines.extend(["", "## 局限性", ""])
        lines.extend(f"- {item}" for item in artifact.review.limitations)
        if artifact.warnings:
            lines.extend(["", "## 运行警告", ""])
            lines.extend(f"- {item}" for item in artifact.warnings)
        lines.extend(["", "## 检索计划", "", f"- 查询：{'; '.join(artifact.plan.queries)}"])
        if artifact.plan.source_pmids:
            lines.append("- 约束综述：" + ", ".join(f"PMID:{p}" for p in artifact.plan.source_pmids))
        return "\n".join(lines) + "\n"

    @classmethod
    def write_run(
        cls,
        artifact: ReviewRunArtifact,
        *,
        out_root: str | Path,
        run_id: str | None = None,
    ) -> Path:
        out_root = Path(out_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        final_name = _safe_run_id(run_id or f"{artifact.request.question_id}-{stamp}")
        final_dir = out_root / final_name
        if final_dir.exists():
            raise FileExistsError(f"run目录已存在：{final_dir}")
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_name}-", dir=out_root))
        try:
            run_data = artifact.model_dump(mode="json")
            files: dict[str, bytes] = {
                "run.json": _json_bytes(run_data),
                "plan.json": _json_bytes(artifact.plan.model_dump(mode="json")),
                "retrieval.jsonl": b"".join(
                    json.dumps(p.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
                    + b"\n"
                    for p in artifact.passages
                ),
                "review.json": _json_bytes(artifact.review.model_dump(mode="json")),
                "review.md": cls._review_markdown(artifact).encode("utf-8"),
                "anchor_validation.jsonl": b"".join(
                    json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8")
                    + b"\n"
                    for item in artifact.anchor_checks
                ),
                "judge_input.jsonl": b"".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
                    for row in cls.judge_inputs(artifact)
                ),
            }
            for name, data in files.items():
                (temp_dir / name).write_bytes(data)
            manifest = {
                "run_id": final_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "application_version": artifact.application_version,
                "formal_status": artifact.formal_status,
                "run_kind": artifact.run_kind.value,
                "question_id": artifact.request.question_id,
                "evidence_manifest": {
                    "path": artifact.evidence_manifest_path,
                    "sha256": artifact.evidence_manifest_sha256,
                },
                "files": {name: {"bytes": len(data), "sha256": _sha256(data)} for name, data in files.items()},
                "source_xml": {
                    p.source_path: p.source_sha256 for p in artifact.passages
                },
                "model_calls": [item.model_dump(mode="json") for item in artifact.model_calls],
                "anchor_summary": {
                    status: sum(1 for item in artifact.anchor_checks if item.status == status)
                    for status in ("found", "ambiguous", "not_found", "error")
                },
                "security": {"contains_api_key": False, "contains_reasoning_content": False},
            }
            manifest_data = _json_bytes(manifest)
            (temp_dir / "manifest.json").write_bytes(manifest_data)
            os.replace(temp_dir, final_dir)
        except Exception:
            for path in sorted(temp_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if temp_dir.exists():
                temp_dir.rmdir()
            raise
        return final_dir
