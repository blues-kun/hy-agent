"""仅用于工程回归的离线模型，不得作为Hy3性能或科学结果。"""
from __future__ import annotations

import re

from app.schemas import (
    CorpusPassage,
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    SearchPlan,
)
from evaluator.schemas import Answerability


def _looks_out_of_scope(request: ReviewRequest) -> bool:
    lowered = request.question.casefold()
    clinical_markers = (
        "我父亲",
        "我母亲",
        "患者",
        "吃多少",
        "停掉",
        "停药",
        "剂量",
        "疗程",
        "metformin",
        "patient-specific",
    )
    return any(marker in lowered for marker in clinical_markers)


class OfflineSmokeModel:
    """确定性替身：验证编排、证据引用与审计文件，不生成科学结论。"""

    def plan(self, request: ReviewRequest) -> tuple[SearchPlan, ModelCallAudit]:
        # 冻结英文回退词保证对英文XML至少可产生候选；绝不把它称为模型检索计划。
        words = re.findall(r"[A-Za-z][A-Za-z0-9+-]+", request.question)
        query = " ".join(words[:12]) or "pancreatic beta cell mitochondria insulin secretion"
        answerability = request.answerability_hint or (
            Answerability.OUT_OF_SCOPE
            if _looks_out_of_scope(request)
            else Answerability.PARTIAL
        )
        plan = SearchPlan(
            queries=[query],
            source_pmids=request.source_pmids,
            rationale="OFFLINE_SMOKE：固定回退查询，仅验证流水线。",
            answerability_hint=answerability,
        )
        return plan, ModelCallAudit(
            stage="plan",
            provider="local-test-double",
            model="offline-smoke-v1",
            parse_source="offline_smoke",
        )

    def synthesize(
        self, request: ReviewRequest, passages: list[CorpusPassage]
    ) -> tuple[GeneratedReview, ModelCallAudit]:
        if request.answerability_hint is Answerability.OUT_OF_SCOPE or _looks_out_of_scope(request):
            review = GeneratedReview(
                answerability=Answerability.OUT_OF_SCOPE,
                answer="该问题超出科研证据综述边界；本工程回归不会生成个体化诊疗或用药建议。",
                claims=[],
                limitations=["OFFLINE_SMOKE：未调用Hy3，不可作为模型结果。"],
            )
        elif passages:
            first = passages[0]
            sentence = re.split(r"(?<=[.!?])\s+", first.text, maxsplit=1)[0]
            review = GeneratedReview(
                answerability=Answerability.PARTIAL,
                answer=(
                    "离线工程回归已检索到冻结全文段落并完成证据绑定；"
                    "本输出不尝试回答科研问题，正式答案必须由Hy3运行后再经评估与人工审核。"
                ),
                claims=[
                    GeneratedClaim(
                        claim_id="SMOKE-C1",
                        text=sentence,
                        is_core=False,
                        evidence_passage_ids=[first.passage_id],
                    )
                ],
                limitations=[
                    "OFFLINE_SMOKE：未调用Hy3。",
                    "引用句仅用于验证 Claim→EvidenceSpan→Judge 输入链路，不是科研结论。",
                ],
            )
        else:
            review = GeneratedReview(
                answerability=Answerability.INSUFFICIENT,
                answer="冻结语料中没有召回可用全文段落，因此停止生成科学结论。",
                claims=[],
                limitations=["OFFLINE_SMOKE：未调用Hy3；无证据时安全停止。"],
            )
        return review, ModelCallAudit(
            stage="synthesis",
            provider="local-test-double",
            model="offline-smoke-v1",
            parse_source="offline_smoke",
        )
