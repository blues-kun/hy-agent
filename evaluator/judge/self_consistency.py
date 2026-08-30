"""自一致性置信：k 次采样多数票 + 一致率（方案 7.3 / 8.4）。

为什么用采样一致率而不是 token 概率：logprobs 被 TokenHub 静默忽略（实测），
token 级置信不可用；且实测 temperature=0 时判定标签一致率 100%、采样无信息量，
因此自一致性必须 temperature>0（默认 0.7）。

聚合口径（实现选择，见 eval/rubric.md「待澄清 N」）：
  - agreement_rate = 多数票票数 / k，分母含采样失败——失败视为不一致（保守方向）；
  - 多数票并列时取更保守的判定（unknown > refuted > not_supported >
    partially_supported > fully_supported）并强制升级人工；
  - agreement 票数 < min_agreement_votes（默认 5/7）→ escalate_to_human=true；
  - 多数票为 refuted（方向冲突）时按方案 8.4「方向冲突必须人工复核」强制升级。
"""
from __future__ import annotations

from collections import Counter

from pydantic import Field

from evaluator.judge.config import JudgeConfig, default_judge_config
from evaluator.judge.hy3_client import Hy3Client, JudgeCallResult, TokenUsage
from evaluator.schemas import AtomicClaim, EvidenceSpan, JudgeVerdict, StrictModel, SupportVerdict

# 并列时的保守优先序：越靠前越「不给分」，宁可少给分也不虚报支持。
CONSERVATIVE_ORDER = (
    SupportVerdict.UNKNOWN,
    SupportVerdict.REFUTED,
    SupportVerdict.NOT_SUPPORTED,
    SupportVerdict.PARTIALLY_SUPPORTED,
    SupportVerdict.FULLY_SUPPORTED,
)


class JudgeSample(StrictModel):
    """一次采样的记录（方案 8.4：重复评测须记录随机参数和响应哈希）。"""

    index: int
    ok: bool
    verdict: JudgeVerdict | None = None
    error: str = ""
    parse_source: str = ""
    response_sha256: str = ""
    temperature: float | None = None
    seed: int | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class JudgeAggregate(StrictModel):
    """单个原子主张的自一致性聚合结果。"""

    claim_id: str
    k: int
    n_valid: int = Field(description="成功返回合规判定的采样数")
    votes: dict[str, int] = Field(default_factory=dict, description="verdict → 票数")
    final_verdict: SupportVerdict
    final: JudgeVerdict = Field(description="最终判定对象；confidence = agreement_rate")
    agreement_rate: float = Field(ge=0.0, le=1.0, description="多数票票数 / k")
    escalate_to_human: bool
    escalate_reasons: list[str] = Field(default_factory=list)
    samples: list[JudgeSample] = Field(default_factory=list)
    usage_total: TokenUsage = Field(default_factory=TokenUsage)


def aggregate_samples(
    claim_id: str,
    samples: list[JudgeSample],
    k: int,
    min_agreement_votes: int,
    escalate_on_refuted: bool = True,
) -> JudgeAggregate:
    """多数票聚合。纯函数，离线可测。"""
    usage_total = TokenUsage()
    for sample in samples:
        usage_total = usage_total.merged(sample.usage)
    valid = [s for s in samples if s.ok and s.verdict is not None]
    n_failed = len(samples) - len(valid)

    if not valid:
        final = JudgeVerdict(
            claim_id=claim_id,
            verdict=SupportVerdict.UNKNOWN,
            confidence=0.0,
            reason="k 次采样全部失败，无法判定",
        )
        return JudgeAggregate(
            claim_id=claim_id,
            k=k,
            n_valid=0,
            votes={},
            final_verdict=SupportVerdict.UNKNOWN,
            final=final,
            agreement_rate=0.0,
            escalate_to_human=True,
            escalate_reasons=[f"{n_failed}/{len(samples)} 次采样全部失败，无任何合规判定"],
            samples=samples,
            usage_total=usage_total,
        )

    counts = Counter(s.verdict.verdict for s in valid)
    top_count = max(counts.values())
    tied = [v for v, c in counts.items() if c == top_count]
    # 并列：取保守优先序里最靠前者，并强制人工升级。
    final_label = min(tied, key=CONSERVATIVE_ORDER.index)
    agreement_rate = top_count / k

    escalate_reasons: list[str] = []
    if len(tied) > 1:
        escalate_reasons.append(
            "多数票并列（" + "、".join(sorted(v.value for v in tied)) + "），取保守判定并升级人工"
        )
    if top_count < min_agreement_votes:
        detail = f"（其中 {n_failed} 次采样失败计为不一致）" if n_failed else ""
        escalate_reasons.append(
            f"一致票数 {top_count}/{k} 低于阈值 {min_agreement_votes}{detail}"
            "（低置信须人工复核，方案 8.4）"
        )
    if escalate_on_refuted and final_label is SupportVerdict.REFUTED:
        escalate_reasons.append("多数票为 refuted：方向冲突必须人工复核（方案 8.4）")

    majority_samples = [s for s in valid if s.verdict.verdict is final_label]
    representative = majority_samples[0].verdict
    merged_refs = sorted({ref for s in majority_samples for ref in s.verdict.evidence_span_refs})
    final = JudgeVerdict(
        claim_id=claim_id,
        verdict=final_label,
        confidence=round(agreement_rate, 4),
        reason=f"自一致性多数票 {top_count}/{k}；代表理由：{representative.reason}",
        evidence_span_refs=merged_refs,
    )
    escalate = bool(escalate_reasons)
    return JudgeAggregate(
        claim_id=claim_id,
        k=k,
        n_valid=len(valid),
        votes={v.value: c for v, c in sorted(counts.items(), key=lambda item: item[0].value)},
        final_verdict=final_label,
        final=final,
        agreement_rate=round(agreement_rate, 4),
        escalate_to_human=escalate,
        escalate_reasons=escalate_reasons,
        samples=samples,
        usage_total=usage_total,
    )


def run_self_consistency(
    client: Hy3Client,
    claim: AtomicClaim,
    spans: list[EvidenceSpan],
    question: str = "",
    k: int | None = None,
    temperature: float | None = None,
    base_seed: int | None = None,
    config: JudgeConfig | None = None,
) -> JudgeAggregate:
    """对单个原子主张做 k 次采样并聚合。

    k 次请求共享完全相同的消息（稳定 system 前缀 + 同一 user 消息），
    自第 2 次起 Prompt Cache 应命中；命中量在 usage_total.cached_tokens 记账。
    """
    cfg = config or client.config or default_judge_config()
    sc = cfg.self_consistency
    effective_k = int(sc["k"]) if k is None else int(k)
    effective_temperature = float(sc["temperature"]) if temperature is None else float(temperature)
    effective_base_seed = sc.get("base_seed") if base_seed is None else base_seed
    min_votes = int(sc["min_agreement_votes"])
    if k is not None and k != int(sc["k"]):
        # 调用方改 k 时按同比例折算票数阈值（向上取整），保持 5/7 的相对口径。
        min_votes = -(-min_votes * effective_k // int(sc["k"]))

    samples: list[JudgeSample] = []
    for index in range(effective_k):
        seed = None if effective_base_seed is None else int(effective_base_seed) + index
        result: JudgeCallResult = client.judge_once(
            claim, spans, question=question, temperature=effective_temperature, seed=seed
        )
        samples.append(
            JudgeSample(
                index=index,
                ok=result.ok,
                verdict=result.verdict,
                error=result.error,
                parse_source=result.parse_source,
                response_sha256=result.response_sha256,
                temperature=result.temperature,
                seed=result.seed,
                usage=result.usage,
            )
        )
    return aggregate_samples(
        claim.claim_id,
        samples,
        k=effective_k,
        min_agreement_votes=min_votes,
        escalate_on_refuted=bool(sc.get("escalate_on_refuted", True)),
    )
