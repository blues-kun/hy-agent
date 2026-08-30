"""L2 语义 Judge 的离线测试。

全部用 mock session，不发起任何真实请求、不产生真实等待；
真实联网冒烟由 scripts/run_judge.py 单独执行。
"""
from __future__ import annotations

import copy
import json

import pytest
import yaml

from evaluator.judge.config import (
    DEFAULT_JUDGE_CONFIG_PATH,
    JudgeConfig,
    JudgeConfigError,
    default_judge_config,
)
from evaluator.judge.hy3_client import (
    EMIT_JUDGE_VERDICT_TOOL,
    JUDGE_OUTPUT_SCHEMA,
    Hy3Client,
    Hy3Transport,
    TokenUsage,
    usage_from_body,
    validate_judge_payload,
)
from evaluator.judge.prompts import (
    EVIDENCE_BEGIN,
    EVIDENCE_END,
    build_messages,
    build_user_message,
    system_prefix,
)
from evaluator.judge.self_consistency import (
    JudgeSample,
    aggregate_samples,
    run_self_consistency,
)
from evaluator.schemas import (
    AtomicClaim,
    EvidenceSpan,
    JudgeVerdict,
    SourceAccess,
    SupportVerdict,
    TextAnchor,
)

# ---------------------------------------------------------------------------
# 固定素材与测试替身
# ---------------------------------------------------------------------------

CLAIM = AtomicClaim(
    claim_id="C1",
    text="在 INS-1E 细胞中，33.3 mM 高糖处理 24 小时增加线粒体碎片化。",
    is_core=True,
    conditions={
        "species": "rat",
        "cell_type": "INS-1E",
        "dose": "33.3 mM",
        "time": "24 h",
        "effect_direction": "increase",
    },
)
SPAN = EvidenceSpan(
    span_id="S1",
    paper_id="P1",
    doi_or_pmid="10.9999/sample.frag.2025",
    section="Results",
    source_access=SourceAccess.FULLTEXT,
    anchor=TextAnchor(
        prefix="为评估慢性高糖的作用，",
        exact="INS-1E 细胞暴露于 33.3 mM 葡萄糖 24 小时后，线粒体网络碎片化指数显著升高（p<0.01）",
        postfix="，同时膜电位无显著变化。",
    ),
)
SPAN_META_ONLY = EvidenceSpan(
    span_id="S9",
    paper_id="P9",
    doi_or_pmid="10.9999/meta.only",
    source_access=SourceAccess.METADATA_ONLY,
)

GOOD_ARGS = {
    "verdict": "fully_supported",
    "confidence": 0.9,
    "reason": "原文同时支持细胞类型、剂量、时间与效应方向。",
    "evidence_span_refs": ["S1"],
}

USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "completion_tokens_details": {"reasoning_tokens": 30},
    "prompt_tokens_details": {"cached_tokens": 80},
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = {} if payload is None else payload

    def json(self):
        return self._payload


class FakeSession:
    """responses 为 FakeResponse/异常的列表，或 call → FakeResponse 的可调用对象。"""

    def __init__(self, responses):
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []
        self._responses = responses

    def request(self, method, url, json=None, headers=None, timeout=None):
        call = {"method": method, "url": url, "json": json, "headers": headers, "timeout": timeout}
        self.calls.append(call)
        if callable(self._responses):
            item = self._responses(call)
        elif self._responses:
            item = self._responses.pop(0)
        else:
            item = FakeResponse(200, {})
        if isinstance(item, BaseException):
            raise item
        return item


def tool_call_body(arguments: dict, reasoning_content="内部思考……", usage=None) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": reasoning_content,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "emit_judge_verdict",
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": usage or USAGE,
    }


def content_body(text: str, usage=None) -> dict:
    return {"choices": [{"message": {"content": text}}], "usage": usage or USAGE}


def make_client(responses, config=None, **kwargs) -> tuple[Hy3Client, FakeSession, list[float]]:
    sleeps: list[float] = []
    session = FakeSession(responses)
    client = Hy3Client(
        config=config or default_judge_config(),
        api_key="dummy-key-for-tests",
        base_url="https://tokenhub.example/v1",
        model="hy3",
        session=session,
        sleep_fn=sleeps.append,
        **kwargs,
    )
    return client, session, sleeps


def sample(index: int, verdict_label: str | None, ok: bool = True, refs=("S1",)) -> JudgeSample:
    verdict = None
    if verdict_label is not None:
        needs_refs = verdict_label in ("fully_supported", "partially_supported", "refuted")
        verdict = JudgeVerdict(
            claim_id="C1",
            verdict=SupportVerdict(verdict_label),
            confidence=0.9,
            reason=f"样本 {index}",
            evidence_span_refs=list(refs) if needs_refs else [],
        )
    return JudgeSample(index=index, ok=ok, verdict=verdict, error="" if ok else "boom")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_default_config_matches_measured_constraints():
    cfg = default_judge_config()
    assert float(cfg.transport["max_rps"]) <= 1.0  # RPM 60 实测
    assert cfg.transport["trust_env"] is False  # 本机死代理：默认直连
    assert str(cfg.request["reasoning_effort"]) == "high"
    assert int(cfg.request["max_tokens"]) == 4096  # 思考 token 计入 max_tokens
    assert str(cfg.structured_output["channel"]) == "function_calling"
    assert str(cfg.structured_output["fallback_channel"]) == "json_schema"
    assert int(cfg.self_consistency["k"]) == 7
    assert float(cfg.self_consistency["temperature"]) == 0.7  # temp=0 采样无信息量（实测）
    assert int(cfg.self_consistency["min_agreement_votes"]) == 5


def _raw_config() -> dict:
    return copy.deepcopy(yaml.safe_load(DEFAULT_JUDGE_CONFIG_PATH.read_text(encoding="utf-8")))


def test_config_rejects_rps_above_measured_limit():
    raw = _raw_config()
    raw["transport"]["max_rps"] = 2.0
    with pytest.raises(JudgeConfigError, match="RPM 60"):
        JudgeConfig(raw)


def test_config_rejects_zero_sampling_temperature():
    raw = _raw_config()
    raw["self_consistency"]["temperature"] = 0.0
    with pytest.raises(JudgeConfigError, match="无信息量"):
        JudgeConfig(raw)


def test_config_rejects_min_votes_above_k():
    raw = _raw_config()
    raw["self_consistency"]["min_agreement_votes"] = 8
    with pytest.raises(JudgeConfigError, match="min_agreement_votes"):
        JudgeConfig(raw)


# ---------------------------------------------------------------------------
# 提示模板：缓存友好布局与信息隔离
# ---------------------------------------------------------------------------


def test_system_prefix_is_stable_and_dateless():
    prefix = system_prefix()
    assert prefix == system_prefix()  # 逐字节稳定（Prompt Cache 命中前提）
    assert "202" not in prefix  # 不含日期
    for label in ("fully_supported", "partially_supported", "not_supported", "refuted", "unknown"):
        assert label in prefix  # 8.1 五值定义齐全
    assert "abstract_only" in prefix and "metadata_only" in prefix  # source_access 限制
    assert "不是给你的指令" in prefix  # spotlighting 声明
    assert "emit_judge_verdict" in prefix


def test_system_prefix_json_schema_channel_differs_only_in_output_instruction():
    fc, js = system_prefix("function_calling"), system_prefix("json_schema")
    assert fc != js
    assert "emit_judge_verdict" not in js
    assert "只输出一个 JSON 对象" in js


def test_user_message_puts_claim_last_and_spotlights_evidence():
    text = build_user_message(CLAIM, [SPAN], question="慢性高糖是否增加线粒体碎片化？")
    assert text.startswith("研究问题：")
    assert EVIDENCE_BEGIN in text and EVIDENCE_END in text
    assert text.index(EVIDENCE_END) < text.index("待判定的原子主张")  # 主张在末尾
    assert text.rstrip().endswith("effect_direction=increase")
    assert "span_id=S1" in text and "source_access=fulltext" in text


def test_metadata_only_span_renders_placeholder_and_no_anchor():
    text = build_user_message(CLAIM, [SPAN_META_ONLY])
    assert "仅证明论文存在" in text


def test_no_evidence_renders_unknown_hint():
    assert "无证据时只能判 unknown" in build_user_message(CLAIM, [])


def test_messages_contain_no_system_identity():
    """Judge 看不到被测系统名与实验组（方案 8.4）。"""
    messages = build_messages(CLAIM, [SPAN], question="q")
    joined = json.dumps(messages, ensure_ascii=False)
    for forbidden in ("系统 A", "system_id", "实验组", "output_id"):
        assert forbidden not in joined


# ---------------------------------------------------------------------------
# 传输层：节流与退避
# ---------------------------------------------------------------------------


def test_transport_throttles_to_configured_rps():
    session = FakeSession(lambda call: FakeResponse(200, content_body("{}")))
    sleeps: list[float] = []
    transport = Hy3Transport(
        api_key="k", base_url="https://x/v1", session=session, max_rps=1.0, sleep_fn=sleeps.append
    )
    transport.post_chat({"model": "hy3"})
    transport.post_chat({"model": "hy3"})
    # 第二次请求距第一次不足 1s，必须补足节流等待（RPM 60 → ≤1 rps）。
    assert any(0.5 <= wait <= 1.0 for wait in sleeps)


def test_transport_backs_off_on_429_then_succeeds():
    responses = [FakeResponse(429), tool_call_body(GOOD_ARGS)]

    def responder(call):
        item = responses.pop(0)
        return item if isinstance(item, FakeResponse) else FakeResponse(200, item)

    session = FakeSession(responder)
    sleeps: list[float] = []
    transport = Hy3Transport(
        api_key="k", base_url="https://x/v1", session=session, max_rps=1.0, sleep_fn=sleeps.append
    )
    status, body, error = transport.post_chat({"model": "hy3"})
    assert (status, error) == (200, "")
    assert 2.0 in sleeps  # 429 指数退避 2^(attempt+1)


def test_transport_honours_retry_after_header():
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"})] * 4)
    sleeps: list[float] = []
    transport = Hy3Transport(
        api_key="k", base_url="https://x/v1", session=session, max_rps=0, sleep_fn=sleeps.append
    )
    status, _, error = transport.post_chat({"model": "hy3"})
    assert status == -1 and "429" in error
    assert 7.0 in sleeps


def test_transport_client_error_is_not_retried():
    session = FakeSession([FakeResponse(401, {"error": "invalid api key"})])
    transport = Hy3Transport(
        api_key="bad", base_url="https://x/v1", session=session, max_rps=0, sleep_fn=lambda _: None
    )
    status, _, error = transport.post_chat({"model": "hy3"})
    assert status == 401 and "401" in error
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# 结构化输出：双通道解析 + 本地校验 + 有界修复重试
# ---------------------------------------------------------------------------


def test_judge_once_parses_function_calling_channel():
    client, session, _ = make_client([FakeResponse(200, tool_call_body(GOOD_ARGS))])
    result = client.judge_once(CLAIM, [SPAN])

    assert result.ok is True
    assert result.parse_source == "tool_call"
    assert result.verdict.verdict is SupportVerdict.FULLY_SUPPORTED
    assert result.verdict.claim_id == "C1"  # claim_id 由本地注入，模型不回显
    assert result.verdict.evidence_span_refs == ["S1"]
    assert result.response_sha256
    payload = session.calls[0]["json"]
    assert payload["tools"] == [EMIT_JUDGE_VERDICT_TOOL]
    assert payload["tool_choice"] == "auto"  # 交错式思考模式仅支持 auto
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 4096
    assert payload["prompt_cache_key"] == "mitoevidence-judge-v0_1"
    assert session.calls[0]["headers"]["X-Session-ID"] == "mitoevidence-judge-v0_1"


def test_judge_once_usage_accounting_reads_details():
    client, _, _ = make_client([FakeResponse(200, tool_call_body(GOOD_ARGS))])
    result = client.judge_once(CLAIM, [SPAN])
    assert result.usage.n_requests == 1
    assert result.usage.prompt_tokens == 100
    assert result.usage.reasoning_tokens == 30  # completion_tokens_details
    assert result.usage.cached_tokens == 80  # prompt_tokens_details（Prompt Cache 命中）
    assert result.usage.cache_hit_rate == pytest.approx(0.8)


def test_usage_reads_cached_token_singular_key_too():
    body = {"usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_token": 4}}}
    assert usage_from_body(body).cached_tokens == 4


def test_judge_once_falls_back_to_content_json_when_no_tool_call():
    body = content_body("```json\n" + json.dumps(GOOD_ARGS, ensure_ascii=False) + "\n```")
    client, _, _ = make_client([FakeResponse(200, body)])
    result = client.judge_once(CLAIM, [SPAN])
    assert result.ok is True
    assert result.parse_source == "content_json"


def test_invalid_enum_triggers_repair_with_reasoning_refill():
    bad = dict(GOOD_ARGS, verdict="support")  # 非五值枚举
    client, session, _ = make_client(
        [FakeResponse(200, tool_call_body(bad)), FakeResponse(200, tool_call_body(GOOD_ARGS))]
    )
    result = client.judge_once(CLAIM, [SPAN])

    assert result.ok is True
    assert len(session.calls) == 2
    repair_messages = session.calls[1]["json"]["messages"]
    assistant = next(m for m in repair_messages if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "内部思考……"  # 思考+工具调用逐轮回填
    assert assistant["tool_calls"]
    tool_reply = next(m for m in repair_messages if m["role"] == "tool")
    assert tool_reply["tool_call_id"] == "call_1"
    assert "校验" in tool_reply["content"]


def test_hallucinated_span_ref_is_rejected_after_bounded_retries():
    bad = dict(GOOD_ARGS, evidence_span_refs=["S99"])
    client, session, _ = make_client(lambda call: FakeResponse(200, tool_call_body(bad)))
    result = client.judge_once(CLAIM, [SPAN])
    assert result.ok is False
    assert "幻觉引用" in result.error
    # 1 次首发 + max_parse_retries 次修复，之后放弃（有界）。
    assert len(session.calls) == 1 + client.max_parse_retries
    assert result.usage.n_requests == len(session.calls)


def test_supported_verdict_without_refs_fails_local_validation():
    bad = dict(GOOD_ARGS, evidence_span_refs=[])
    client, _, _ = make_client(lambda call: FakeResponse(200, tool_call_body(bad)))
    result = client.judge_once(CLAIM, [SPAN])
    assert result.ok is False
    assert "校验" in result.error


def test_unknown_verdict_without_refs_is_valid():
    args = {"verdict": "unknown", "confidence": 0.3, "reason": "证据片段不足。"}
    client, _, _ = make_client([FakeResponse(200, tool_call_body(args))])
    result = client.judge_once(CLAIM, [SPAN])
    assert result.ok is True
    assert result.verdict.verdict is SupportVerdict.UNKNOWN


def test_transport_failure_is_not_parse_retried():
    client, session, _ = make_client([ConnectionError("代理连接被拒")] * 8)
    result = client.judge_once(CLAIM, [SPAN])
    assert result.ok is False
    assert "传输失败" in result.error
    # transport 内部重试 4 次后放弃；judge_once 不再做解析级重试。
    assert len(session.calls) == 4


def test_json_schema_fallback_channel_builds_response_format():
    raw = _raw_config()
    raw["structured_output"]["channel"] = "json_schema"
    cfg = JudgeConfig(raw)
    body = content_body(json.dumps(GOOD_ARGS, ensure_ascii=False))
    client, session, _ = make_client([FakeResponse(200, body)], config=cfg)
    result = client.judge_once(CLAIM, [SPAN])

    assert result.ok is True
    payload = session.calls[0]["json"]
    assert "tools" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"] == JUDGE_OUTPUT_SCHEMA


def test_validate_judge_payload_projects_known_keys_only():
    payload = dict(GOOD_ARGS, extra_key="ignored")
    verdict = validate_judge_payload(payload, "C1", {"S1"})
    assert verdict.verdict is SupportVerdict.FULLY_SUPPORTED


# ---------------------------------------------------------------------------
# 自一致性聚合：多数票、边界 4/7 与 5/7、并列、refuted 升级
# ---------------------------------------------------------------------------


def test_unanimous_seven_of_seven():
    samples = [sample(i, "fully_supported") for i in range(7)]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.FULLY_SUPPORTED
    assert agg.agreement_rate == 1.0
    assert agg.escalate_to_human is False
    assert agg.final.confidence == 1.0
    assert agg.votes == {"fully_supported": 7}


def test_five_of_seven_is_exactly_at_threshold_no_escalation():
    samples = [sample(i, "fully_supported") for i in range(5)] + [
        sample(5, "partially_supported"),
        sample(6, "unknown", refs=()),
    ]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.FULLY_SUPPORTED
    assert agg.agreement_rate == pytest.approx(5 / 7, abs=1e-4)
    assert agg.escalate_to_human is False  # 5/7 恰好达标


def test_four_of_seven_escalates():
    samples = [sample(i, "fully_supported") for i in range(4)] + [
        sample(i, "not_supported", refs=()) for i in range(4, 7)
    ]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.FULLY_SUPPORTED
    assert agg.agreement_rate == pytest.approx(4 / 7, abs=1e-4)
    assert agg.escalate_to_human is True  # 4/7 低于 5/7 阈值
    assert any("低于阈值" in reason for reason in agg.escalate_reasons)


def test_tie_takes_conservative_verdict_and_escalates():
    samples = (
        [sample(i, "fully_supported") for i in range(3)]
        + [sample(i, "not_supported", refs=()) for i in range(3, 6)]
        + [sample(6, "unknown", refs=())]
    )
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.NOT_SUPPORTED  # 并列取保守方向
    assert agg.escalate_to_human is True
    assert any("并列" in reason for reason in agg.escalate_reasons)


def test_refuted_majority_escalates_even_when_unanimous():
    samples = [sample(i, "refuted") for i in range(7)]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.REFUTED
    assert agg.agreement_rate == 1.0
    assert agg.escalate_to_human is True  # 方向冲突必须人工复核（方案 8.4）
    assert any("refuted" in reason for reason in agg.escalate_reasons)


def test_failed_samples_count_against_agreement():
    samples = [sample(i, "fully_supported") for i in range(4)] + [
        sample(i, None, ok=False) for i in range(4, 7)
    ]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.n_valid == 4
    assert agg.agreement_rate == pytest.approx(4 / 7, abs=1e-4)  # 失败计为不一致（保守）
    assert agg.escalate_to_human is True


def test_all_failed_yields_unknown_and_escalates():
    samples = [sample(i, None, ok=False) for i in range(7)]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final_verdict is SupportVerdict.UNKNOWN
    assert agg.final.confidence == 0.0
    assert agg.escalate_to_human is True


def test_final_verdict_merges_majority_evidence_refs():
    samples = [
        sample(0, "fully_supported", refs=("S1",)),
        sample(1, "fully_supported", refs=("S2",)),
        sample(2, "fully_supported", refs=("S1", "S2")),
        sample(3, "fully_supported", refs=("S1",)),
        sample(4, "fully_supported", refs=("S1",)),
        sample(5, "unknown", refs=()),
        sample(6, "unknown", refs=()),
    ]
    agg = aggregate_samples("C1", samples, k=7, min_agreement_votes=5)
    assert agg.final.evidence_span_refs == ["S1", "S2"]


# ---------------------------------------------------------------------------
# run_self_consistency：端到端（mock 客户端）
# ---------------------------------------------------------------------------


def make_sc_client(responder) -> tuple[Hy3Client, FakeSession]:
    session = FakeSession(responder)
    client = Hy3Client(
        config=default_judge_config(),
        api_key="dummy",
        base_url="https://tokenhub.example/v1",
        model="hy3",
        session=session,
        sleep_fn=lambda _: None,
    )
    return client, session


def test_run_self_consistency_k7_majority_and_usage_totals():
    outputs = (
        [tool_call_body(GOOD_ARGS)] * 5
        + [tool_call_body({"verdict": "unknown", "confidence": 0.4, "reason": "不确定。"})] * 2
    )
    client, session = make_sc_client(lambda call: FakeResponse(200, outputs.pop(0)))

    agg = run_self_consistency(client, CLAIM, [SPAN], base_seed=42)

    assert agg.k == 7 and len(agg.samples) == 7
    assert agg.final_verdict is SupportVerdict.FULLY_SUPPORTED
    assert agg.agreement_rate == pytest.approx(5 / 7, abs=1e-4)
    assert agg.escalate_to_human is False
    assert [s.seed for s in agg.samples] == list(range(42, 49))  # 记录随机参数（方案 8.4）
    assert all(s.temperature == 0.7 for s in agg.samples)  # temp=0 无信息量 → 0.7
    assert agg.usage_total.n_requests == 7
    assert agg.usage_total.cached_tokens == 7 * 80
    # k 次采样共享同一消息（Prompt Cache 命中前提）
    payloads = [call["json"]["messages"] for call in session.calls]
    assert all(p == payloads[0] for p in payloads)


def test_run_self_consistency_scales_threshold_when_k_overridden():
    """k=5 时阈值按 5/7 比例向上取整 → 4/5。"""
    outputs = [tool_call_body(GOOD_ARGS)] * 4 + [
        tool_call_body({"verdict": "unknown", "confidence": 0.4, "reason": "不确定。"})
    ]
    client, _ = make_sc_client(lambda call: FakeResponse(200, outputs.pop(0)))
    agg = run_self_consistency(client, CLAIM, [SPAN], k=5)
    assert agg.k == 5
    assert agg.agreement_rate == pytest.approx(4 / 5, abs=1e-4)
    assert agg.escalate_to_human is False  # 4/5 达到折算阈值 4


def test_token_usage_merge():
    a = TokenUsage(n_requests=1, prompt_tokens=10, completion_tokens=5, reasoning_tokens=3, cached_tokens=8)
    b = TokenUsage(n_requests=2, prompt_tokens=20, completion_tokens=10, reasoning_tokens=6, cached_tokens=0)
    merged = a.merged(b)
    assert merged.n_requests == 3
    assert merged.prompt_tokens == 30
    assert merged.reasoning_tokens == 9
    assert merged.cache_hit_rate == pytest.approx(8 / 30)
