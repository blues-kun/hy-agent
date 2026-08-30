"""TokenHub（Hy3）OpenAI 兼容客户端。

全部按 2026-08-28 实测事实实现（scripts/hy3_smoke_test.py 九步探针），不按公开
文档想当然：

  - 模型级上限 RPM 60 → 内置 ≤1 请求/秒节流 + 429 指数退避（尊重 Retry-After）；
  - 默认开启思考：所有请求显式传 reasoning_effort（Judge 用 high），思考 token
    计入 max_tokens（预算给足，默认 4096）；
  - 思考 + 工具调用的多轮消息必须逐轮回填 reasoning_content（官方硬性要求）；
  - 结构化输出主通道 Function Calling（emit_judge_verdict，tool_choice=auto——
    交错式思考模式下 tool_choice 仅支持 auto）；备选通道 response_format
    json_schema+strict（实测约束解码强制生效但无文档，只留配置开关）；
    无论哪条通道，返回都过 schemas.JudgeVerdict 本地校验 + 有界修复重试；
  - logprobs 被静默忽略（实测）：置信不用 token 概率，走自一致性（self_consistency.py）；
  - Prompt Cache：请求体 prompt_cache_key + Header X-Session-ID；命中量记
    usage.prompt_tokens_details（cached_token / cached_tokens 两个键名都查），
    思考开销记 usage.completion_tokens_details.reasoning_tokens。

网络环境：本机 http_proxy/https_proxy 可能指向已失效代理，默认 trust_env=False
直连。API Key 只从环境变量读取（config.resolve_api_key），绝不落盘。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, ValidationError

from evaluator.judge.config import JudgeConfig, default_judge_config
from evaluator.judge.prompts import build_messages
from evaluator.schemas import AtomicClaim, EvidenceSpan, JudgeVerdict, StrictModel, SupportVerdict

USER_AGENT = "MitoEvidence-Hy3-judge/0.2"

# 模型侧输出 Schema：与 schemas.JudgeVerdict 对齐（claim_id 由调用方注入，模型不回显）。
JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in SupportVerdict]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "evidence_span_refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

EMIT_JUDGE_VERDICT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_judge_verdict",
        "description": "输出结构化的主张—证据判定结果（五值判定 + 置信 + 理由 + 依据的证据片段）",
        "parameters": JUDGE_OUTPUT_SCHEMA,
    },
}

_PAYLOAD_KEYS = ("verdict", "confidence", "reason", "evidence_span_refs")


# ---------------------------------------------------------------------------
# usage 记账
# ---------------------------------------------------------------------------


class TokenUsage(StrictModel):
    """一次或多次请求的 token 开销汇总。"""

    n_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = Field(
        default=0, description="usage.completion_tokens_details.reasoning_tokens 累计"
    )
    cached_tokens: int = Field(
        default=0, description="usage.prompt_tokens_details.cached_token(s) 累计（Prompt Cache 命中）"
    )

    def merged(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            n_requests=self.n_requests + other.n_requests,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    @property
    def cache_hit_rate(self) -> float | None:
        if self.prompt_tokens <= 0:
            return None
        return self.cached_tokens / self.prompt_tokens


def usage_from_body(body: Any) -> TokenUsage:
    usage = (body or {}).get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    # 文档转述为 cached_token，OpenAI 惯例为 cached_tokens：两个键都查（实测脚本同策）。
    cached = prompt_details.get("cached_token", prompt_details.get("cached_tokens")) or 0
    return TokenUsage(
        n_requests=1,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        cached_tokens=int(cached),
    )


# ---------------------------------------------------------------------------
# 传输层：节流 + 退避
# ---------------------------------------------------------------------------


@dataclass
class Hy3Transport:
    """POST /chat/completions 执行器。session 与 sleep_fn 可注入以便离线测试。"""

    api_key: str = ""
    base_url: str = ""
    session: Any = None
    max_rps: float = 1.0  # 模型级 RPM 60（实测）
    max_retries: int = 4
    timeout: float = 180.0
    sleep_fn: Callable[[float], None] = time.sleep
    trust_env: bool = False
    _last_call: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.session is None:
            import requests

            self.session = requests.Session()
            self.session.trust_env = self.trust_env  # 本机死代理：默认直连
        headers = getattr(self.session, "headers", None)
        if hasattr(headers, "update"):
            headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": USER_AGENT,
                }
            )

    def _throttle(self) -> None:
        interval = 1.0 / self.max_rps if self.max_rps > 0 else 0.0
        wait = interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self.sleep_fn(wait)
        self._last_call = time.monotonic()

    def post_chat(
        self, payload: Mapping[str, Any], extra_headers: Mapping[str, str] | None = None
    ) -> tuple[int, Any, str]:
        """返回 (status, 响应 JSON 或 None, 错误说明)。错误说明非空即本次调用失败。"""
        url = f"{self.base_url}/chat/completions"
        last_error = "未发起请求"
        for attempt in range(self.max_retries):
            is_last = attempt == self.max_retries - 1
            self._throttle()
            try:
                response = self.session.request(
                    "POST", url, json=dict(payload), headers=dict(extra_headers or {}),
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 —— 传输层异常统一按可重试失败处理
                last_error = f"传输失败（{type(exc).__name__}: {exc}）"
                if not is_last:
                    self.sleep_fn(min(2.0**attempt, 8.0))
                continue

            status = int(getattr(response, "status_code", 0))
            if status == 429:
                retry_after = (getattr(response, "headers", {}) or {}).get("Retry-After")
                delay = (
                    float(retry_after)
                    if str(retry_after or "").replace(".", "", 1).isdigit()
                    else 2.0 ** (attempt + 1)
                )
                last_error = f"HTTP 429 限流，退避 {delay:.1f}s"
                if not is_last:
                    self.sleep_fn(delay)
                continue
            if 500 <= status < 600:
                last_error = f"HTTP {status} 服务端错误"
                if not is_last:
                    self.sleep_fn(min(2.0**attempt, 8.0))
                continue
            try:
                body = response.json()
            except Exception as exc:  # noqa: BLE001 —— 非 JSON 响应无法给出可信结论
                return status, None, f"响应非 JSON（{type(exc).__name__}: {exc}）"
            if status != 200:
                # 401/402/400 等客户端错误：重试无意义，带回错误体摘要。
                detail = str(body)[:200] if body else ""
                return status, body, f"HTTP {status}：{detail}"
            return status, body, ""
        return -1, None, f"重试 {self.max_retries} 次后仍失败：{last_error}"


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------


def _message_of(body: Any) -> dict[str, Any]:
    try:
        return dict(body["choices"][0]["message"] or {})
    except (KeyError, IndexError, TypeError):
        return {}


def parse_json_loose(text: str) -> Any:
    """宽松 JSON 提取：容忍 ```json 围栏与前后缀文本。"""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except ValueError:
        return None


def _extract_payload(message: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """从响应消息提取判定 payload。返回 (payload, 来源标签)。

    优先工具通道；tool_choice=auto 下模型可能不调工具而在正文输出 JSON，
    此时回退宽松解析正文（仍会过本地校验）。
    """
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if function.get("name") != "emit_judge_verdict":
            continue
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            return None, "tool_arguments_invalid_json"
        if isinstance(arguments, dict):
            return arguments, "tool_call"
        return None, "tool_arguments_not_object"
    data = parse_json_loose(message.get("content") or "")
    if isinstance(data, dict):
        return data, "content_json"
    return None, "no_parsable_output"


def validate_judge_payload(
    payload: Mapping[str, Any], claim_id: str, allowed_span_ids: set[str]
) -> JudgeVerdict:
    """本地 Schema 校验（方案 5.3：模型侧结构约束不视为安全边界）。

    只投影已知字段；evidence_span_refs 引用未提供的 span_id 视为幻觉，直接拒绝。
    校验失败抛 ValueError / ValidationError，由调用方做有界修复重试。
    """
    data = {key: payload[key] for key in _PAYLOAD_KEYS if key in payload}
    refs = data.get("evidence_span_refs") or []
    if not isinstance(refs, list):
        raise ValueError(f"evidence_span_refs 必须是数组，得到 {type(refs).__name__}")
    unknown_refs = [r for r in refs if r not in allowed_span_ids]
    if unknown_refs:
        raise ValueError(f"evidence_span_refs 引用了未提供的 span_id（幻觉引用）：{unknown_refs}")
    return JudgeVerdict(claim_id=claim_id, **data)


def _repair_messages(
    messages: list[dict[str, Any]], assistant_message: Mapping[str, Any], error_text: str
) -> list[dict[str, Any]]:
    """构造修复轮消息。思考模式硬性要求：assistant 消息必须回填 reasoning_content。"""
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.get("content") or "",
    }
    if assistant_message.get("reasoning_content") is not None:
        assistant["reasoning_content"] = assistant_message["reasoning_content"]
    tool_calls = assistant_message.get("tool_calls") or []
    follow_ups: list[dict[str, Any]]
    if tool_calls:
        assistant["tool_calls"] = tool_calls
        # 协议要求每个 tool_call_id 都要有对应的 tool 消息。
        follow_ups = [
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": (
                    f"参数未通过本地 Schema 校验：{error_text}。"
                    "请重新调用 emit_judge_verdict，严格符合参数 Schema。"
                ),
            }
            for tool_call in tool_calls
        ]
    else:
        follow_ups = [
            {
                "role": "user",
                "content": f"输出未通过本地 Schema 校验：{error_text}。请按【输出方式】要求重新输出。",
            }
        ]
    return [*messages, assistant, *follow_ups]


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class JudgeCallResult(StrictModel):
    """一次 judge_once 调用的结果（含修复重试的累计开销）。"""

    ok: bool
    verdict: JudgeVerdict | None = None
    error: str = ""
    parse_source: str = Field(default="", description="tool_call / content_json")
    response_sha256: str = Field(default="", description="最终响应消息哈希（方案 8.4 复现记录）")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    temperature: float | None = None
    seed: int | None = None


class Hy3Client:
    """Hy3 Judge 客户端。transport/session 可注入，离线测试不发真实请求。"""

    def __init__(
        self,
        config: JudgeConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        session: Any = None,
        transport: Hy3Transport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        cfg = config or default_judge_config()
        self.config = cfg
        self.model = model or cfg.resolve_model()
        self.channel = str(cfg.structured_output["channel"])
        self.max_parse_retries = int(cfg.structured_output["max_parse_retries"])
        self.transport = transport or Hy3Transport(
            api_key=api_key if api_key is not None else cfg.resolve_api_key(),
            base_url=base_url or cfg.resolve_base_url(),
            session=session,
            max_rps=float(cfg.transport["max_rps"]),
            max_retries=int(cfg.transport["max_retries"]),
            timeout=float(cfg.transport["timeout_s"]),
            sleep_fn=sleep_fn,
            trust_env=bool(cfg.transport["trust_env"]),
        )

    # -- 请求组装 -------------------------------------------------------------

    def _base_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None,
        seed: int | None,
    ) -> dict[str, Any]:
        request_cfg = self.config.request
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # 实测默认开启思考：所有请求显式传 reasoning_effort；思考 token 计入 max_tokens。
            "reasoning_effort": str(request_cfg["reasoning_effort"]),
            "max_tokens": int(request_cfg["max_tokens"]),
            "temperature": (
                float(request_cfg["temperature"]) if temperature is None else float(temperature)
            ),
        }
        effective_seed = request_cfg.get("seed") if seed is None else seed
        if effective_seed is not None:
            payload["seed"] = int(effective_seed)
        cache_cfg = self.config.prompt_cache
        if bool(cache_cfg.get("enabled", False)):
            payload["prompt_cache_key"] = str(cache_cfg["cache_key"])
        if self.channel == "function_calling":
            payload["tools"] = [EMIT_JUDGE_VERDICT_TOOL]
            # 交错式思考模式下 tool_choice 仅支持 auto（官方文档 + 实测）。
            payload["tool_choice"] = str(self.config.structured_output.get("tool_choice", "auto"))
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_verdict",
                    "strict": True,
                    "schema": JUDGE_OUTPUT_SCHEMA,
                },
            }
        return payload

    def _headers(self) -> dict[str, str]:
        cache_cfg = self.config.prompt_cache
        if not bool(cache_cfg.get("enabled", False)):
            return {}
        return {str(cache_cfg["session_header"]): str(cache_cfg["session_id"])}

    # -- 判定 ----------------------------------------------------------------

    def judge_once(
        self,
        claim: AtomicClaim,
        spans: list[EvidenceSpan],
        question: str = "",
        temperature: float | None = None,
        seed: int | None = None,
    ) -> JudgeCallResult:
        """对单个原子主张做一次判定。解析/校验失败时做有界修复重试。"""
        messages = build_messages(claim, spans, question, channel=self.channel)
        allowed_span_ids = {span.span_id for span in spans}
        usage = TokenUsage()
        effective_temperature = (
            float(self.config.request["temperature"]) if temperature is None else float(temperature)
        )
        last_error = ""

        for _attempt in range(1 + self.max_parse_retries):
            payload = self._base_payload(messages, temperature, seed)
            status, body, error = self.transport.post_chat(payload, self._headers())
            if body is not None:
                usage = usage.merged(usage_from_body(body))
            if error or status != 200:
                # HTTP/传输失败不做解析重试：由 transport 内部已重试到界。
                return JudgeCallResult(
                    ok=False,
                    error=error or f"HTTP {status}",
                    usage=usage,
                    temperature=effective_temperature,
                    seed=seed,
                )
            message = _message_of(body)
            extracted, source = _extract_payload(message)
            if extracted is None:
                last_error = f"无法提取判定输出（{source}）"
            else:
                try:
                    verdict = validate_judge_payload(extracted, claim.claim_id, allowed_span_ids)
                except (ValueError, ValidationError) as exc:
                    last_error = f"本地校验失败：{exc}"
                else:
                    digest = hashlib.sha256(
                        json.dumps(message, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    return JudgeCallResult(
                        ok=True,
                        verdict=verdict,
                        parse_source=source,
                        response_sha256=digest,
                        usage=usage,
                        temperature=effective_temperature,
                        seed=seed,
                    )
            messages = _repair_messages(messages, message, last_error)

        return JudgeCallResult(
            ok=False,
            error=f"{1 + self.max_parse_retries} 次尝试后输出仍不合规：{last_error}",
            usage=usage,
            temperature=effective_temperature,
            seed=seed,
        )
