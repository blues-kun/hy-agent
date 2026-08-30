"""Secret-safe serialization helpers for persisted experiment artifacts.

Exception strings and provider-returned free text are untrusted.  They may
contain request URLs, echoed Authorization headers or opaque credentials.  A
manifest may only claim ``contains_api_key=false`` after values have passed
these sanitizers (or an equivalent no-change assertion).
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


REDACTED_SECRET = "<REDACTED_SECRET>"
REDACTED_BEARER = "<REDACTED_BEARER>"
REDACTED_QUERY = "REDACTED_QUERY"
REDACTED_FRAGMENT = "REDACTED_FRAGMENT"
REDACTED_LONG_TOKEN = "<REDACTED_LONG_TOKEN>"
REDACTED_MODEL_REASONING = "<REDACTED_MODEL_REASONING>"

_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>]+", re.IGNORECASE)
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|secret|token|password)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_LONG_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9._~+/=-]{39,})(?![A-Za-z0-9])"
)
_JSON_REASONING_RE = re.compile(
    r'(?is)(["\']reasoning_content["\']\s*:\s*)["\'](?:\\.|[^"\'])*["\']'
)
_PLAIN_REASONING_RE = re.compile(r"(?is)\breasoning_content\s*[:=]\s*[^\r\n]*")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAILING_URL_PUNCTUATION = ".,;:!)]}"


class ArtifactSecurityError(ValueError):
    """Raised when a value would leak sensitive material if persisted."""


def _known_secrets(extra: Sequence[str] = ()) -> list[str]:
    values = [os.getenv("HY3_API_KEY", ""), *extra]
    return sorted({value for value in values if value}, key=len, reverse=True)


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    suffix = ""
    while raw and raw[-1] in _TRAILING_URL_PUNCTUATION:
        suffix = raw[-1] + suffix
        raw = raw[:-1]
    try:
        parts = urlsplit(raw)
        host = parts.hostname or "<REDACTED_HOST>"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = f":{parts.port}" if parts.port is not None else ""
        except ValueError:
            port = ""
        safe = f"{parts.scheme}://{host}{port}{parts.path}"
        if parts.query:
            safe += f"?{REDACTED_QUERY}"
        if parts.fragment:
            safe += f"#{REDACTED_FRAGMENT}"
        return safe + suffix
    except Exception:
        return "<REDACTED_URL>" + suffix


def redact_sensitive_text(
    value: str,
    *,
    extra_secrets: Sequence[str] = (),
    preserve_sha256: bool = False,
    redact_long_tokens: bool = True,
) -> str:
    """Redact known keys, auth headers, URL credentials/query and opaque tokens."""

    redacted = value
    for secret in _known_secrets(extra_secrets):
        redacted = redacted.replace(secret, REDACTED_SECRET)
    redacted = _BEARER_RE.sub(f"Bearer {REDACTED_BEARER}", redacted)
    redacted = _URL_RE.sub(_redact_url, redacted)
    redacted = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED_SECRET}",
        redacted,
    )

    def redact_long(match: re.Match[str]) -> str:
        token = match.group(1)
        if preserve_sha256 and _SHA256_RE.fullmatch(token):
            return token
        return REDACTED_LONG_TOKEN

    return _LONG_TOKEN_RE.sub(redact_long, redacted) if redact_long_tokens else redacted


def sanitize_failure_text(value: str, *, extra_secrets: Sequence[str] = ()) -> str:
    """Sanitize an exception without ever persisting model reasoning text."""

    without_json_reasoning = _JSON_REASONING_RE.sub(
        lambda match: f'{match.group(1)}"{REDACTED_MODEL_REASONING}"',
        value,
    )
    without_reasoning = _PLAIN_REASONING_RE.sub(
        f"reasoning_content={REDACTED_MODEL_REASONING}",
        without_json_reasoning,
    )
    return redact_sensitive_text(without_reasoning, extra_secrets=extra_secrets)[:4000]


def _redact_reasoning_text(value: str) -> str:
    """Remove private-reasoning fields embedded in otherwise successful data."""

    without_json_reasoning = _JSON_REASONING_RE.sub(
        lambda match: f'{match.group(1)}"{REDACTED_MODEL_REASONING}"',
        value,
    )
    return _PLAIN_REASONING_RE.sub(
        f"reasoning_content={REDACTED_MODEL_REASONING}",
        without_json_reasoning,
    )


def sanitize_json_value(
    value: Any,
    *,
    extra_secrets: Sequence[str] = (),
    field_name: str | None = None,
) -> Any:
    """Recursively sanitize a JSON-compatible object without changing hashes."""

    if isinstance(value, Mapping):
        return {
            key: sanitize_json_value(
                item,
                extra_secrets=extra_secrets,
                field_name=(
                    field_name
                    if field_name == "blind_input_sha256_by_review_id"
                    else str(key)
                ),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_json_value(item, extra_secrets=extra_secrets, field_name=field_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            sanitize_json_value(item, extra_secrets=extra_secrets, field_name=field_name)
            for item in value
        ]
    if isinstance(value, str):
        if field_name and field_name.lower() == "reasoning_content":
            return REDACTED_MODEL_REASONING
        preserve_hash = bool(field_name and field_name.lower().endswith("sha256"))
        # These schema-controlled values are long by design and cannot carry
        # an opaque token without first changing a separately validated
        # contract/path field. Known secrets, headers and URL credentials are
        # still redacted; only the generic long-token heuristic is skipped.
        schema_long_fields = {
            "formal_status",
            "selection_algorithm",
            "order_algorithm",
            "sampling_seed_policy",
            "seed_policy",
            "shared_plan_policy",
            "evidence_budget_policy",
            "cache_namespace",
            "endpoint_origin",
            "endpoint_url",
            "retrieval",
            "method",
            "construction_source",
            "rule",
            "prompt_hash_scope",
            "response_hash_scope",
            "structured_output_hash_scope",
            "cache_namespace",
            "paper_id",
            "limitations",
        }
        schema_long = bool(
            field_name
            and (
                field_name in schema_long_fields
                or field_name == "blind_input_sha256_by_review_id"
                or field_name.endswith("_path")
                or field_name.endswith("_hash_scope")
                or field_name == "cell_dir"
            )
        )
        return redact_sensitive_text(
            _redact_reasoning_text(value),
            extra_secrets=extra_secrets,
            preserve_sha256=preserve_hash,
            redact_long_tokens=not schema_long,
        )
    return value


def assert_json_safe(value: Any, *, extra_secrets: Sequence[str] = ()) -> None:
    """Refuse persistence when sanitization would alter the value."""

    sanitized = sanitize_json_value(value, extra_secrets=extra_secrets)
    if sanitized != value:
        raise ArtifactSecurityError(
            "artifact/state contains credential-like text; sanitize before persistence"
        )
