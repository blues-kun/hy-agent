"""Frozen execution identity for formal and offline Pilot artifacts."""
from __future__ import annotations

from enum import Enum
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from app.schemas import ModelCallAudit
from evaluator.schemas import StrictModel


FORMAL_HY3_MODEL = "hy3"
FORMAL_TOKENHUB_HOSTS = frozenset({"tokenhub.tencentmaas.com"})


class PilotExecutionKind(str, Enum):
    REMOTE_HY3 = "remote_hy3"
    OFFLINE_FIXTURE = "offline_fixture"


class PilotExecutionIdentity(StrictModel):
    execution_kind: PilotExecutionKind
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint_origin: str = ""
    endpoint_url: str = ""

    @model_validator(mode="after")
    def _safe_closed_identity(self) -> "PilotExecutionIdentity":
        if self.execution_kind is PilotExecutionKind.OFFLINE_FIXTURE:
            if self.provider != "offline-fixture":
                raise ValueError("offline fixture provider 必须固定为 offline-fixture")
            if self.endpoint_origin or self.endpoint_url:
                raise ValueError("offline fixture 不得声明远程 endpoint")
            return self
        if self.provider != "tencent-tokenhub":
            raise ValueError("formal remote_hy3 provider 必须是 tencent-tokenhub")
        if self.model != FORMAL_HY3_MODEL:
            raise ValueError("formal remote_hy3 model 必须精确为 hy3")
        origin = urlsplit(self.endpoint_origin)
        endpoint = urlsplit(self.endpoint_url)
        for name, value in (("endpoint_origin", origin), ("endpoint_url", endpoint)):
            if (
                value.scheme != "https"
                or not value.hostname
                or value.username is not None
                or value.password is not None
                or value.query
                or value.fragment
            ):
                raise ValueError(f"formal {name} 必须是无凭据/query/fragment的HTTPS URL")
        expected_origin = f"{endpoint.scheme}://{endpoint.netloc}"
        if self.endpoint_origin != expected_origin or origin.path not in {"", "/"}:
            raise ValueError("endpoint_origin 与 endpoint_url origin 不一致")
        if not endpoint.path.endswith("/chat/completions"):
            raise ValueError("formal endpoint_url 必须指向 chat/completions")
        if endpoint.hostname not in FORMAL_TOKENHUB_HOSTS:
            raise ValueError("formal endpoint_url 必须属于已确认的腾讯 TokenHub 官方域名")
        return self


def is_formal_hy3_metadata(metadata: object) -> bool:
    """Return whether provider metadata satisfies the frozen formal allowlist."""

    if not isinstance(metadata, dict):
        return False
    try:
        PilotExecutionIdentity.model_validate(
            {
                "execution_kind": metadata.get("execution_kind"),
                "provider": metadata.get("provider"),
                "model": metadata.get("model"),
                "endpoint_origin": metadata.get("endpoint_origin", ""),
                "endpoint_url": metadata.get("endpoint_url", ""),
            }
        )
    except (TypeError, ValueError):
        return False
    return True


def _identity_from_provider_metadata(metadata: object) -> PilotExecutionIdentity:
    if is_formal_hy3_metadata(metadata):
        assert isinstance(metadata, dict)
        return PilotExecutionIdentity.model_validate(
            {key: metadata.get(key) for key in (
                "execution_kind", "provider", "model", "endpoint_origin", "endpoint_url"
            )}
        )
    model = metadata.get("model", "custom-model") if isinstance(metadata, dict) else "custom-model"
    return PilotExecutionIdentity(
        execution_kind=PilotExecutionKind.OFFLINE_FIXTURE,
        provider="offline-fixture",
        model=str(model),
    )


def audit_matches_identity(audit: ModelCallAudit, identity: PilotExecutionIdentity) -> bool:
    return (
        audit.provider == identity.provider
        and audit.model == identity.model
        and audit.endpoint_origin == identity.endpoint_origin
        and audit.endpoint_url == identity.endpoint_url
    )


def identity_from_structured_client(client: object) -> PilotExecutionIdentity:
    audit_identity = getattr(client, "audit_identity", None)
    if audit_identity is not None:
        metadata = audit_identity() if callable(audit_identity) else audit_identity
        if not isinstance(metadata, dict):
            raise ValueError("audit_identity 必须是字典")
        return _identity_from_provider_metadata(metadata)
    metadata_fn = getattr(client, "execution_metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
        return _identity_from_provider_metadata(metadata)
    # The repository's audited Hy3 structured client predates the explicit
    # metadata hook.  Derive only from its concrete transport, never merely
    # from a caller-supplied model name.
    if (
        type(client).__module__ == "app.hy3_review"
        and type(client).__name__ == "Hy3ReviewModel"
        and hasattr(client, "transport")
    ):
        base_url = str(getattr(getattr(client, "transport"), "base_url", "")).rstrip("/")
        endpoint_url = f"{base_url}/chat/completions"
        parsed = urlsplit(endpoint_url)
        return _identity_from_provider_metadata({
            "execution_kind": "remote_hy3", "provider": "tencent-tokenhub",
            "model": str(getattr(client, "model")),
            "endpoint_origin": f"{parsed.scheme}://{parsed.netloc}",
            "endpoint_url": endpoint_url,
        })
    return PilotExecutionIdentity(
        execution_kind=PilotExecutionKind.OFFLINE_FIXTURE,
        provider="offline-fixture",
        model=str(getattr(client, "model", "offline-fixture-model")),
    )


def identity_from_pilot_model(model: object) -> PilotExecutionIdentity:
    """Obtain a frozen identity; generic test doubles are always non-formal."""

    declared = getattr(model, "execution_identity", None)
    if declared is not None:
        return PilotExecutionIdentity.model_validate(declared)
    return PilotExecutionIdentity(
        execution_kind=PilotExecutionKind.OFFLINE_FIXTURE,
        provider="offline-fixture",
        model=str(getattr(model, "model", "offline-fixture-model")),
    )
