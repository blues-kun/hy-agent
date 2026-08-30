"""Honest A/B/C/D Pilot generation with complete failure accounting.

The four arms are deliberately narrower than the proposal's future formal
system:

* A: Hy3 direct generation with no retrieval;
* B: Hy3 + deterministic sparse TF-IDF full-text passage retrieval;
* C: B + frozen text/metadata evidence-graph expansion and reranking;
* D: the exact C draft/evidence, filtered by an automatic Hy3 claim-evidence
  Judge gate.  There is no human revision and no additional retrieval.

Every question/replicate/arm cell is materialized as either a success artifact
or a failure artifact.  The runner never removes failed API or Schema calls
from the suite denominator.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, model_validator

from app.corpus import FrozenReviewCorpus
from app.experiment_retrieval import (
    FROZEN_GRAPH_METHOD,
    SPARSE_TFIDF_METHOD,
    FrozenEvidenceGraphRetriever,
    RetrievalResult,
    SparseTfidfIndex,
)
from app.schemas import (
    CorpusPassage,
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    SearchPlan,
)
from evaluator.judge import (
    Hy3Client,
    JudgeAggregate,
    JudgeSample,
    aggregate_samples,
    run_self_consistency,
)
from evaluator.judge.config import JudgeConfig
from evaluator.judge.hy3_client import JUDGE_OUTPUT_SCHEMA
from evaluator.judge.prompts import build_messages, system_prefix
from evaluator.artifact_security import (
    assert_json_safe,
    sanitize_failure_text as sanitize_shared_failure_text,
)
from evaluator.schemas import (
    Answerability,
    AtomicClaim,
    Citation,
    EvidenceSpan,
    SourceAccess,
    StrictModel,
    SupportVerdict,
    TextAnchor,
)


LEGACY_ABLATION_ARTIFACT_VERSION = "mitoevidence.pilot-ablation.v1"
ABLATION_ARTIFACT_VERSION_V2 = "mitoevidence.pilot-ablation.v2"
ABLATION_ARTIFACT_VERSION_V3 = "mitoevidence.pilot-ablation.v3"
ABLATION_ARTIFACT_VERSION = "mitoevidence.pilot-ablation.v4"
SUPPORTED_ABLATION_ARTIFACT_VERSIONS = (
    LEGACY_ABLATION_ARTIFACT_VERSION,
    ABLATION_ARTIFACT_VERSION_V2,
    ABLATION_ARTIFACT_VERSION_V3,
    ABLATION_ARTIFACT_VERSION,
)
GENERATOR_SEED_POLICY = (
    "sha256_v1(base_seed+nul+question_id+nul+replicate+nul+arm+nul+stage)_31bit"
)
GENERATOR_TEMPERATURE = 0.2
JUDGE_SEED_POLICY = (
    "sha256_v1(root_seed+nul+question_id+nul+replicate+nul+claim_id)_31bit;"
    "sample_seed=derived_base_seed+sample_index"
)
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
FAILURE_REDACTION_POLICY = "mitoevidence.failure-redaction.v1"
ABLATION_FORMAL_STATUS = "pilot_ablation_generation_unscored"
ABLATION_FORMAL_STATUS_BY_SCHEMA_VERSION = {
    version: ABLATION_FORMAL_STATUS
    for version in SUPPORTED_ABLATION_ARTIFACT_VERSIONS
}
SUITE_INPUT_SNAPSHOT_COPY = "pilot_input_snapshot.jsonl"
SUITE_EVIDENCE_MANIFEST_COPY = "evidence_manifest_snapshot.json"
MAX_FAILURE_TEXT_CHARS = 2000

_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_LABELED_SECRET = re.compile(
    r'''(?ix)
    (?P<prefix>["']?(?:api[_-]?key|x-api-key|authorization|access[_-]?token|
    refresh[_-]?token|client[_-]?secret|secret|password)["']?\s*[:=]\s*)
    (?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s,;}]+))
    '''
)
_URL_WITH_AUTH_OR_QUERY = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+"
)
_KNOWN_TOKEN_PREFIX = re.compile(
    r"(?i)\b(?:sk|ghp|github_pat|token|key)-[A-Za-z0-9_-]{12,}\b"
)
_LONG_OPAQUE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/=-]{24,}(?![A-Za-z0-9_+/=-])"
)


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    trailing = ""
    while raw and raw[-1] in ".,;)}":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    try:
        parsed = urlsplit(raw)
        if not parsed.scheme or not parsed.netloc:
            return "[REDACTED_URL]" + trailing
        # Drop userinfo completely.  Keeping the host/path is useful for
        # diagnostics; query and fragment values are never persisted.
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        query = "[REDACTED]" if parsed.query else ""
        fragment = "[REDACTED]" if parsed.fragment else ""
        return urlunsplit(
            (parsed.scheme, netloc, parsed.path, query, fragment)
        ) + trailing
    except ValueError:
        return "[REDACTED_URL]" + trailing


def sanitize_failure_text(value: object, *, max_chars: int = MAX_FAILURE_TEXT_CHARS) -> str:
    """Return an idempotently redacted, bounded diagnostic string.

    Failure text is data, but exceptions frequently include request headers,
    signed URLs, credentials or provider response bodies.  This sanitizer is
    applied before *both* failure.json and suite_state.json are written.
    """

    text = str(value)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    text = _LABELED_SECRET.sub(
        lambda match: (
            f'{match.group("prefix")}"[REDACTED]"'
            if match.group("double") is not None
            else (
                f"{match.group('prefix')}'[REDACTED]'"
                if match.group("single") is not None
                else f"{match.group('prefix')}[REDACTED]"
            )
        ),
        text,
    )
    text = _URL_WITH_AUTH_OR_QUERY.sub(_redact_url, text)
    text = _KNOWN_TOKEN_PREFIX.sub("[REDACTED]", text)
    text = _LONG_OPAQUE_TOKEN.sub("[REDACTED]", text)
    # Normalize through the repository-wide sanitizer last so the persisted
    # representation is exactly what assert_json_safe accepts.  This also
    # removes short reasoning_content values, which token-length heuristics do
    # not catch.
    text = sanitize_shared_failure_text(text)
    if len(text) > max_chars:
        text = text[:max_chars] + "…[TRUNCATED]"
    return text.strip() or "unspecified failure"


def failure_text_contains_sensitive_material(value: object) -> bool:
    """Conservative detector used by the artifact auditor and tests."""

    text = str(value)
    return (
        sanitize_shared_failure_text(text) != text
        or sanitize_failure_text(text, max_chars=max(len(text), 1)) != text.strip()
    )


class PilotArm(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class CellOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SuiteStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"


def derive_generator_seed(
    base_seed: int,
    question_id: str,
    replicate: int,
    arm: str,
    stage: str,
) -> int:
    material = (
        f"{base_seed}\0{question_id}\0{replicate}\0{arm}\0{stage}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def generator_cache_namespace(
    base_namespace: str,
    question_id: str,
    replicate: int,
    arm: str,
    stage: str,
) -> str:
    identity = hashlib.sha256(
        f"{question_id}\0{replicate}\0{arm}\0{stage}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{base_namespace}-{identity}"


def derive_judge_claim_seed(
    root_seed: int,
    question_id: str,
    replicate: int,
    claim_id: str,
) -> int:
    material = f"{root_seed}\0{question_id}\0{replicate}\0{claim_id}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


class GeneratorProvenance(StrictModel):
    execution_kind: Literal["remote_hy3", "test_fixture"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint_origin: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash_scope: Literal["source_file_bytes"] = "source_file_bytes"
    temperature: float = GENERATOR_TEMPERATURE
    base_seed: int
    seed_policy: Literal[
        "sha256_v1(base_seed+nul+question_id+nul+replicate+nul+arm+nul+stage)_31bit"
    ] = GENERATOR_SEED_POLICY
    cache_namespace: str = Field(
        pattern=r"^mitoevidence-[A-Za-z0-9_.-]{1,96}$"
    )
    max_parse_retries: int = Field(default=2, ge=0)
    fallback_channel: str = "json_schema"
    max_attempts: int = Field(default=4, ge=1)
    repair_policy: Literal["bounded_schema_repair_v1"] = (
        "bounded_schema_repair_v1"
    )

    @model_validator(mode="after")
    def _identity_is_safe(self) -> "GeneratorProvenance":
        endpoint = urlsplit(self.endpoint_origin)
        if (
            not endpoint.scheme
            or not endpoint.netloc
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in {"", "/"}
        ):
            raise ValueError("generator endpoint_origin 必须是无凭据/query/path 的 origin")
        endpoint_url = urlsplit(self.endpoint_url)
        if (
            not endpoint_url.scheme
            or not endpoint_url.netloc
            or endpoint_url.username is not None
            or endpoint_url.password is not None
            or endpoint_url.query
            or endpoint_url.fragment
            or not endpoint_url.path.endswith("/chat/completions")
            or f"{endpoint_url.scheme}://{endpoint_url.netloc}" != self.endpoint_origin
        ):
            raise ValueError(
                "generator endpoint_url 必须是无凭据/query 的完整 chat/completions URL"
            )
        if self.temperature != GENERATOR_TEMPERATURE:
            raise ValueError(f"generator temperature 必须冻结为 {GENERATOR_TEMPERATURE}")
        if self.execution_kind == "remote_hy3" and self.provider != "tencent-tokenhub":
            raise ValueError("remote_hy3 generator provider 必须是 tencent-tokenhub")
        if self.fallback_channel not in {"", "json_schema"}:
            raise ValueError("generator fallback_channel 只能为空或 json_schema")
        expected_attempts = (
            1
            + self.max_parse_retries
            + (1 if self.fallback_channel else 0)
        )
        if self.max_attempts != expected_attempts:
            raise ValueError(
                "generator max_attempts 必须等于 "
                "1 + max_parse_retries + fallback_channel_enabled"
            )
        return self


class ArmDefinition(StrictModel):
    arm: PilotArm
    name: str
    generation: str
    retrieval: str
    verification: str
    limitations: list[str]


ARM_DEFINITIONS = (
    ArmDefinition(
        arm=PilotArm.A,
        name="Hy3 direct / no retrieval",
        generation="Hy3 directly answers the neutral question without supplied evidence",
        retrieval="none",
        verification="none",
        limitations=[
            "Model-memory claims have no passage evidence and must not be treated as traceable science."
        ],
    ),
    ArmDefinition(
        arm=PilotArm.B,
        name="Hy3 + sparse TF-IDF full-text vector retrieval",
        generation="Hy3 synthesizes only from selected frozen passages",
        retrieval=SPARSE_TFIDF_METHOD,
        verification="local passage-id and Schema validation only",
        limitations=[
            "This is a sparse lexical TF-IDF vector baseline, not dense embedding/vector RAG.",
            "The frozen corpus currently contains passages from seven OA review articles, not a complete primary-study corpus.",
        ],
    ),
    ArmDefinition(
        arm=PilotArm.C,
        name="B + frozen evidence-graph reranking/expansion",
        generation="Hy3 synthesizes from graph-reranked frozen passages",
        retrieval=FROZEN_GRAPH_METHOD,
        verification="local passage-id and Schema validation only",
        limitations=[
            "The graph is built only from frozen text/metadata adjacency and lexical overlap; it is not an expert-curated scientific claim graph.",
            "Graph connectivity is a retrieval heuristic and never counts as scientific support by itself.",
        ],
    ),
    ArmDefinition(
        arm=PilotArm.D,
        name="C + automatic Hy3 claim-evidence Judge gate",
        generation="Deterministic rendering of claims retained from the exact C draft",
        retrieval="identical C evidence snapshot; no additional retrieval",
        verification="Hy3 claim-evidence Judge gate; supported, non-escalated claims only",
        limitations=[
            "D is automatic and unreviewed; Hy3 Judge is not expert truth.",
            "The gate filters the C draft and does not perform a fluent human or model rewrite.",
        ],
    ),
)
ARM_DEFINITION_BY_ID = {definition.arm: definition for definition in ARM_DEFINITIONS}


class RetrievalAudit(StrictModel):
    method: str
    construction_source: str
    queries: list[str]
    source_pmids: list[str]
    top_k: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    seed_passage_ids: list[str]
    selected_passage_ids: list[str]
    expanded_candidate_count: int = Field(ge=0)
    graph_node_count: int = Field(ge=0)
    graph_edge_count: int = Field(ge=0)
    graph_adjacency_edge_count: int = Field(ge=0)
    graph_lexical_edge_count: int = Field(ge=0)
    expert_labels_used: bool = False

    @model_validator(mode="after")
    def _no_label_leakage(self) -> "RetrievalAudit":
        if self.expert_labels_used:
            raise ValueError("A/B/C/D 检索不得使用专家标签")
        if len(self.selected_passage_ids) > self.top_k:
            raise ValueError("selected_passage_ids 不得超过 top_k")
        return self


class ClaimGateAudit(StrictModel):
    claim_id: str
    passed: bool
    rule: str = "supported_or_partially_supported_and_no_escalation"
    aggregate: JudgeAggregate

    @model_validator(mode="after")
    def _claim_matches(self) -> "ClaimGateAudit":
        if self.aggregate.claim_id != self.claim_id:
            raise ValueError("gate claim_id 与 aggregate.claim_id 不一致")
        expected_passed = (
            self.aggregate.final_verdict
            in {
                SupportVerdict.FULLY_SUPPORTED,
                SupportVerdict.PARTIALLY_SUPPORTED,
            }
            and not self.aggregate.escalate_to_human
        )
        if self.passed is not expected_passed:
            raise ValueError("gate.passed 与冻结门控规则不一致")
        return self


class JudgeSampleBinding(StrictModel):
    """Hash binding from one provenance record to one stored Judge sample."""

    index: int = Field(ge=0)
    ok: bool
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^(?:[0-9a-f]{64})?$")
    temperature: float | None = None
    seed: int | None = None

    @classmethod
    def from_sample(cls, sample: JudgeSample) -> "JudgeSampleBinding":
        return cls(
            index=sample.index,
            ok=sample.ok,
            sample_sha256=_sha_model(sample),
            response_sha256=sample.response_sha256,
            temperature=sample.temperature,
            seed=sample.seed,
        )


class JudgeClaimProvenance(StrictModel):
    """Base prompt plus exact sample bindings for one gated C claim."""

    claim_id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_hash_scope: Literal["base_messages_before_repair"] = (
        "base_messages_before_repair"
    )
    derived_base_seed: int | None = None
    samples: list[JudgeSampleBinding]

    @model_validator(mode="after")
    def _unique_sample_indices(self) -> "JudgeClaimProvenance":
        indices = [sample.index for sample in self.samples]
        if len(indices) != len(set(indices)):
            raise ValueError("Judge provenance sample index 不得重复")
        return self


class JudgeProvenanceIdentity(StrictModel):
    """Frozen Judge identity and sampling policy, independent of claim count."""

    execution_kind: Literal["remote_hy3", "test_fixture"]
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint_origin: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash_scope: Literal["source_file_bytes"]
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    structured_output_channel: Literal["function_calling", "json_schema"]
    k: int = Field(ge=1)
    temperature: float = Field(gt=0, le=2)
    base_seed: int | None = None
    sampling_seed_policy: Literal[
        "sha256_v1(root_seed+nul+question_id+nul+replicate+nul+claim_id)_31bit;sample_seed=derived_base_seed+sample_index"
    ] = JUDGE_SEED_POLICY
    min_agreement_votes: int = Field(ge=1)
    escalate_on_refuted: bool

    @model_validator(mode="after")
    def _identity_is_coherent(self) -> "JudgeProvenanceIdentity":
        if self.min_agreement_votes > self.k:
            raise ValueError("Judge min_agreement_votes 不得超过 k")
        endpoint = urlsplit(self.endpoint_url)
        if (
            not endpoint.scheme
            or not endpoint.netloc
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or not endpoint.path.endswith("/chat/completions")
        ):
            raise ValueError("Judge endpoint_url 必须是无凭据、无 query 的 chat/completions URL")
        expected_origin = f"{endpoint.scheme}://{endpoint.netloc}"
        if self.endpoint_origin != expected_origin:
            raise ValueError("Judge endpoint_origin 与 endpoint_url 不一致")
        if self.execution_kind == "remote_hy3" and self.provider != "tencent-tokenhub":
            raise ValueError("remote_hy3 provider 必须明确标记 tencent-tokenhub")
        return self


class JudgeProvenance(JudgeProvenanceIdentity):
    """Auditable D-arm Judge provenance, including every per-claim call."""

    execution_status: Literal[
        "remote_invoked",
        "test_fixture_invoked",
        "no_claims_no_request",
    ]
    calls: list[JudgeClaimProvenance]

    @model_validator(mode="after")
    def _unique_claim_calls(self) -> "JudgeProvenance":
        claim_ids = [call.claim_id for call in self.calls]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Judge provenance claim_id 不得重复")
        if not self.calls:
            if self.execution_status != "no_claims_no_request":
                raise ValueError("无 Judge calls 时 execution_status 必须是 no_claims_no_request")
        elif self.execution_kind == "remote_hy3":
            if self.execution_status != "remote_invoked":
                raise ValueError("remote_hy3 calls 必须标记 remote_invoked")
        elif self.execution_status != "test_fixture_invoked":
            raise ValueError("test_fixture calls 必须标记 test_fixture_invoked")
        return self


class AblationCellArtifact(StrictModel):
    schema_version: Literal[
        "mitoevidence.pilot-ablation.v1",
        "mitoevidence.pilot-ablation.v2",
        "mitoevidence.pilot-ablation.v3",
        "mitoevidence.pilot-ablation.v4",
    ] = ABLATION_ARTIFACT_VERSION
    arm: PilotArm
    arm_definition: ArmDefinition
    question_id: str
    replicate: int = Field(ge=1)
    request: ReviewRequest
    shared_plan: SearchPlan | None = None
    shared_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_manifest_path: str
    evidence_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passages: list[CorpusPassage]
    retrieval: RetrievalAudit
    review: GeneratedReview
    model_calls: list[ModelCallAudit]
    generator_provenance: GeneratorProvenance | None = None
    claim_gates: list[ClaimGateAudit] = Field(default_factory=list)
    judge_provenance: JudgeProvenance | None = None
    parent_c_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    warnings: list[str] = Field(default_factory=list)
    formal_status: Literal["pilot_ablation_generation_unscored"] = (
        ABLATION_FORMAL_STATUS
    )

    @model_validator(mode="after")
    def _arm_contract(self) -> "AblationCellArtifact":
        if self.arm_definition.arm is not self.arm:
            raise ValueError("arm_definition 与 arm 不一致")
        if self.question_id != self.request.question_id:
            raise ValueError("question_id 与 request.question_id 不一致")
        if self.arm is PilotArm.A:
            if self.passages or self.shared_plan is not None or self.claim_gates:
                raise ValueError("Arm A 不得包含检索、共享计划或 Judge gate")
            if any(claim.evidence_passage_ids for claim in self.review.claims):
                raise ValueError("Arm A 的 claim 不得声称 passage evidence")
        if self.arm in {PilotArm.B, PilotArm.C, PilotArm.D} and self.shared_plan is None:
            raise ValueError("B/C/D 必须绑定共享检索计划")
        if self.shared_plan is not None and self.shared_plan_sha256 != _sha_model(self.shared_plan):
            raise ValueError("shared_plan_sha256 与实际 shared_plan 不一致")
        expected_method = {
            PilotArm.A: "none",
            PilotArm.B: SPARSE_TFIDF_METHOD,
            PilotArm.C: FROZEN_GRAPH_METHOD,
            PilotArm.D: FROZEN_GRAPH_METHOD,
        }[self.arm]
        if self.retrieval.method != expected_method:
            raise ValueError(
                f"Arm {self.arm.value} retrieval.method 必须为 {expected_method}"
            )
        passage_ids = [passage.passage_id for passage in self.passages]
        if self.retrieval.selected_passage_ids != passage_ids:
            raise ValueError("retrieval.selected_passage_ids 必须与 passages 顺序完全一致")
        if self.schema_version in {
            ABLATION_ARTIFACT_VERSION_V3,
            ABLATION_ARTIFACT_VERSION,
        }:
            if self.generator_provenance is None:
                raise ValueError("v3/v4 cell 必须保存 generator_provenance")
            expected_stages = (
                ["ablation_A_direct"]
                if self.arm is PilotArm.A
                else ["plan", "synthesis"]
            )
            if [call.stage for call in self.model_calls] != expected_stages:
                raise ValueError("v3/v4 cell model_calls stage 组合不合规")
            generation_arm = PilotArm.C.value if self.arm is PilotArm.D else self.arm.value
            for call in self.model_calls:
                identity = self.generator_provenance
                if (
                    call.provider != identity.provider
                    or call.model != identity.model
                    or call.endpoint_origin != identity.endpoint_origin
                    or call.endpoint_url != identity.endpoint_url
                    or call.config_sha256 != identity.config_sha256
                    or call.temperature != identity.temperature
                ):
                    raise ValueError("v3/v4 cell model_call 与 generator_provenance identity 不一致")
                seed_replicate = 0 if call.stage == "plan" else self.replicate
                seed_arm = "shared" if call.stage == "plan" else generation_arm
                expected_seed = derive_generator_seed(
                    identity.base_seed,
                    self.question_id,
                    seed_replicate,
                    seed_arm,
                    call.stage,
                )
                expected_namespace = generator_cache_namespace(
                    identity.cache_namespace,
                    self.question_id,
                    seed_replicate,
                    seed_arm,
                    call.stage,
                )
                if call.requested_seed != expected_seed:
                    raise ValueError("v3/v4 cell model_call requested_seed 不符合冻结派生策略")
                if call.cache_namespace != expected_namespace:
                    raise ValueError("v3/v4 cell model_call cache_namespace 不符合冻结派生策略")
                hash_fields = (
                    call.prompt_sha256,
                    call.base_prompt_sha256,
                    call.schema_sha256,
                    call.response_sha256,
                    call.structured_output_sha256,
                )
                if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hash_fields):
                    raise ValueError("v3/v4 cell model_call 必须保存完整 prompt/schema/response/output hash")
                if (
                    self.schema_version == ABLATION_ARTIFACT_VERSION_V3
                    and call.attempt_count != 1
                ):
                    raise ValueError("v3 formal cell 当前只允许无需 repair 的单次成功调用")
                if self.schema_version == ABLATION_ARTIFACT_VERSION:
                    if not 1 <= call.attempt_count <= identity.max_attempts:
                        raise ValueError(
                            "v4 cell attempt_count 必须在 1..max_attempts 内"
                        )
                    if (
                        call.attempt_count == 1
                        and call.prompt_sha256 != call.base_prompt_sha256
                    ):
                        raise ValueError(
                            "v4 单次成功调用的 prompt 必须等于 base prompt"
                        )
        if self.arm is not PilotArm.D and self.claim_gates:
            raise ValueError("只有 Arm D 可以包含 claim_gates")
        if self.arm is PilotArm.D:
            if self.parent_c_artifact_sha256 is None:
                raise ValueError("Arm D 必须绑定精确 C artifact hash")
            gate_ids = [gate.claim_id for gate in self.claim_gates]
            if len(gate_ids) != len(set(gate_ids)):
                raise ValueError("Arm D claim_gates.claim_id 不得重复")
            passed_gate_ids = [
                gate.claim_id for gate in self.claim_gates if gate.passed
            ]
            review_claim_ids = [claim.claim_id for claim in self.review.claims]
            if review_claim_ids != passed_gate_ids:
                raise ValueError("Arm D review.claims 必须与 passed gates 同序一一对应")
            if self.review.answerability is not Answerability.OUT_OF_SCOPE:
                if self.review.claims:
                    if self.review.answerability is Answerability.INSUFFICIENT:
                        raise ValueError("Arm D 保留 claims 时 answerability 不得为 insufficient")
                    expected_answer = (
                        "经自动 Claim—Evidence Judge 门控保留的主张：\n"
                        + "\n".join(f"- {claim.text}" for claim in self.review.claims)
                    )
                    if self.review.answer != expected_answer:
                        raise ValueError("Arm D answer 不是 passed claims 的确定性渲染")
                elif (
                    self.review.answerability is not Answerability.INSUFFICIENT
                    or self.review.answer
                    != "C 草稿中的主张均未通过自动 Claim—Evidence Judge 门控，因此不保留科学结论。"
                ):
                    raise ValueError("Arm D 无 passed claims 时必须使用冻结的 insufficient 拒答")
            expected_limitation_suffix = [
                "D 是自动 Hy3 Judge 门控，不是专家复核。",
                f"门控保留 {len(self.review.claims)}/{len(self.claim_gates)} 条主张。",
            ]
            if self.review.limitations[-2:] != expected_limitation_suffix:
                raise ValueError("Arm D limitations 缺少冻结的门控后缀")
            judge_k = (
                self.judge_provenance.k
                if self.judge_provenance is not None
                else (
                    self.claim_gates[0].aggregate.k
                    if self.claim_gates
                    else None
                )
            )
            if judge_k is not None:
                expected_warning_suffix = [
                    "D 未追加检索，且输出是通过 gate 的 C 主张确定性渲染。",
                    (
                        "D Pilot Judge k=1，仅为单次自动门控，不是自一致性稳定性实验。"
                        if judge_k == 1
                        else f"D Judge 使用自一致性 k={judge_k}。"
                    ),
                ]
                if self.warnings[-2:] != expected_warning_suffix:
                    raise ValueError("Arm D warnings 缺少冻结的派生后缀")
            if (
                self.schema_version != LEGACY_ABLATION_ARTIFACT_VERSION
                and self.judge_provenance is None
            ):
                raise ValueError("v2/v3 Arm D 必须保存可审计 Judge provenance")
            if self.judge_provenance is None:
                return self
            if (
                self.schema_version
                in {ABLATION_ARTIFACT_VERSION_V3, ABLATION_ARTIFACT_VERSION}
                and self.judge_provenance.base_seed is None
            ):
                raise ValueError("v3/v4 Arm D Judge base_seed 不得为空")
            gate_ids = [gate.claim_id for gate in self.claim_gates]
            call_ids = [call.claim_id for call in self.judge_provenance.calls]
            if call_ids != gate_ids:
                raise ValueError("Judge provenance calls 必须与 claim_gates 同序一一对应")
            for gate, call in zip(
                self.claim_gates,
                self.judge_provenance.calls,
                strict=True,
            ):
                if gate.aggregate.k != self.judge_provenance.k:
                    raise ValueError("Judge aggregate.k 与 provenance.k 不一致")
                if len(gate.aggregate.samples) != self.judge_provenance.k:
                    raise ValueError("Judge aggregate 必须保存全部 k 个 samples")
                if [sample.index for sample in gate.aggregate.samples] != list(
                    range(self.judge_provenance.k)
                ):
                    raise ValueError("Judge sample indices 必须恰为 0..k-1")
                for sample in gate.aggregate.samples:
                    if sample.ok:
                        if (
                            sample.verdict is None
                            or sample.error
                            or not sample.parse_source
                            or not re.fullmatch(
                                r"[0-9a-f]{64}", sample.response_sha256
                            )
                        ):
                            raise ValueError("Judge 成功 sample 的 verdict/response/error 不自洽")
                        if sample.verdict.claim_id != gate.claim_id:
                            raise ValueError("Judge sample verdict.claim_id 与 gate 不一致")
                        if not set(sample.verdict.evidence_span_refs).issubset(
                            passage_ids
                        ):
                            raise ValueError("Judge sample 引用了未选中的 passage")
                    elif sample.verdict is not None or not sample.error:
                        raise ValueError("Judge 失败 sample 的 verdict/error 不自洽")
                expected_aggregate = aggregate_samples(
                    gate.claim_id,
                    gate.aggregate.samples,
                    k=self.judge_provenance.k,
                    min_agreement_votes=self.judge_provenance.min_agreement_votes,
                    escalate_on_refuted=self.judge_provenance.escalate_on_refuted,
                )
                if gate.aggregate != expected_aggregate:
                    raise ValueError("Judge aggregate 不是由已存 samples 按冻结规则重算得到")
                expected_samples = [
                    JudgeSampleBinding.from_sample(sample)
                    for sample in gate.aggregate.samples
                ]
                if call.samples != expected_samples:
                    raise ValueError("Judge provenance sample 绑定与 aggregate.samples 不一致")
                if any(
                    sample.temperature != self.judge_provenance.temperature
                    for sample in gate.aggregate.samples
                ):
                    raise ValueError("Judge sample temperature 与 provenance 不一致")
                effective_base_seed = (
                    call.derived_base_seed
                    if call.derived_base_seed is not None
                    else self.judge_provenance.base_seed
                )
                if self.schema_version in {
                    ABLATION_ARTIFACT_VERSION_V3,
                    ABLATION_ARTIFACT_VERSION,
                }:
                    if self.judge_provenance.base_seed is None:
                        raise ValueError("v3/v4 Judge root seed 不得为空")
                    expected_derived = derive_judge_claim_seed(
                        self.judge_provenance.base_seed,
                        self.question_id,
                        self.replicate,
                        gate.claim_id,
                    )
                    if call.derived_base_seed != expected_derived:
                        raise ValueError("v3 Judge derived_base_seed 不符合冻结派生策略")
                expected_seeds = (
                    [None] * self.judge_provenance.k
                    if effective_base_seed is None
                    else [
                        effective_base_seed + index
                        for index in range(self.judge_provenance.k)
                    ]
                )
                if [sample.seed for sample in gate.aggregate.samples] != expected_seeds:
                    raise ValueError("Judge sample seeds 与 provenance base_seed 不一致")
        elif self.parent_c_artifact_sha256 is not None:
            raise ValueError("只有 Arm D 可以填写 parent_c_artifact_sha256")
        elif self.judge_provenance is not None:
            raise ValueError("只有 Arm D 可以填写 judge_provenance")
        return self


class AblationCellRecord(StrictModel):
    question_id: str
    replicate: int = Field(ge=1)
    arm: PilotArm
    outcome: CellOutcome
    cell_dir: str
    cell_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    failure_type: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _outcome_fields(self) -> "AblationCellRecord":
        if self.outcome is CellOutcome.SUCCEEDED:
            if self.cell_manifest_sha256 is None:
                raise ValueError("成功 cell 必须提供 manifest hash")
            if self.failure_type is not None or self.failure_reason is not None:
                raise ValueError("成功 cell 不得填写 failure 字段")
        else:
            if self.cell_manifest_sha256 is not None:
                raise ValueError("失败 cell 不得伪造成功 manifest hash")
            if not self.failure_type or not self.failure_reason:
                raise ValueError("失败 cell 必须保留 failure_type/reason")
        return self


class InputSnapshot(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: list[str] = Field(min_length=1)
    fields_exposed_to_generator: list[str] = Field(
        default_factory=lambda: ["question_id", "question", "scope"]
    )

    @model_validator(mode="after")
    def _neutral_fields_only(self) -> "InputSnapshot":
        if self.fields_exposed_to_generator != ["question_id", "question", "scope"]:
            raise ValueError(
                "A/B/C/D generator 只能接收 question_id/question/scope；不得暴露专家字段"
            )
        return self


class AblationSafety(StrictModel):
    expert_labels_available_to_generator: Literal[False] = False
    d_additional_retrieval: Literal[False] = False
    failed_cells_removed: Literal[False] = False
    contains_api_key: Literal[False] = False
    contains_reasoning_content: Literal[False] = False


class PilotAblationSuiteState(StrictModel):
    schema_version: Literal[
        "mitoevidence.pilot-ablation.v1",
        "mitoevidence.pilot-ablation.v2",
        "mitoevidence.pilot-ablation.v3",
        "mitoevidence.pilot-ablation.v4",
    ] = ABLATION_ARTIFACT_VERSION
    suite_id: str
    status: SuiteStatus
    created_at_utc: str
    completed_at_utc: str | None = None
    input_snapshot: InputSnapshot
    evidence_manifest_path: str
    evidence_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_definitions: list[ArmDefinition]
    replicates: int = Field(ge=1)
    top_k: int = Field(ge=1)
    judge_k: int = Field(ge=1)
    generator_provenance: GeneratorProvenance | None = None
    judge_provenance_identity: JudgeProvenanceIdentity | None = None
    shared_plan_policy: Literal[
        "one_plan_per_question_shared_by_B_C_D_and_all_replicates"
    ] = "one_plan_per_question_shared_by_B_C_D_and_all_replicates"
    evidence_budget_policy: Literal[
        "B_and_C_same_top_k;_D_exact_C_snapshot"
    ] = "B_and_C_same_top_k;_D_exact_C_snapshot"
    expected_grid_cells: int = Field(ge=0)
    records: list[AblationCellRecord]
    planning_failures: dict[str, str] = Field(default_factory=dict)
    formal_status: Literal["pilot_ablation_generation_unscored"] = (
        ABLATION_FORMAL_STATUS
    )
    safety: AblationSafety = Field(default_factory=AblationSafety)

    @model_validator(mode="after")
    def _grid_is_auditable(self) -> "PilotAblationSuiteState":
        if (
            self.schema_version
            in {ABLATION_ARTIFACT_VERSION_V3, ABLATION_ARTIFACT_VERSION}
            and self.generator_provenance is None
        ):
            raise ValueError("v3/v4 suite 必须在顶层固定 generator_provenance")
        if self.schema_version in {
            ABLATION_ARTIFACT_VERSION_V3,
            ABLATION_ARTIFACT_VERSION,
        }:
            if self.judge_provenance_identity is None:
                raise ValueError("v3/v4 suite 必须在顶层固定 judge_provenance_identity")
            if self.judge_provenance_identity.base_seed is None:
                raise ValueError("v3/v4 formal suite 的 Judge base_seed 不得为空")
            if self.judge_provenance_identity.k != self.judge_k:
                raise ValueError("v3/v4 suite judge_k 与 Judge provenance 不一致")
        if Counter(definition.arm for definition in self.arm_definitions) != Counter(PilotArm):
            raise ValueError("arm_definitions 必须且只能包含 A/B/C/D 各一次")
        if self.arm_definitions != list(ARM_DEFINITIONS):
            raise ValueError("arm_definitions 必须与冻结的 A/B/C/D runtime 定义完全一致")
        if len(self.input_snapshot.question_ids) != len(set(self.input_snapshot.question_ids)):
            raise ValueError("input_snapshot.question_ids 必须唯一")
        expected = len(self.input_snapshot.question_ids) * self.replicates * len(PilotArm)
        if self.expected_grid_cells != expected:
            raise ValueError(
                f"expected_grid_cells 应为 {expected}，得到 {self.expected_grid_cells}"
            )
        keys = [(row.question_id, row.replicate, row.arm) for row in self.records]
        duplicates = sorted(
            (question, replicate, arm.value)
            for (question, replicate, arm), count in Counter(keys).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"suite cell key 重复：{duplicates}")
        allowed_questions = set(self.input_snapshot.question_ids)
        invalid = [
            (row.question_id, row.replicate, row.arm.value)
            for row in self.records
            if row.question_id not in allowed_questions or row.replicate > self.replicates
        ]
        if invalid:
            raise ValueError(f"suite record 超出预注册网格：{invalid}")
        if self.status is SuiteStatus.COMPLETED:
            if len(self.records) != self.expected_grid_cells:
                raise ValueError("completed suite 必须完整记录每个预期网格单元")
            expected_keys = {
                (question_id, replicate, arm)
                for question_id in self.input_snapshot.question_ids
                for replicate in range(1, self.replicates + 1)
                for arm in PilotArm
            }
            if set(keys) != expected_keys:
                raise ValueError("completed suite 的 question × replicate × arm 网格不完整")
            if self.completed_at_utc is None:
                raise ValueError("completed suite 必须填写 completed_at_utc")
        return self


def audit_pilot_ablation_grid(state: PilotAblationSuiteState) -> dict[str, object]:
    """Audit the runtime suite-state grid without opening cell artifacts.

    ``PilotAblationSuiteState`` already rejects duplicate/out-of-range keys and
    a dishonest ``expected_grid_cells`` declaration.  This report makes the
    complete Cartesian product and every explicit failed outcome visible.  It
    deliberately does not claim to verify D's ``parent_c_artifact_sha256``:
    that binding lives inside cell artifacts and requires artifact-level hash
    auditing, not projection from the suite journal.
    """

    records = {
        (record.question_id, record.arm, record.replicate): record
        for record in state.records
    }
    missing: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    by_arm: dict[str, dict[str, int]] = {}
    for question_id in state.input_snapshot.question_ids:
        for arm in PilotArm:
            for replicate in range(1, state.replicates + 1):
                record = records.get((question_id, arm, replicate))
                if record is None:
                    missing.append(
                        {
                            "question_id": question_id,
                            "arm": arm.value,
                            "replicate": replicate,
                        }
                    )
                else:
                    outcomes[record.outcome.value] += 1

    expected_per_arm = len(state.input_snapshot.question_ids) * state.replicates
    for arm in PilotArm:
        arm_records = [record for record in state.records if record.arm is arm]
        by_arm[arm.value] = {
            "expected": expected_per_arm,
            "recorded": len(arm_records),
            "succeeded": sum(
                record.outcome is CellOutcome.SUCCEEDED for record in arm_records
            ),
            "failed": sum(
                record.outcome is CellOutcome.FAILED for record in arm_records
            ),
        }

    computed_expected = (
        len(state.input_snapshot.question_ids) * state.replicates * len(PilotArm)
    )
    recorded = len(state.records)
    grid_complete = not missing
    suite_finalized = state.status is SuiteStatus.COMPLETED
    return {
        "schema_version": "mitoevidence.pilot-ablation-grid-audit.v1",
        "input_schema": state.schema_version,
        "suite_id": state.suite_id,
        "suite_status": state.status.value,
        "formal_status": ABLATION_FORMAL_STATUS_BY_SCHEMA_VERSION[
            state.schema_version
        ],
        "formal_status_source": "runtime_constant_by_input_schema",
        "expected_grid_cells": computed_expected,
        "declared_expected_grid_cells": state.expected_grid_cells,
        "recorded_grid_cells": recorded,
        "missing_grid_cells": len(missing),
        "grid_complete": grid_complete,
        "suite_finalized": suite_finalized,
        "runtime_complete": grid_complete and suite_finalized,
        "outcomes": dict(sorted(outcomes.items())),
        "by_arm": by_arm,
        "missing": missing,
        "checks": {
            "canonical_arm_definitions_match_runtime": (
                state.arm_definitions == list(ARM_DEFINITIONS)
            ),
            "fixed_shared_plan_and_evidence_budget_policies": (
                state.shared_plan_policy
                == "one_plan_per_question_shared_by_B_C_D_and_all_replicates"
                and state.evidence_budget_policy
                == "B_and_C_same_top_k;_D_exact_C_snapshot"
            ),
            "safety_flags_fail_closed": state.safety == AblationSafety(),
            "declared_expected_matches_cartesian_product": (
                state.expected_grid_cells == computed_expected
            ),
            "recorded_outcomes_sum_to_recorded_grid_cells": (
                sum(outcomes.values()) == recorded
            ),
            "explicit_failed_cells_retained_in_denominator": (
                outcomes.get(CellOutcome.FAILED.value, 0)
                == sum(
                    record.outcome is CellOutcome.FAILED
                    for record in state.records
                )
            ),
        },
        "artifact_level_checks": {
            "scope": "not_checked_by_suite_state_only_audit",
            "cell_manifest_and_file_hashes": {
                "checked": False,
                "reason": "需要 suite 目录中的 cell manifest 和实际文件",
            },
            "failure_json_matches_suite_record": {
                "checked": False,
                "reason": "需要打开每个失败 cell 的 failure.json",
            },
            "cell_dir_containment": {
                "checked": False,
                "reason": "纯 JSON state 审计不解析或访问 cell_dir",
            },
            "input_and_evidence_snapshot_file_hashes": {
                "checked": False,
                "reason": "suite_state 只记录固定值；需要仓库/输入文件进行重算",
            },
            "arm_semantics_and_shared_plan_evidence": {
                "checked": False,
                "reason": (
                    "A/B/C/D 的检索方法、共享 plan、top-k 和 D 精确复用 C 证据"
                    "均位于 cell artifacts"
                ),
            },
            "d_parent_c_artifact_hash_binding": {
                "checked": False,
                "reason": (
                    "suite_state 只固定各 cell manifest；D 的 "
                    "parent_c_artifact_sha256 必须另行打开并核验 C/D artifact"
                ),
            },
        },
        "interpretation": (
            "Runtime suite-state Cartesian-grid completeness audit. "
            "Explicit outcome=failed cells remain recorded and are not missing. "
            "Artifact semantics and file hashes are outside this state-only audit."
        ),
    }


class AblationReviewModel(Protocol):
    @property
    def audit_identity(self) -> dict[str, object]: ...

    def plan(
        self,
        request: ReviewRequest,
        *,
        seed: int | None = None,
        cache_namespace: str = "mitoevidence-ablation-v3",
    ) -> tuple[SearchPlan, ModelCallAudit]: ...

    def synthesize(
        self,
        request: ReviewRequest,
        passages: list[CorpusPassage],
        *,
        seed: int | None = None,
        cache_namespace: str = "mitoevidence-ablation-v3",
    ) -> tuple[GeneratedReview, ModelCallAudit]: ...

    def synthesize_direct(
        self,
        request: ReviewRequest,
        *,
        seed: int | None = None,
        cache_namespace: str = "mitoevidence-ablation-v3",
    ) -> tuple[GeneratedReview, ModelCallAudit]: ...


class ClaimGate(Protocol):
    @property
    def k(self) -> int: ...

    @property
    def provenance_identity(self) -> JudgeProvenanceIdentity: ...

    @property
    def last_call_provenance(self) -> JudgeClaimProvenance | None: ...

    def judge(
        self,
        claim: AtomicClaim,
        spans: list[EvidenceSpan],
        *,
        question: str,
        base_seed_override: int | None = None,
    ) -> JudgeAggregate: ...


class Hy3ClaimGate:
    """Production D-arm gate; tests inject a deterministic implementation."""

    def __init__(
        self,
        client: Hy3Client,
        *,
        config: JudgeConfig,
        k: int = 1,
        temperature: float | None = None,
        base_seed: int | None = None,
    ):
        if k <= 0:
            raise ValueError("Judge k 必须为正整数")
        self.client = client
        self.config = config
        if client.config.raw != config.raw or client.config.sha256 != config.sha256:
            raise ValueError("Judge client.config 必须与 gate config 完全一致")
        if not re.fullmatch(r"[0-9a-f]{64}", config.sha256):
            raise ValueError("Judge config 必须来自可哈希的 source file bytes")
        self._k = int(k)
        self.temperature = (
            float(config.self_consistency["temperature"])
            if temperature is None
            else float(temperature)
        )
        if not 0 < self.temperature <= 2:
            raise ValueError("Judge temperature 必须在 (0, 2] 内")
        configured_seed = config.self_consistency.get("base_seed")
        self.base_seed = (
            (None if configured_seed is None else int(configured_seed))
            if base_seed is None
            else int(base_seed)
        )
        parsed_endpoint = urlsplit(client.transport.base_url)
        if not parsed_endpoint.scheme or not parsed_endpoint.netloc:
            raise ValueError("Judge endpoint 必须包含 scheme 和 host")
        schema_sha256 = _sha_bytes(
            json.dumps(
                JUDGE_OUTPUT_SCHEMA,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        prompt_template_sha256 = _sha_bytes(
            system_prefix(client.channel).encode("utf-8")
        )
        configured_k = int(config.self_consistency["k"])
        configured_min_votes = int(
            config.self_consistency["min_agreement_votes"]
        )
        effective_min_votes = (
            configured_min_votes
            if self._k == configured_k
            else -(-configured_min_votes * self._k // configured_k)
        )
        endpoint_base = client.transport.base_url.rstrip("/")
        self._provenance_identity = JudgeProvenanceIdentity(
            execution_kind="remote_hy3",
            provider="tencent-tokenhub",
            model=client.model,
            endpoint_origin=(
                f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"
            ),
            endpoint_url=f"{endpoint_base}/chat/completions",
            config_sha256=config.sha256,
            config_hash_scope="source_file_bytes",
            schema_sha256=schema_sha256,
            prompt_template_sha256=prompt_template_sha256,
            structured_output_channel=client.channel,
            k=self._k,
            temperature=self.temperature,
            base_seed=self.base_seed,
            min_agreement_votes=effective_min_votes,
            escalate_on_refuted=bool(
                config.self_consistency.get("escalate_on_refuted", True)
            ),
        )
        self._last_call_provenance: JudgeClaimProvenance | None = None

    @property
    def k(self) -> int:
        return self._k

    @property
    def provenance_identity(self) -> JudgeProvenanceIdentity:
        return self._provenance_identity

    @property
    def last_call_provenance(self) -> JudgeClaimProvenance | None:
        return self._last_call_provenance

    def judge(
        self,
        claim: AtomicClaim,
        spans: list[EvidenceSpan],
        *,
        question: str,
        base_seed_override: int | None = None,
    ) -> JudgeAggregate:
        self._last_call_provenance = None
        base_messages = build_messages(
            claim,
            spans,
            question,
            channel=self.client.channel,
        )
        prompt_sha256 = _sha_bytes(
            json.dumps(
                base_messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        effective_base_seed = (
            self.base_seed
            if base_seed_override is None
            else int(base_seed_override)
        )
        aggregate = run_self_consistency(
            self.client,
            claim,
            spans,
            question=question,
            k=self._k,
            temperature=self.temperature,
            base_seed=effective_base_seed,
            config=self.config,
        )
        self._last_call_provenance = JudgeClaimProvenance(
            claim_id=claim.claim_id,
            prompt_sha256=prompt_sha256,
            derived_base_seed=effective_base_seed,
            samples=[
                JudgeSampleBinding.from_sample(sample)
                for sample in aggregate.samples
            ],
        )
        return aggregate


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_model(value: StrictModel) -> str:
    return _sha_bytes(_json_bytes(value.model_dump(mode="json")))


def _safe_id(value: str) -> str:
    cleaned = SAFE_ID.sub("-", value).strip("-.")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"非法路径标识：{value!r}")
    return cleaned


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _validate_neutral_input_snapshot(
    path: Path,
    requests: Sequence[ReviewRequest],
) -> None:
    rows: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取 Pilot 输入快照：{exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Pilot 输入第 {lineno} 行 JSON 无效：{exc}") from exc
        if not isinstance(row, dict) or not row.get("question_id"):
            raise ValueError(f"Pilot 输入第 {lineno} 行缺 question_id object")
        question_id = str(row["question_id"])
        if question_id in rows:
            raise ValueError(f"Pilot 输入 question_id 重复：{question_id}")
        rows[question_id] = row
    for request in requests:
        row = rows.get(request.question_id)
        if row is None:
            raise ValueError(f"Pilot 输入快照缺少 question_id={request.question_id}")
        if str(row.get("question") or "") != request.question:
            raise ValueError(f"{request.question_id} 的 request.question 与输入快照不一致")
        if str(row.get("scope") or "") != request.scope:
            raise ValueError(f"{request.question_id} 的 request.scope 与输入快照不一致")


def _retrieval_audit(
    result: RetrievalResult,
    *,
    queries: Sequence[str],
    source_pmids: Sequence[str],
    top_k: int,
) -> RetrievalAudit:
    return RetrievalAudit(
        method=result.method,
        construction_source=result.construction_source,
        queries=list(queries),
        source_pmids=list(source_pmids),
        top_k=top_k,
        candidate_count=result.candidate_count,
        seed_passage_ids=result.seed_passage_ids,
        selected_passage_ids=[passage.passage_id for passage in result.passages],
        expanded_candidate_count=result.expanded_candidate_count,
        graph_node_count=result.graph_node_count,
        graph_edge_count=result.graph_edge_count,
        graph_adjacency_edge_count=result.graph_adjacency_edge_count,
        graph_lexical_edge_count=result.graph_lexical_edge_count,
        expert_labels_used=False,
    )


def _evidence_span(passage: CorpusPassage) -> EvidenceSpan:
    return EvidenceSpan(
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


def _judge_unit(
    generated: GeneratedClaim,
    passages: dict[str, CorpusPassage],
) -> tuple[AtomicClaim, list[EvidenceSpan]]:
    by_pmid: dict[str, list[str]] = {}
    spans: list[EvidenceSpan] = []
    for passage_id in generated.evidence_passage_ids:
        passage = passages[passage_id]
        by_pmid.setdefault(passage.pmid, []).append(passage_id)
        spans.append(_evidence_span(passage))
    claim = AtomicClaim(
        claim_id=generated.claim_id,
        text=generated.text,
        is_core=generated.is_core,
        conditions=generated.conditions,
        citations=[
            Citation(
                doi_or_pmid=f"PMID:{pmid}",
                paper_id=f"PMID:{pmid}",
                evidence_span_ids=span_ids,
            )
            for pmid, span_ids in sorted(by_pmid.items())
        ],
    )
    return claim, spans


def derive_d_review_and_warnings(
    c_artifact: AblationCellArtifact,
    gates: Sequence[ClaimGateAudit],
    *,
    judge_k: int,
) -> tuple[GeneratedReview, list[str]]:
    """Deterministically derive D from the exact C draft and gate decisions."""

    c_claim_ids = [claim.claim_id for claim in c_artifact.review.claims]
    gate_ids = [gate.claim_id for gate in gates]
    if gate_ids != c_claim_ids:
        raise ValueError("D gates 必须与 C claims 同序一一对应")
    accepted = [
        claim
        for claim, gate in zip(c_artifact.review.claims, gates, strict=True)
        if gate.passed
    ]
    if c_artifact.review.answerability is Answerability.OUT_OF_SCOPE:
        accepted = []
        answerability = Answerability.OUT_OF_SCOPE
        answer = c_artifact.review.answer
    elif accepted:
        answerability = (
            Answerability.PARTIAL
            if c_artifact.review.answerability is Answerability.INSUFFICIENT
            else c_artifact.review.answerability
        )
        answer = "经自动 Claim—Evidence Judge 门控保留的主张：\n" + "\n".join(
            f"- {claim.text}" for claim in accepted
        )
    else:
        answerability = Answerability.INSUFFICIENT
        answer = "C 草稿中的主张均未通过自动 Claim—Evidence Judge 门控，因此不保留科学结论。"
    review = GeneratedReview(
        answerability=answerability,
        answer=answer,
        claims=accepted,
        limitations=[
            *c_artifact.review.limitations,
            "D 是自动 Hy3 Judge 门控，不是专家复核。",
            f"门控保留 {len(accepted)}/{len(c_artifact.review.claims)} 条主张。",
        ],
    )
    warnings = [
        *c_artifact.warnings,
        "D 未追加检索，且输出是通过 gate 的 C 主张确定性渲染。",
        (
            "D Pilot Judge k=1，仅为单次自动门控，不是自一致性稳定性实验。"
            if judge_k == 1
            else f"D Judge 使用自一致性 k={judge_k}。"
        ),
    ]
    return review, warnings


class PilotAblationRunner:
    def __init__(
        self,
        *,
        model: AblationReviewModel,
        corpus: FrozenReviewCorpus,
        claim_gate: ClaimGate,
        top_k: int = 12,
        generator_base_seed: int = 20260831,
        generator_cache_namespace: str = "mitoevidence-ablation-v4",
    ):
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.model = model
        self.corpus = corpus
        self.claim_gate = claim_gate
        self.top_k = top_k
        identity = getattr(model, "audit_identity", None)
        if not isinstance(identity, dict):
            raise ValueError("ablation model 必须暴露 audit_identity")
        self.generator_provenance = GeneratorProvenance.model_validate(
            {
                **identity,
                "temperature": GENERATOR_TEMPERATURE,
                "base_seed": generator_base_seed,
                "cache_namespace": generator_cache_namespace,
            }
        )
        if claim_gate.provenance_identity.base_seed is None:
            raise ValueError("v4 formal ablation 要求 Judge base_seed 非空")
        passages = corpus.load()
        self.tfidf = SparseTfidfIndex(passages)
        self.graph = FrozenEvidenceGraphRetriever(self.tfidf)

    @property
    def _manifest_label(self) -> str:
        try:
            return str(self.corpus.manifest_path.relative_to(self.corpus.repo_root))
        except ValueError:
            return self.corpus.manifest_path.name

    @staticmethod
    def _validate_grounded_review(
        review: GeneratedReview,
        passages: Sequence[CorpusPassage],
    ) -> list[str]:
        known = {passage.passage_id for passage in passages}
        unknown = sorted(
            {
                passage_id
                for claim in review.claims
                for passage_id in claim.evidence_passage_ids
                if passage_id not in known
            }
        )
        if unknown:
            raise ValueError(f"模型引用未提供的 passage_id：{unknown}")
        if not passages and review.answerability in {
            Answerability.ANSWERABLE,
            Answerability.PARTIAL,
        }:
            raise ValueError("没有召回段落但模型仍声称 answerable/partial")
        ungrounded = [claim.claim_id for claim in review.claims if not claim.evidence_passage_ids]
        return (
            ["存在未绑定检索证据的主张，D gate 应判 unknown：" + ", ".join(ungrounded)]
            if ungrounded
            else []
        )

    def run_a(self, request: ReviewRequest, replicate: int) -> AblationCellArtifact:
        seed = derive_generator_seed(
            self.generator_provenance.base_seed,
            request.question_id,
            replicate,
            PilotArm.A.value,
            "ablation_A_direct",
        )
        namespace = generator_cache_namespace(
            self.generator_provenance.cache_namespace,
            request.question_id,
            replicate,
            PilotArm.A.value,
            "ablation_A_direct",
        )
        review, audit = self.model.synthesize_direct(
            request,
            seed=seed,
            cache_namespace=namespace,
        )
        return AblationCellArtifact(
            arm=PilotArm.A,
            arm_definition=ARM_DEFINITION_BY_ID[PilotArm.A],
            question_id=request.question_id,
            replicate=replicate,
            request=request,
            evidence_manifest_path=self._manifest_label,
            evidence_manifest_sha256=self.corpus.manifest_sha256,
            passages=[],
            retrieval=RetrievalAudit(
                method="none",
                construction_source="none",
                queries=[],
                source_pmids=[],
                top_k=0,
                candidate_count=0,
                seed_passage_ids=[],
                selected_passage_ids=[],
                expanded_candidate_count=0,
                graph_node_count=0,
                graph_edge_count=0,
                graph_adjacency_edge_count=0,
                graph_lexical_edge_count=0,
            ),
            review=review,
            model_calls=[audit],
            generator_provenance=self.generator_provenance,
            warnings=["Arm A 无检索；所有科学主张均无外部证据绑定。"],
        )

    def _run_grounded(
        self,
        arm: PilotArm,
        request: ReviewRequest,
        replicate: int,
        plan: SearchPlan,
        plan_audit: ModelCallAudit,
    ) -> AblationCellArtifact:
        if arm not in {PilotArm.B, PilotArm.C}:
            raise ValueError("_run_grounded 只支持 B/C")
        # PMID constraints are fixed caller inputs.  The planner is not shown
        # the corpus ID inventory, so any model-emitted PMID must not silently
        # narrow retrieval or create a stochastic/invented filter.
        requested_pmids = request.source_pmids
        result = (
            self.tfidf.search(
                plan.queries, source_pmids=requested_pmids, top_k=self.top_k
            )
            if arm is PilotArm.B
            else self.graph.search(
                plan.queries, source_pmids=requested_pmids, top_k=self.top_k
            )
        )
        synthesis_request = request.model_copy(
            update={"answerability_hint": plan.answerability_hint}
        )
        seed = derive_generator_seed(
            self.generator_provenance.base_seed,
            request.question_id,
            replicate,
            arm.value,
            "synthesis",
        )
        namespace = generator_cache_namespace(
            self.generator_provenance.cache_namespace,
            request.question_id,
            replicate,
            arm.value,
            "synthesis",
        )
        review, synthesis_audit = self.model.synthesize(
            synthesis_request,
            result.passages,
            seed=seed,
            cache_namespace=namespace,
        )
        warnings = self._validate_grounded_review(review, result.passages)
        if arm is PilotArm.B:
            warnings.append("B 是稀疏 TF-IDF 向量检索，不是 dense embedding RAG。")
        else:
            warnings.append("C 图仅由冻结文本/元数据构造，未读取专家金标。")
        return AblationCellArtifact(
            arm=arm,
            arm_definition=ARM_DEFINITION_BY_ID[arm],
            question_id=request.question_id,
            replicate=replicate,
            request=request,
            shared_plan=plan,
            shared_plan_sha256=_sha_model(plan),
            evidence_manifest_path=self._manifest_label,
            evidence_manifest_sha256=self.corpus.manifest_sha256,
            passages=result.passages,
            retrieval=_retrieval_audit(
                result,
                queries=plan.queries,
                source_pmids=requested_pmids,
                top_k=self.top_k,
            ),
            review=review,
            model_calls=[plan_audit, synthesis_audit],
            generator_provenance=self.generator_provenance,
            warnings=warnings,
        )

    def run_b(
        self,
        request: ReviewRequest,
        replicate: int,
        plan: SearchPlan,
        plan_audit: ModelCallAudit,
    ) -> AblationCellArtifact:
        return self._run_grounded(PilotArm.B, request, replicate, plan, plan_audit)

    def run_c(
        self,
        request: ReviewRequest,
        replicate: int,
        plan: SearchPlan,
        plan_audit: ModelCallAudit,
    ) -> AblationCellArtifact:
        return self._run_grounded(PilotArm.C, request, replicate, plan, plan_audit)

    def run_d(self, c_artifact: AblationCellArtifact) -> AblationCellArtifact:
        if c_artifact.arm is not PilotArm.C:
            raise ValueError("D 必须从 C artifact 开始")
        passage_by_id = {passage.passage_id: passage for passage in c_artifact.passages}
        gates: list[ClaimGateAudit] = []
        judge_calls: list[JudgeClaimProvenance] = []
        for generated in c_artifact.review.claims:
            claim, spans = _judge_unit(generated, passage_by_id)
            judge_seed = derive_judge_claim_seed(
                self.claim_gate.provenance_identity.base_seed,
                c_artifact.question_id,
                c_artifact.replicate,
                generated.claim_id,
            )
            aggregate = self.claim_gate.judge(
                claim,
                spans,
                question=c_artifact.request.question,
                base_seed_override=judge_seed,
            )
            if aggregate.claim_id != generated.claim_id:
                raise ValueError("Judge aggregate.claim_id 与输入 claim 不一致")
            if aggregate.n_valid == 0:
                raise RuntimeError(f"D gate 对 {generated.claim_id} 没有任何有效 Judge 判定")
            call_provenance = self.claim_gate.last_call_provenance
            if (
                call_provenance is None
                or call_provenance.claim_id != generated.claim_id
            ):
                raise RuntimeError(
                    f"D gate 未为 {generated.claim_id} 返回匹配的 Judge provenance"
                )
            passed = (
                aggregate.final_verdict
                in {SupportVerdict.FULLY_SUPPORTED, SupportVerdict.PARTIALLY_SUPPORTED}
                and not aggregate.escalate_to_human
            )
            gates.append(
                ClaimGateAudit(
                    claim_id=generated.claim_id,
                    passed=passed,
                    aggregate=aggregate,
                )
            )
            judge_calls.append(call_provenance)
        review, warnings = derive_d_review_and_warnings(
            c_artifact,
            gates,
            judge_k=self.claim_gate.k,
        )
        return AblationCellArtifact(
            arm=PilotArm.D,
            arm_definition=ARM_DEFINITION_BY_ID[PilotArm.D],
            question_id=c_artifact.question_id,
            replicate=c_artifact.replicate,
            request=c_artifact.request,
            shared_plan=c_artifact.shared_plan,
            shared_plan_sha256=c_artifact.shared_plan_sha256,
            evidence_manifest_path=c_artifact.evidence_manifest_path,
            evidence_manifest_sha256=c_artifact.evidence_manifest_sha256,
            passages=c_artifact.passages,
            retrieval=c_artifact.retrieval,
            review=review,
            model_calls=c_artifact.model_calls,
            generator_provenance=c_artifact.generator_provenance,
            claim_gates=gates,
            judge_provenance=JudgeProvenance(
                **self.claim_gate.provenance_identity.model_dump(mode="python"),
                execution_status=(
                    "no_claims_no_request"
                    if not judge_calls
                    else (
                        "remote_invoked"
                        if self.claim_gate.provenance_identity.execution_kind
                        == "remote_hy3"
                        else "test_fixture_invoked"
                    )
                ),
                calls=judge_calls,
            ),
            parent_c_artifact_sha256=_sha_model(c_artifact),
            warnings=warnings,
        )

    @staticmethod
    def _write_cell(
        suite_dir: Path,
        artifact: AblationCellArtifact,
    ) -> AblationCellRecord:
        artifact_payload = artifact.model_dump(mode="json")
        review_payload = artifact.review.model_dump(mode="json")
        passage_payloads = [
            passage.model_dump(mode="json") for passage in artifact.passages
        ]
        gate_payloads = [
            gate.model_dump(mode="json") for gate in artifact.claim_gates
        ]
        # Successful model/evidence text is untrusted too.  Refuse the entire
        # cell before creating its directory if any serialized payload would
        # be changed by the shared credential/private-reasoning sanitizer.
        # Scientific text is never silently rewritten.
        for payload in (
            artifact_payload,
            review_payload,
            passage_payloads,
            gate_payloads,
        ):
            assert_json_safe(payload)
        relative = Path(_safe_id(artifact.question_id)) / f"replicate-{artifact.replicate:02d}" / artifact.arm.value
        final_dir = suite_dir / relative
        if final_dir.exists():
            raise FileExistsError(f"cell 目录已存在：{relative}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact.arm.value}-", dir=final_dir.parent))
        try:
            files = {
                "artifact.json": _json_bytes(artifact_payload),
                "review.json": _json_bytes(review_payload),
                "retrieval.jsonl": b"".join(
                    json.dumps(
                        passage,
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                    for passage in passage_payloads
                ),
                "claim_gates.jsonl": b"".join(
                    json.dumps(
                        gate,
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                    for gate in gate_payloads
                ),
            }
            for name, data in files.items():
                (temporary / name).write_bytes(data)
            manifest = {
                "schema_version": artifact.schema_version,
                "question_id": artifact.question_id,
                "replicate": artifact.replicate,
                "arm": artifact.arm.value,
                "formal_status": artifact.formal_status,
                "files": {
                    name: {"bytes": len(data), "sha256": _sha_bytes(data)}
                    for name, data in files.items()
                },
                "security": {
                    "contains_api_key": False,
                    "contains_reasoning_content": False,
                },
            }
            assert_json_safe(manifest)
            manifest_data = _json_bytes(manifest)
            (temporary / "manifest.json").write_bytes(manifest_data)
            os.replace(temporary, final_dir)
        except BaseException:
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
            raise
        return AblationCellRecord(
            question_id=artifact.question_id,
            replicate=artifact.replicate,
            arm=artifact.arm,
            outcome=CellOutcome.SUCCEEDED,
            cell_dir=str(relative),
            cell_manifest_sha256=_sha_bytes(manifest_data),
        )

    @staticmethod
    def _write_failure(
        suite_dir: Path,
        *,
        question_id: str,
        replicate: int,
        arm: PilotArm,
        exc: BaseException,
        schema_version: str = ABLATION_ARTIFACT_VERSION,
    ) -> AblationCellRecord:
        relative = Path(_safe_id(question_id)) / f"replicate-{replicate:02d}" / arm.value
        final_dir = suite_dir / relative
        if final_dir.exists():
            raise FileExistsError(f"failure cell 目录已存在：{relative}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir()
        failure_type = sanitize_failure_text(type(exc).__name__, max_chars=120)
        failure_reason = sanitize_failure_text(str(exc) or repr(exc))
        failure_payload = {
                    "schema_version": schema_version,
                    "question_id": question_id,
                    "replicate": replicate,
                    "arm": arm.value,
                    "outcome": "failed",
                    "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "security": {
                        "contains_api_key": False,
                        "contains_reasoning_content": False,
                        "failure_text_sanitized": True,
                        "redaction_policy": FAILURE_REDACTION_POLICY,
                    },
                }
        assert_json_safe(failure_payload)
        _write_atomic(final_dir / "failure.json", _json_bytes(failure_payload))
        return AblationCellRecord(
            question_id=question_id,
            replicate=replicate,
            arm=arm,
            outcome=CellOutcome.FAILED,
            cell_dir=str(relative),
            failure_type=failure_type,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _write_state(suite_dir: Path, state: PilotAblationSuiteState) -> None:
        payload = state.model_dump(mode="json")
        assert_json_safe(payload)
        _write_atomic(
            suite_dir / "suite_state.json",
            _json_bytes(payload),
        )

    def run_suite(
        self,
        requests: Sequence[ReviewRequest],
        *,
        replicates: int,
        out_root: str | Path,
        suite_id: str,
        input_path: str | Path,
        resume: bool = False,
    ) -> tuple[Path, PilotAblationSuiteState]:
        if resume:
            raise RuntimeError(
                "严格 --resume 尚未实现：为避免把不同模型、配置、输入或 Judge "
                "策略混入同一 suite，本 runner 拒绝恢复；请使用新的 suite_id 重跑"
            )
        if replicates <= 0:
            raise ValueError("replicates 必须为正整数")
        question_ids = [request.question_id for request in requests]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("suite question_id 必须唯一")
        non_neutral = [
            request.question_id
            for request in requests
            if request.source_pmids
            or request.answerability_hint is not None
            or request.prohibited_inferences
        ]
        if non_neutral:
            raise ValueError(
                "Pilot ablation 只接受 question_id/question/scope 中性输入；"
                f"以下请求含额外提示字段：{non_neutral}"
            )
        root = Path(out_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        safe_suite = _safe_id(suite_id)
        suite_dir = root / safe_suite
        if suite_dir.exists():
            raise FileExistsError(f"suite 目录已存在：{suite_dir}")
        input_file = Path(input_path).resolve()
        _validate_neutral_input_snapshot(input_file, requests)
        suite_dir.mkdir()
        # Keep byte-identical source snapshots beside the journal.  The
        # declared paths remain unchanged (and are still written into every
        # cell), while downstream formal scoring can recompute both hashes
        # even if the original invocation used a path outside repo_root.
        _write_atomic(
            suite_dir / SUITE_INPUT_SNAPSHOT_COPY,
            input_file.read_bytes(),
        )
        _write_atomic(
            suite_dir / SUITE_EVIDENCE_MANIFEST_COPY,
            self.corpus.manifest_path.read_bytes(),
        )
        try:
            input_label = str(input_file.relative_to(self.corpus.repo_root))
        except ValueError:
            input_label = input_file.name
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = PilotAblationSuiteState(
            suite_id=safe_suite,
            status=SuiteStatus.RUNNING,
            created_at_utc=created,
            input_snapshot=InputSnapshot(
                path=input_label,
                sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                question_ids=question_ids,
            ),
            evidence_manifest_path=self._manifest_label,
            evidence_manifest_sha256=self.corpus.manifest_sha256,
            arm_definitions=list(ARM_DEFINITIONS),
            replicates=replicates,
            top_k=self.top_k,
            judge_k=self.claim_gate.k,
            generator_provenance=self.generator_provenance,
            judge_provenance_identity=self.claim_gate.provenance_identity,
            expected_grid_cells=len(requests) * replicates * len(PilotArm),
            records=[],
        )
        self._write_state(suite_dir, state)

        records: list[AblationCellRecord] = []
        planning_failures: dict[str, str] = {}
        for request in requests:
            plan: SearchPlan | None = None
            plan_audit: ModelCallAudit | None = None
            try:
                plan_seed = derive_generator_seed(
                    self.generator_provenance.base_seed,
                    request.question_id,
                    0,
                    "shared",
                    "plan",
                )
                plan_namespace = generator_cache_namespace(
                    self.generator_provenance.cache_namespace,
                    request.question_id,
                    0,
                    "shared",
                    "plan",
                )
                plan, plan_audit = self.model.plan(
                    request,
                    seed=plan_seed,
                    cache_namespace=plan_namespace,
                )
                _write_atomic(
                    suite_dir / _safe_id(request.question_id) / "shared_plan.json",
                    _json_bytes(
                        {
                            "plan": plan.model_dump(mode="json"),
                            "audit": plan_audit.model_dump(mode="json"),
                            "plan_sha256": _sha_model(plan),
                            "expert_labels_used": False,
                        }
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - retained in every dependent cell
                planning_failures[request.question_id] = sanitize_failure_text(
                    f"{type(exc).__name__}: {exc}"
                )

            for replicate in range(1, replicates + 1):
                try:
                    a_artifact = self.run_a(request, replicate)
                    record = self._write_cell(suite_dir, a_artifact)
                except Exception as exc:  # noqa: BLE001 - cell failure is data
                    record = self._write_failure(
                        suite_dir,
                        question_id=request.question_id,
                        replicate=replicate,
                        arm=PilotArm.A,
                        exc=exc,
                    )
                records.append(record)
                state = state.model_copy(
                    update={"records": list(records), "planning_failures": dict(planning_failures)}
                )
                self._write_state(suite_dir, state)

                c_artifact: AblationCellArtifact | None = None
                for arm in (PilotArm.B, PilotArm.C):
                    if plan is None or plan_audit is None:
                        exc = RuntimeError(
                            "共享检索计划失败：" + planning_failures[request.question_id]
                        )
                        record = self._write_failure(
                            suite_dir,
                            question_id=request.question_id,
                            replicate=replicate,
                            arm=arm,
                            exc=exc,
                        )
                    else:
                        try:
                            artifact = (
                                self.run_b(request, replicate, plan, plan_audit)
                                if arm is PilotArm.B
                                else self.run_c(request, replicate, plan, plan_audit)
                            )
                            if arm is PilotArm.C:
                                c_artifact = artifact
                            record = self._write_cell(suite_dir, artifact)
                        except Exception as exc:  # noqa: BLE001
                            record = self._write_failure(
                                suite_dir,
                                question_id=request.question_id,
                                replicate=replicate,
                                arm=arm,
                                exc=exc,
                            )
                    records.append(record)
                    state = state.model_copy(
                        update={"records": list(records), "planning_failures": dict(planning_failures)}
                    )
                    self._write_state(suite_dir, state)

                if c_artifact is None:
                    exc = RuntimeError("C cell 未成功，D 无法从同一 C 草稿执行 Judge gate")
                    record = self._write_failure(
                        suite_dir,
                        question_id=request.question_id,
                        replicate=replicate,
                        arm=PilotArm.D,
                        exc=exc,
                    )
                else:
                    try:
                        d_artifact = self.run_d(c_artifact)
                        record = self._write_cell(suite_dir, d_artifact)
                    except Exception as exc:  # noqa: BLE001
                        record = self._write_failure(
                            suite_dir,
                            question_id=request.question_id,
                            replicate=replicate,
                            arm=PilotArm.D,
                            exc=exc,
                        )
                records.append(record)
                state = state.model_copy(
                    update={"records": list(records), "planning_failures": dict(planning_failures)}
                )
                self._write_state(suite_dir, state)

        completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state = state.model_copy(
            update={
                "status": SuiteStatus.COMPLETED,
                "completed_at_utc": completed,
                "records": records,
                "planning_failures": planning_failures,
            }
        )
        # model_copy(update=...) intentionally avoids repeated validation while
        # journaling, so enforce the completed-grid invariant once at closure.
        state = PilotAblationSuiteState.model_validate(state.model_dump(mode="json"))
        final_state_payload = state.model_dump(mode="json")
        assert_json_safe(final_state_payload)
        final_state_bytes = _json_bytes(final_state_payload)
        # Both journals are authenticated by their byte identity.  Serialize
        # once so a future serializer change cannot create a semantic-equal
        # but byte-different summary.
        _write_atomic(suite_dir / "suite_state.json", final_state_bytes)
        _write_atomic(suite_dir / "suite_summary.json", final_state_bytes)
        return suite_dir, state
