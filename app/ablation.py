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
from evaluator.judge import Hy3Client, JudgeAggregate, run_self_consistency
from evaluator.judge.config import JudgeConfig
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


ABLATION_ARTIFACT_VERSION = "mitoevidence.pilot-ablation.v1"
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


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
        return self


class AblationCellArtifact(StrictModel):
    schema_version: Literal["mitoevidence.pilot-ablation.v1"] = ABLATION_ARTIFACT_VERSION
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
    claim_gates: list[ClaimGateAudit] = Field(default_factory=list)
    parent_c_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    warnings: list[str] = Field(default_factory=list)
    formal_status: str = "pilot_ablation_generation_unscored"

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
        if self.arm is not PilotArm.D and self.claim_gates:
            raise ValueError("只有 Arm D 可以包含 claim_gates")
        if self.arm is PilotArm.D:
            if self.parent_c_artifact_sha256 is None:
                raise ValueError("Arm D 必须绑定精确 C artifact hash")
        elif self.parent_c_artifact_sha256 is not None:
            raise ValueError("只有 Arm D 可以填写 parent_c_artifact_sha256")
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
    schema_version: Literal["mitoevidence.pilot-ablation.v1"] = ABLATION_ARTIFACT_VERSION
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
    expected_grid_cells: int = Field(ge=0)
    records: list[AblationCellRecord]
    planning_failures: dict[str, str] = Field(default_factory=dict)
    formal_status: str = "pilot_ablation_generation_unscored"
    safety: AblationSafety = Field(default_factory=AblationSafety)

    @model_validator(mode="after")
    def _grid_is_auditable(self) -> "PilotAblationSuiteState":
        if Counter(definition.arm for definition in self.arm_definitions) != Counter(PilotArm):
            raise ValueError("arm_definitions 必须且只能包含 A/B/C/D 各一次")
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


class AblationReviewModel(Protocol):
    def plan(self, request: ReviewRequest) -> tuple[SearchPlan, ModelCallAudit]: ...

    def synthesize(
        self, request: ReviewRequest, passages: list[CorpusPassage]
    ) -> tuple[GeneratedReview, ModelCallAudit]: ...

    def synthesize_direct(
        self, request: ReviewRequest
    ) -> tuple[GeneratedReview, ModelCallAudit]: ...


class ClaimGate(Protocol):
    @property
    def k(self) -> int: ...

    def judge(
        self, claim: AtomicClaim, spans: list[EvidenceSpan], *, question: str
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
        self._k = int(k)
        self.temperature = temperature
        self.base_seed = base_seed

    @property
    def k(self) -> int:
        return self._k

    def judge(
        self, claim: AtomicClaim, spans: list[EvidenceSpan], *, question: str
    ) -> JudgeAggregate:
        return run_self_consistency(
            self.client,
            claim,
            spans,
            question=question,
            k=self._k,
            temperature=self.temperature,
            base_seed=self.base_seed,
            config=self.config,
        )


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


class PilotAblationRunner:
    def __init__(
        self,
        *,
        model: AblationReviewModel,
        corpus: FrozenReviewCorpus,
        claim_gate: ClaimGate,
        top_k: int = 12,
    ):
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.model = model
        self.corpus = corpus
        self.claim_gate = claim_gate
        self.top_k = top_k
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
        review, audit = self.model.synthesize_direct(request)
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
        review, synthesis_audit = self.model.synthesize(
            synthesis_request, result.passages
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
        accepted: list[GeneratedClaim] = []
        for generated in c_artifact.review.claims:
            claim, spans = _judge_unit(generated, passage_by_id)
            aggregate = self.claim_gate.judge(
                claim, spans, question=c_artifact.request.question
            )
            if aggregate.claim_id != generated.claim_id:
                raise ValueError("Judge aggregate.claim_id 与输入 claim 不一致")
            if aggregate.n_valid == 0:
                raise RuntimeError(f"D gate 对 {generated.claim_id} 没有任何有效 Judge 判定")
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
            if passed:
                accepted.append(generated)

        if c_artifact.review.answerability is Answerability.OUT_OF_SCOPE:
            # D includes an automatic refusal boundary in addition to the
            # evidence gate; an out-of-scope answer never publishes claims.
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
            claim_gates=gates,
            parent_c_artifact_sha256=_sha_model(c_artifact),
            warnings=[
                *c_artifact.warnings,
                "D 未追加检索，且输出是通过 gate 的 C 主张确定性渲染。",
                (
                    "D Pilot Judge k=1，仅为单次自动门控，不是自一致性稳定性实验。"
                    if self.claim_gate.k == 1
                    else f"D Judge 使用自一致性 k={self.claim_gate.k}。"
                ),
            ],
        )

    @staticmethod
    def _write_cell(
        suite_dir: Path,
        artifact: AblationCellArtifact,
    ) -> AblationCellRecord:
        relative = Path(_safe_id(artifact.question_id)) / f"replicate-{artifact.replicate:02d}" / artifact.arm.value
        final_dir = suite_dir / relative
        if final_dir.exists():
            raise FileExistsError(f"cell 目录已存在：{relative}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact.arm.value}-", dir=final_dir.parent))
        try:
            files = {
                "artifact.json": _json_bytes(artifact.model_dump(mode="json")),
                "review.json": _json_bytes(artifact.review.model_dump(mode="json")),
                "retrieval.jsonl": b"".join(
                    json.dumps(
                        passage.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                    for passage in artifact.passages
                ),
                "claim_gates.jsonl": b"".join(
                    json.dumps(
                        gate.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                    for gate in artifact.claim_gates
                ),
            }
            for name, data in files.items():
                (temporary / name).write_bytes(data)
            manifest = {
                "schema_version": ABLATION_ARTIFACT_VERSION,
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
    ) -> AblationCellRecord:
        relative = Path(_safe_id(question_id)) / f"replicate-{replicate:02d}" / arm.value
        final_dir = suite_dir / relative
        if final_dir.exists():
            raise FileExistsError(f"failure cell 目录已存在：{relative}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir()
        failure_type = type(exc).__name__
        failure_reason = str(exc) or repr(exc)
        _write_atomic(
            final_dir / "failure.json",
            _json_bytes(
                {
                    "schema_version": ABLATION_ARTIFACT_VERSION,
                    "question_id": question_id,
                    "replicate": replicate,
                    "arm": arm.value,
                    "outcome": "failed",
                    "failure_type": failure_type,
                    "failure_reason": failure_reason,
                    "security": {"contains_api_key": False},
                }
            ),
        )
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
        _write_atomic(
            suite_dir / "suite_state.json",
            _json_bytes(state.model_dump(mode="json")),
        )

    def run_suite(
        self,
        requests: Sequence[ReviewRequest],
        *,
        replicates: int,
        out_root: str | Path,
        suite_id: str,
        input_path: str | Path,
    ) -> tuple[Path, PilotAblationSuiteState]:
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
                plan, plan_audit = self.model.plan(request)
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
                planning_failures[request.question_id] = f"{type(exc).__name__}: {exc}"

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
        self._write_state(suite_dir, state)
        _write_atomic(
            suite_dir / "suite_summary.json",
            _json_bytes(state.model_dump(mode="json")),
        )
        return suite_dir, state
