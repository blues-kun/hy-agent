"""Fail-closed contracts and preflight checks for the formal experiments.

This module deliberately does not call Hy3 or any literature service.  It
answers a narrower question before paid/network execution starts: are the
declared inputs and runtime implementations sufficient for the experiment we
are about to name?

Two distinctions are enforced in the schema:

* one user-declared expert reference can be compared with an automatic
  evaluator, but it cannot produce inter-rater reliability;
* an A/B/C/D result table is only complete when every planned question, arm
  and replicate has either a success record or an explicit failure record.

The latter prevents API/schema failures from disappearing from the
denominator.  The repository implements a bounded Pilot A/B/C/D runtime with
explicitly named sparse TF-IDF and frozen-corpus graph arms; the preflight
report does not relabel either one as dense RAG or an expert-curated claim
graph.  Formal scoring and inferential statistics remain separate downstream
work.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from evaluator.schemas import StrictModel
from evaluator.validation import (
    NominalRatingPair,
    OrdinalRatingPair,
    TotalScorePair,
    ValidationInput,
    _nominal_agreement,
    _ordinal_agreement,
    _total_score_agreement,
)


SCHEMA_VERSION = "mitoevidence.experiment-preflight.v1"
EXPERT_CONCORDANCE_SCHEMA_VERSION = "mitoevidence.expert-concordance.v1"
ABLATION_SCHEMA_VERSION = "mitoevidence.ablation.v1"
ABLATION_RUNTIME_SCHEMA_VERSIONS = (
    "mitoevidence.pilot-ablation.v1",
    "mitoevidence.pilot-ablation.v2",
    "mitoevidence.pilot-ablation.v3",
)

FORMAL_DISCRIMINATION_PER_TIER = 20
FORMAL_EXPERT_OUTPUTS = 60
FORMAL_STABILITY_OUTPUTS = 30
FORMAL_STABILITY_REPEATS = 5
FORMAL_ADVERSARIAL_PAIRS = 36
FORMAL_CLEAN_CONTROLS = 60

DEFAULT_EXPERT_REFERENCE_PATHS = (
    "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl",
    "annotation_prelabel/claim_review_sample/claim_review_sample.jsonl",
    "annotation_prelabel/terminology_blacklist/terminology_blacklist.jsonl",
    "annotation_prelabel/review_pool_assessment/review_pool_assessment.jsonl",
)


class ReadinessStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


class ExperimentStage(str, Enum):
    REAL_HY3_PILOT = "real_hy3_pilot"
    EXPERT_REFERENCE_CONCORDANCE = "expert_reference_concordance"
    DISCRIMINATION = "discrimination"
    STABILITY = "stability"
    ADVERSARIAL = "adversarial"
    ABLATION_ABCD = "ablation_abcd"


class AblationArm(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class RunOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ArtifactDigest(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: int | None = Field(default=None, ge=0)


class PreflightCheck(StrictModel):
    code: str
    passed: bool
    required: bool = True
    detail: str
    artifacts: list[ArtifactDigest] = Field(default_factory=list)


class StageReadiness(StrictModel):
    stage: ExperimentStage
    status: ReadinessStatus
    checks: list[PreflightCheck]
    next_command: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_checks(self) -> "StageReadiness":
        failed_required = any(not check.passed and check.required for check in self.checks)
        if self.status is ReadinessStatus.READY and failed_required:
            raise ValueError("ready 阶段不能包含失败的 required check")
        if self.status is ReadinessStatus.BLOCKED and not failed_required:
            raise ValueError("blocked 阶段必须至少包含一个失败的 required check")
        return self


class ExpertReferenceSummary(StrictModel):
    authority: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[ArtifactDigest]
    total_records: int = Field(ge=0)
    labelled_records: int = Field(ge=0)
    id_field_counts: dict[str, int]
    label_field_counts: dict[str, int]
    inter_expert_agreement_computable: bool | None = None
    limitations: list[str] = Field(default_factory=list)


class PreflightSafety(StrictModel):
    network_calls_performed: Literal[False] = False
    contains_api_key: Literal[False] = False
    mutates_experiment_inputs: Literal[False] = False


class ExperimentPreflightReport(StrictModel):
    schema_version: str = SCHEMA_VERSION
    repository: str
    api_key_env: str
    api_key_present: bool
    expert_reference: ExpertReferenceSummary
    stages: list[StageReadiness]
    safety: PreflightSafety = Field(default_factory=PreflightSafety)

    @model_validator(mode="after")
    def _all_stages_once(self) -> "ExperimentPreflightReport":
        stages = [stage.stage for stage in self.stages]
        if Counter(stages) != Counter(ExperimentStage):
            raise ValueError("preflight 必须且只能报告每个实验阶段一次")
        return self


class ExpertOrdinalComparison(StrictModel):
    item_id: str
    dimension: str = Field(pattern=r"^D[1-9]$")
    expert_score: int | None = Field(default=None, ge=0, le=4)
    automatic_score: int | None = Field(default=None, ge=0, le=4)


class ExpertNominalComparison(StrictModel):
    item_id: str
    task: str
    expert_label: str | None = None
    automatic_label: str | None = None
    automatic_error: str | None = None

    @model_validator(mode="after")
    def _labels_and_error_are_coherent(self) -> "ExpertNominalComparison":
        if not self.item_id.strip() or not self.task.strip():
            raise ValueError("nominal item_id/task 不能为空")
        if self.expert_label is not None and not self.expert_label.strip():
            raise ValueError("expert_label 不能是空白字符串")
        if self.automatic_label is not None and not self.automatic_label.strip():
            raise ValueError("automatic_label 不能是空白字符串")
        if self.automatic_label is None:
            if not self.automatic_error or not self.automatic_error.strip():
                raise ValueError("缺失 automatic_label 时必须保留 automatic_error")
        elif self.automatic_error is not None:
            raise ValueError("已有 automatic_label 时不得同时填写 automatic_error")
        return self


class ExpertTotalComparison(StrictModel):
    item_id: str
    expert_score: float | None = Field(default=None, ge=0, le=100)
    automatic_score: float | None = Field(default=None, ge=0, le=100)


class ExpertConcordanceInput(StrictModel):
    """Automatic-evaluator versus one declared expert reference.

    This is intentionally not named ``inter_rater_agreement``.  Its kappa and
    correlation describe concordance with one reference snapshot, not the
    reliability or uncertainty of the expert annotation process.
    """

    schema_version: Literal["mitoevidence.expert-concordance.v1"] = (
        EXPERT_CONCORDANCE_SCHEMA_VERSION
    )
    reference_authority: str = Field(
        description="Provenance declaration, for example user_declared_expert."
    )
    reference_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    automatic_system_role: str = "automatic_evaluator"
    automatic_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    nominal: list[ExpertNominalComparison] = Field(default_factory=list)
    ordinal_0_4: list[ExpertOrdinalComparison] = Field(default_factory=list)
    total_scores: list[ExpertTotalComparison] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_items(self) -> "ExpertConcordanceInput":
        nominal_keys = [(row.task, row.item_id) for row in self.nominal]
        duplicates = sorted(key for key, count in Counter(nominal_keys).items() if count > 1)
        if duplicates:
            raise ValueError(f"nominal (task,item_id) 必须唯一：{duplicates}")
        ordinal_keys = [(row.dimension, row.item_id) for row in self.ordinal_0_4]
        duplicates = sorted(key for key, count in Counter(ordinal_keys).items() if count > 1)
        if duplicates:
            raise ValueError(f"ordinal (dimension,item_id) 必须唯一：{duplicates}")
        ids = [row.item_id for row in self.total_scores]
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"total_scores.item_id 必须唯一：{duplicates}")
        if not self.reference_authority.strip():
            raise ValueError("reference_authority 不能为空")
        if not self.automatic_system_role.strip():
            raise ValueError("automatic_system_role 不能为空")
        return self


class AblationRunRecord(StrictModel):
    question_id: str
    arm: AblationArm
    replicate: int = Field(ge=1)
    outcome: RunOutcome
    run_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    final_score: float | None = Field(default=None, ge=0, le=100)
    d2_support_precision: float | None = Field(default=None, ge=0, le=1)
    d3_evidence_recall: float | None = Field(default=None, ge=0, le=1)
    failure_type: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _success_or_failure_is_auditable(self) -> "AblationRunRecord":
        if not self.question_id.strip():
            raise ValueError("question_id 不能为空")
        metrics = (self.final_score, self.d2_support_precision, self.d3_evidence_recall)
        if self.outcome is RunOutcome.SUCCEEDED:
            if self.run_manifest_sha256 is None or any(value is None for value in metrics):
                raise ValueError("成功运行必须提供 manifest hash、总分、D2 和 D3")
            if self.failure_type is not None or self.failure_reason is not None:
                raise ValueError("成功运行不得同时填写 failure 字段")
        else:
            if not self.failure_type or not self.failure_type.strip():
                raise ValueError("失败运行必须提供 failure_type")
            if not self.failure_reason or not self.failure_reason.strip():
                raise ValueError("失败运行必须提供 failure_reason")
            if any(value is not None for value in metrics):
                raise ValueError("失败运行不得伪造成功指标；分母由 outcome=failed 保留")
        return self


class AblationInput(StrictModel):
    schema_version: Literal["mitoevidence.ablation.v1"] = ABLATION_SCHEMA_VERSION
    protocol_id: str
    question_ids: list[str] = Field(min_length=1)
    replicates_per_arm: int = Field(default=3, ge=1)
    model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: list[AblationRunRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _grid_keys_are_valid(self) -> "AblationInput":
        if not self.protocol_id.strip():
            raise ValueError("protocol_id 不能为空")
        if len(set(self.question_ids)) != len(self.question_ids):
            raise ValueError("question_ids 必须唯一")
        allowed = set(self.question_ids)
        keys: list[tuple[str, AblationArm, int]] = []
        for row in self.records:
            if row.question_id not in allowed:
                raise ValueError(f"记录引用了未声明 question_id：{row.question_id}")
            if row.replicate > self.replicates_per_arm:
                raise ValueError(
                    f"replicate={row.replicate} 超过 replicates_per_arm={self.replicates_per_arm}"
                )
            keys.append((row.question_id, row.arm, row.replicate))
        duplicates = sorted(
            (question, arm.value, replicate)
            for (question, arm, replicate), count in Counter(keys).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"A/B/C/D 运行键必须唯一：{duplicates}")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_path(repo_root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _path_label(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return path.name


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno} 不是合法 JSON：{exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{lineno} 必须是 JSON object")
        rows.append(row)
    return rows


def summarize_expert_reference(
    repo_root: Path,
    paths: Sequence[str | Path],
    *,
    authority: str = "user_declared_expert",
) -> ExpertReferenceSummary:
    """Hash and count caller-declared reference files without changing labels."""

    id_fields = ("question_id", "review_id", "term_id", "assessment_id", "item_id")
    label_fields = (
        "expert_decision",
        "expert_answerability",
        "decision",
        "answerability",
        "correct",
        # Kept for backward-compatible ingestion when the repository owner has
        # declared the existing snapshot authoritative but has not migrated
        # historical field names yet.  The field name itself is not evidence
        # of annotation authority; ``authority`` is explicit above.
        "ai_decision",
    )
    artifacts: list[ArtifactDigest] = []
    id_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    total = labelled = 0
    for raw in paths:
        path = Path(raw)
        path = path if path.is_absolute() else repo_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _jsonl(path)
        total += len(rows)
        for row in rows:
            for field in id_fields:
                if row.get(field) not in (None, ""):
                    id_counts[field] += 1
                    break
            matched = False
            for field in label_fields:
                if row.get(field) not in (None, ""):
                    label_counts[field] += 1
                    matched = True
                    break
            labelled += int(matched)
        try:
            label = str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            label = path.name
        artifacts.append(ArtifactDigest(path=label, sha256=_sha256(path), records=len(rows)))

    manifest_bytes = json.dumps(
        [item.model_dump(mode="json") for item in artifacts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ExpertReferenceSummary(
        authority=authority,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        files=artifacts,
        total_records=total,
        labelled_records=labelled,
        id_field_counts=dict(sorted(id_counts.items())),
        label_field_counts=dict(sorted(label_counts.items())),
        inter_expert_agreement_computable=None,
        limitations=[
            "自定义参考未声明独立评分者结构；不能据此推断双专家一致性是否可计算。"
        ],
    )


def _default_expert_gold_summary(
    repo_root: Path,
) -> tuple[ExpertReferenceSummary, bool, str]:
    """Use the owner-designated, hash-pinned manifest when it is available."""

    from evaluator.expert_gold import ExpertGoldAuditError, audit_expert_gold

    manifest_path = repo_root / "annotation_prelabel/expert_gold_manifest.json"
    if not manifest_path.is_file():
        summary = summarize_expert_reference(
            repo_root,
            DEFAULT_EXPERT_REFERENCE_PATHS,
            authority="user_declared_expert_without_designation_manifest",
        )
        return summary, False, "专家金标 designation manifest 不存在"
    try:
        audit = audit_expert_gold(manifest_path, repo_root=repo_root)
    except ExpertGoldAuditError as exc:
        summary = summarize_expert_reference(
            repo_root,
            DEFAULT_EXPERT_REFERENCE_PATHS,
            authority="user_declared_expert_manifest_invalid",
        )
        return summary, False, f"专家金标 manifest 审计异常：{exc}"

    artifacts: list[ArtifactDigest] = []
    id_counts: Counter[str] = Counter()
    for dataset in audit.get("datasets", {}).values():
        path = str(dataset.get("path") or "")
        digest = dataset.get("sha256")
        count = int(dataset.get("record_count") or 0)
        if path and isinstance(digest, str) and len(digest) == 64:
            artifacts.append(ArtifactDigest(path=path, sha256=digest, records=count))
        id_field = dataset.get("id_field")
        if isinstance(id_field, str):
            id_counts[id_field] += int(dataset.get("unique_non_empty_ids") or 0)
    total = int(audit.get("total_records") or 0)
    summary = ExpertReferenceSummary(
        authority=str(audit.get("designation") or "expert_consensus_gold"),
        manifest_sha256=_sha256(manifest_path),
        files=artifacts,
        total_records=total,
        labelled_records=total if audit.get("ok") else 0,
        id_field_counts=dict(sorted(id_counts.items())),
        label_field_counts={"manifest_designated_gold_records": total},
        inter_expert_agreement_computable=bool(
            audit.get("inter_expert_agreement", {}).get("computable")
        ),
        limitations=[str(item) for item in audit.get("warnings") or []],
    )
    errors = audit.get("errors") or []
    detail = (
        f"expert_consensus_gold manifest 已核验：{total} 条、{len(artifacts)} 个哈希固定数据集"
        if audit.get("ok")
        else "专家金标 manifest 审计失败：" + "；".join(str(error) for error in errors)
    )
    return summary, bool(audit.get("ok")), detail


def build_pilot_answerability_concordance(
    repo_root: str | Path,
    suite_dir: str | Path,
) -> ExpertConcordanceInput:
    """Bind a real-Hy3 Pilot suite to the five expert answerability labels.

    Missing/failed Pilot runs become nominal rows with ``automatic_label=null``
    and an explicit error; they are never omitted.  Offline smoke suites are
    rejected because they are not model observations.
    """

    from app.schemas import GeneratedReview
    from evaluator.expert_gold import (
        ExpertGoldAuditError,
        audit_expert_gold,
        load_expert_gold_records,
    )

    root = Path(repo_root).resolve()
    suite = Path(suite_dir)
    suite = suite.resolve() if suite.is_absolute() else (root / suite).resolve()
    if not suite.is_dir():
        raise ValueError(f"Pilot suite 目录不存在：{suite.name}")
    summary_path = suite / "suite_summary.json"
    try:
        suite_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 suite_summary.json：{exc}") from exc
    if suite_summary.get("run_kind") != "hy3":
        raise ValueError("只允许真实 run_kind=hy3 的套件；offline_smoke 不能用于专家一致度")

    manifest_path = root / "annotation_prelabel/expert_gold_manifest.json"
    try:
        gold_audit = audit_expert_gold(manifest_path, repo_root=root)
        if not gold_audit["ok"]:
            raise ValueError("；".join(gold_audit["errors"]))
        gold = load_expert_gold_records(manifest_path, repo_root=root)
    except ExpertGoldAuditError as exc:
        raise ValueError(f"专家金标审计失败：{exc}") from exc

    summary_records = suite_summary.get("records") or []
    if not isinstance(summary_records, list):
        raise ValueError("suite_summary.records 必须是 array")
    ids = [row.get("pilot_id") for row in summary_records if isinstance(row, dict)]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"suite_summary.pilot_id 重复：{duplicates}")
    by_id = {
        str(row.get("pilot_id")): row
        for row in summary_records
        if isinstance(row, dict) and row.get("pilot_id") is not None
    }

    rows: list[ExpertNominalComparison] = []
    for reference in gold["pilot_questions"]:
        item_id = str(reference["question_id"])
        run_record = by_id.get(item_id)
        automatic_label: str | None = None
        automatic_error: str | None = None
        if run_record is None:
            automatic_error = "suite_summary 缺少该 Pilot 运行记录"
        elif run_record.get("ok") is not True:
            automatic_error = (
                f"{run_record.get('error_type') or 'RunError'}: "
                f"{run_record.get('error') or '运行失败但未提供错误详情'}"
            )
        else:
            run_name = str(run_record.get("run_dir") or item_id)
            run_dir = (suite / run_name).resolve()
            try:
                run_dir.relative_to(suite)
            except ValueError:
                automatic_error = f"run_dir 路径越界：{run_name}"
            if automatic_error is None:
                try:
                    manifest = json.loads(
                        (run_dir / "manifest.json").read_text(encoding="utf-8")
                    )
                    if manifest.get("run_kind") != "hy3":
                        raise ValueError("单题 manifest.run_kind 不是 hy3")
                    if manifest.get("question_id") != item_id:
                        raise ValueError("单题 manifest.question_id 与专家 item_id 不一致")
                    review_path = run_dir / "review.json"
                    expected_hash = (
                        manifest.get("files", {}).get("review.json", {}).get("sha256")
                    )
                    if not isinstance(expected_hash, str) or _sha256(review_path) != expected_hash:
                        raise ValueError("review.json 哈希与单题 manifest 不一致")
                    review = GeneratedReview.model_validate_json(
                        review_path.read_text(encoding="utf-8")
                    )
                    automatic_label = review.answerability.value
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    automatic_error = f"成功记录的审计产物无效：{exc}"
        rows.append(
            ExpertNominalComparison(
                item_id=item_id,
                task="pilot_answerability",
                expert_label=(
                    str(reference["answerability"])
                    if reference.get("answerability") is not None
                    else None
                ),
                automatic_label=automatic_label,
                automatic_error=automatic_error,
            )
        )

    rubric_path = root / "configs/rubric_v0_1.yaml"
    return ExpertConcordanceInput(
        reference_authority=str(gold_audit["designation"]),
        reference_manifest_sha256=_sha256(manifest_path),
        rubric_config_sha256=_sha256(rubric_path),
        automatic_system_role="hy3_application_answerability",
        automatic_artifact_sha256=_sha256(summary_path),
        nominal=rows,
    )


def build_ablation_answerability_concordance(
    repo_root: str | Path,
    suite_dir: str | Path,
    *,
    allow_nonformal: bool = False,
) -> ExpertConcordanceInput:
    """Bind every A/B/C/D cell to the pinned expert answerability label.

    This is a small Pilot diagnostic, not the proposal's 60-output
    dimension/total-score concordance study.  Failed cells remain nominal rows
    with ``automatic_label=null`` and their recorded error.
    """

    from app.ablation import (
        SUITE_EVIDENCE_MANIFEST_COPY,
        SUITE_INPUT_SNAPSHOT_COPY,
        AblationCellArtifact,
        CellOutcome,
        PilotAblationSuiteState,
        SuiteStatus,
    )
    from evaluator.ablation_artifacts import audit_pilot_ablation_artifacts
    from evaluator.expert_gold import (
        ExpertGoldAuditError,
        audit_expert_gold,
        load_expert_gold_records,
    )

    root = Path(repo_root).resolve()
    suite = Path(suite_dir)
    suite = suite.resolve() if suite.is_absolute() else (root / suite).resolve()
    if not suite.is_dir():
        raise ValueError(f"A/B/C/D suite 目录不存在：{suite}")
    state_path = suite / "suite_state.json"
    summary_path = suite / "suite_summary.json"
    for journal_path in (state_path, summary_path):
        if journal_path.is_symlink() or not journal_path.is_file():
            raise ValueError(f"{journal_path.name} 缺失或为不允许的 symlink")
    try:
        state_bytes = state_path.read_bytes()
        summary_bytes = summary_path.read_bytes()
        if state_bytes != summary_bytes:
            raise ValueError("suite_state.json 与 suite_summary.json 必须逐字节一致")
        state = PilotAblationSuiteState.model_validate_json(
            summary_bytes
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"A/B/C/D suite_summary 不合规：{exc}") from exc
    if state.status is not SuiteStatus.COMPLETED and not allow_nonformal:
        raise ValueError("只有 completed A/B/C/D suite 可以构造专家 concordance")

    def bound_snapshot(
        declared_path: str,
        archived_name: str,
        expected_sha256: str,
        label: str,
    ) -> Path:
        raw = Path(declared_path)
        if not raw.is_absolute() and ".." in raw.parts:
            raise ValueError(f"{label} 声明路径包含越界段：{declared_path!r}")
        candidates = [suite / archived_name]
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend((suite / raw, root / raw))
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError as exc:
                raise ValueError(f"无法解析 {label}：{exc}") from exc
            if resolved in seen:
                continue
            seen.add(resolved)
            if not candidate.exists():
                continue
            allowed = False
            for boundary in (suite, root):
                try:
                    resolved.relative_to(boundary)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise ValueError(f"{label} 越出 repo/suite：{candidate}")
            if candidate.is_symlink() or not resolved.is_file():
                raise ValueError(f"{label} 必须是非 symlink 常规文件：{candidate}")
            actual = _sha256(resolved)
            if actual != expected_sha256:
                raise ValueError(
                    f"{label} SHA-256 与 suite 声明不一致：{actual} != {expected_sha256}"
                )
            return resolved
        raise ValueError(f"找不到 suite 绑定的 {label}：{declared_path!r}")

    input_snapshot_path = bound_snapshot(
        state.input_snapshot.path,
        SUITE_INPUT_SNAPSHOT_COPY,
        state.input_snapshot.sha256,
        "input snapshot",
    )
    evidence_manifest_path = bound_snapshot(
        state.evidence_manifest_path,
        SUITE_EVIDENCE_MANIFEST_COPY,
        state.evidence_manifest_sha256,
        "evidence manifest",
    )
    try:
        evidence_payload = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"suite 绑定的 evidence manifest 不是有效 JSON：{exc}") from exc
    if not isinstance(evidence_payload, dict):
        raise ValueError("suite 绑定的 evidence manifest 顶层必须是 object")

    input_rows: dict[str, dict[str, Any]] = {}
    try:
        input_lines = input_snapshot_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"无法读取 suite input snapshot：{exc}") from exc
    for line_number, line in enumerate(input_lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"input snapshot 第 {line_number} 行 JSON 无效：{exc}") from exc
        if not isinstance(row, dict) or not row.get("question_id"):
            raise ValueError(f"input snapshot 第 {line_number} 行缺 question_id object")
        question_id = str(row["question_id"])
        if question_id in input_rows:
            raise ValueError(f"input snapshot question_id 重复：{question_id}")
        input_rows[question_id] = row
    missing_input_ids = [
        question_id
        for question_id in state.input_snapshot.question_ids
        if question_id not in input_rows
    ]
    if missing_input_ids:
        raise ValueError(f"input snapshot 缺少 suite question_id：{missing_input_ids}")

    manifest_path = root / "annotation_prelabel/expert_gold_manifest.json"
    try:
        gold_audit = audit_expert_gold(manifest_path, repo_root=root)
        if not gold_audit["ok"]:
            raise ValueError("；".join(gold_audit["errors"]))
        gold = load_expert_gold_records(manifest_path, repo_root=root)
    except ExpertGoldAuditError as exc:
        raise ValueError(f"专家金标审计失败：{exc}") from exc
    expert_by_id = {
        str(row["question_id"]): row for row in gold["pilot_questions"]
    }
    unknown_questions = sorted(
        set(state.input_snapshot.question_ids) - set(expert_by_id)
    )
    if unknown_questions:
        raise ValueError(f"A/B/C/D suite 含非金标 Pilot ID：{unknown_questions}")
    if not allow_nonformal:
        pilot_dataset = gold_audit.get("datasets", {}).get("pilot_questions", {})
        pinned_pilot_sha256 = pilot_dataset.get("sha256")
        if state.input_snapshot.sha256 != pinned_pilot_sha256:
            raise ValueError(
                "正式 concordance 要求 input snapshot 精确绑定 expert manifest 的 "
                "pilot_questions dataset hash"
            )
        neutral_mismatches = [
            question_id
            for question_id in state.input_snapshot.question_ids
            if (
                str(input_rows[question_id].get("question") or "")
                != str(expert_by_id[question_id].get("question") or "")
                or str(input_rows[question_id].get("scope") or "")
                != str(expert_by_id[question_id].get("scope") or "")
            )
        ]
        if neutral_mismatches:
            raise ValueError(
                "正式 concordance 的 neutral question/scope 与专家 manifest 数据集不一致："
                f"{neutral_mismatches}"
            )
        evidence_relative = Path(state.evidence_manifest_path)
        if evidence_relative.is_absolute() or ".." in evidence_relative.parts:
            raise ValueError("正式 concordance 的 evidence manifest 必须是 repo 内相对路径")
        canonical_evidence = (root / evidence_relative).resolve()
        try:
            canonical_evidence.relative_to(root)
        except ValueError as exc:
            raise ValueError("正式 concordance 的 evidence manifest 越出 repo") from exc
        if canonical_evidence.is_symlink() or not canonical_evidence.is_file():
            raise ValueError("正式 concordance 找不到非 symlink evidence manifest")
        if _sha256(canonical_evidence) != state.evidence_manifest_sha256:
            raise ValueError(
                "正式 concordance 的 evidence manifest hash 与当前 repo 冻结快照不一致"
            )

    artifact_audit = audit_pilot_ablation_artifacts(
        suite,
        allow_test_fixture=allow_nonformal,
    )
    if not allow_nonformal and artifact_audit.get("production_ready") is not True:
        first_errors = artifact_audit.get("errors") or []
        detail = "；".join(
            f"{error.get('code')}: {error.get('detail')}"
            for error in first_errors[:3]
        )
        raise ValueError(
            "正式 concordance 要求 artifact-level production audit 通过"
            + (f"：{detail}" if detail else "")
        )
    if allow_nonformal:
        legacy_waivers = {
            "D_JUDGE_PROVENANCE_UNAVAILABLE_LEGACY_V1",
        }
        nonformal_fatal = [
            error
            for error in artifact_audit.get("errors") or []
            if not (
                state.schema_version == "mitoevidence.pilot-ablation.v1"
                and error.get("code") in legacy_waivers
            )
        ]
        if nonformal_fatal:
            detail = "；".join(
                f"{error.get('code')}: {error.get('detail')}"
                for error in nonformal_fatal[:3]
            )
            raise ValueError(f"非正式 concordance 的 artifact audit 仍失败：{detail}")

    rows: list[ExpertNominalComparison] = []
    for record in state.records:
        expert = expert_by_id[record.question_id]
        automatic_label: str | None = None
        automatic_error: str | None = None
        if record.outcome is CellOutcome.FAILED:
            automatic_error = (
                f"{record.failure_type or 'RunError'}: "
                f"{record.failure_reason or '失败 cell 未提供原因'}"
            )
        else:
            cell_dir = (suite / record.cell_dir).resolve()
            try:
                cell_dir.relative_to(suite)
            except ValueError:
                automatic_error = f"cell_dir 路径越界：{record.cell_dir}"
            if automatic_error is None:
                try:
                    manifest_path_cell = cell_dir / "manifest.json"
                    if _sha256(manifest_path_cell) != record.cell_manifest_sha256:
                        raise ValueError("cell manifest hash 与 suite record 不一致")
                    manifest = json.loads(
                        manifest_path_cell.read_text(encoding="utf-8")
                    )
                    artifact_path = cell_dir / "artifact.json"
                    expected = (
                        manifest.get("files", {}).get("artifact.json", {}).get("sha256")
                    )
                    if not isinstance(expected, str) or _sha256(artifact_path) != expected:
                        raise ValueError("artifact.json hash 与 cell manifest 不一致")
                    artifact = AblationCellArtifact.model_validate_json(
                        artifact_path.read_text(encoding="utf-8")
                    )
                    if (
                        artifact.question_id != record.question_id
                        or artifact.replicate != record.replicate
                        or artifact.arm is not record.arm
                    ):
                        raise ValueError("artifact grid key 与 suite record 不一致")
                    source_row = input_rows[record.question_id]
                    if (
                        artifact.request.question != str(source_row.get("question") or "")
                        or artifact.request.scope != str(source_row.get("scope") or "")
                    ):
                        raise ValueError("artifact request 与绑定 input snapshot 不一致")
                    automatic_label = artifact.review.answerability.value
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    automatic_error = f"成功 cell 的审计产物无效：{exc}"
        rows.append(
            ExpertNominalComparison(
                item_id=f"{record.question_id}:replicate-{record.replicate:02d}",
                task=f"ablation_{record.arm.value}_answerability",
                expert_label=(
                    str(expert["answerability"])
                    if expert.get("answerability") is not None
                    else None
                ),
                automatic_label=automatic_label,
                automatic_error=automatic_error,
            )
        )

    rubric_path = root / "configs/rubric_v0_1.yaml"
    return ExpertConcordanceInput(
        reference_authority=str(gold_audit["designation"]),
        reference_manifest_sha256=_sha256(manifest_path),
        rubric_config_sha256=_sha256(rubric_path),
        automatic_system_role=(
            "nonformal_hy3_pilot_ablation_answerability"
            if allow_nonformal
            else "hy3_pilot_ablation_answerability"
        ),
        automatic_artifact_sha256=_sha256(summary_path),
        nominal=rows,
    )


def analyze_expert_concordance(data: ExpertConcordanceInput) -> dict[str, Any]:
    """Compare automatic scores with one expert snapshot, fully offline."""

    warnings: list[dict[str, Any]] = []
    nominal_grouped: dict[str, list[ExpertNominalComparison]] = defaultdict(list)
    for row in data.nominal:
        nominal_grouped[row.task].append(row)
    nominal_tasks: dict[str, Any] = {}
    for task in sorted(nominal_grouped):
        records = [
            NominalRatingPair(
                item_id=row.item_id,
                rater_a=row.expert_label,
                rater_b=row.automatic_label,
            )
            for row in nominal_grouped[task]
        ]
        section_warnings: list[dict[str, Any]] = []
        result = _nominal_agreement(records, section_warnings)
        result["role_a"] = "expert_reference"
        result["role_b"] = data.automatic_system_role
        result["automatic_errors"] = {
            row.item_id: row.automatic_error
            for row in nominal_grouped[task]
            if row.automatic_error is not None
        }
        nominal_tasks[task] = result
        for warning in section_warnings:
            warning["section"] = warning["section"].replace(
                "agreement.nominal", f"expert_concordance.nominal.{task}"
            )
        warnings.extend(section_warnings)

    grouped: dict[str, list[ExpertOrdinalComparison]] = defaultdict(list)
    for row in data.ordinal_0_4:
        grouped[row.dimension].append(row)
    dimensions: dict[str, Any] = {}
    for dimension in sorted(grouped):
        records = [
            OrdinalRatingPair(
                item_id=row.item_id,
                rater_a=row.expert_score,
                rater_b=row.automatic_score,
            )
            for row in grouped[dimension]
        ]
        section_warnings: list[dict[str, Any]] = []
        result = _ordinal_agreement(records, section_warnings)
        result["role_a"] = "expert_reference"
        result["role_b"] = data.automatic_system_role
        dimensions[dimension] = result
        for warning in section_warnings:
            warning["section"] = warning["section"].replace(
                "agreement.ordinal_0_4", f"expert_concordance.{dimension}"
            )
        warnings.extend(section_warnings)

    total_records = [
        TotalScorePair(
            item_id=row.item_id,
            rater_a=row.expert_score,
            rater_b=row.automatic_score,
        )
        for row in data.total_scores
    ]
    total_warnings: list[dict[str, Any]] = []
    totals = _total_score_agreement(total_records, total_warnings)
    # ICC between an automatic method and a single reference is not an estimate
    # of expert inter-rater reliability.  Keep the arithmetic for audit but
    # attach an explicit interpretation boundary.
    totals["role_a"] = "expert_reference"
    totals["role_b"] = data.automatic_system_role
    totals["icc_interpretation"] = "method-reference absolute agreement; not expert reliability"
    for warning in total_warnings:
        warning["section"] = warning["section"].replace(
            "agreement.total_scores", "expert_concordance.total_scores"
        )
    warnings.extend(total_warnings)
    return {
        "schema_version": EXPERT_CONCORDANCE_SCHEMA_VERSION,
        "reference_authority": data.reference_authority,
        "reference_manifest_sha256": data.reference_manifest_sha256,
        "rubric_config_sha256": data.rubric_config_sha256,
        "automatic_system_role": data.automatic_system_role,
        "automatic_artifact_sha256": data.automatic_artifact_sha256,
        "interpretation": (
            f"{data.automatic_system_role} concordance with one declared expert reference; "
            "does not estimate inter-expert reliability"
        ),
        "nominal_tasks": nominal_tasks,
        "dimensions": dimensions,
        "total_scores": totals,
        "warnings": warnings,
    }


def ablation_grid_audit(data: AblationInput) -> dict[str, Any]:
    """Report the complete planned grid; failed/missing cells remain visible."""

    records = {(row.question_id, row.arm, row.replicate): row for row in data.records}
    missing: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    by_arm: dict[str, dict[str, Any]] = {}
    expected_total = len(data.question_ids) * len(AblationArm) * data.replicates_per_arm
    for question_id in data.question_ids:
        for arm in AblationArm:
            for replicate in range(1, data.replicates_per_arm + 1):
                row = records.get((question_id, arm, replicate))
                if row is None:
                    missing.append(
                        {"question_id": question_id, "arm": arm.value, "replicate": replicate}
                    )
                else:
                    outcomes[row.outcome.value] += 1

    for arm in AblationArm:
        arm_rows = [row for row in data.records if row.arm is arm]
        scores = [float(row.final_score) for row in arm_rows if row.final_score is not None]
        by_arm[arm.value] = {
            "expected": len(data.question_ids) * data.replicates_per_arm,
            "recorded": len(arm_rows),
            "succeeded": sum(row.outcome is RunOutcome.SUCCEEDED for row in arm_rows),
            "failed": sum(row.outcome is RunOutcome.FAILED for row in arm_rows),
            "mean_final_score_successes_only": statistics.fmean(scores) if scores else None,
            "score_denominator": len(scores),
        }
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "protocol_id": data.protocol_id,
        "expected_grid_cells": expected_total,
        "recorded_grid_cells": len(data.records),
        "missing_grid_cells": len(missing),
        "outcomes": dict(sorted(outcomes.items())),
        "grid_complete": not missing,
        "missing": missing,
        "by_arm": by_arm,
        "interpretation": (
            "descriptive completeness audit only; formal Friedman/Wilcoxon/Holm and "
            "cluster-bootstrap analysis must run after the full grid is materialized"
        ),
    }


def _check(
    code: str,
    passed: bool,
    detail: str,
    *,
    required: bool = True,
    artifacts: Sequence[ArtifactDigest] = (),
) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        passed=passed,
        required=required,
        detail=detail,
        artifacts=list(artifacts),
    )


def _stage(
    stage: ExperimentStage,
    checks: Sequence[PreflightCheck],
    *,
    next_command: str | None = None,
    limitations: Sequence[str] = (),
    unsupported: bool = False,
) -> StageReadiness:
    failed = any(not check.passed and check.required for check in checks)
    status = (
        ReadinessStatus.UNSUPPORTED
        if unsupported
        else ReadinessStatus.BLOCKED
        if failed
        else ReadinessStatus.READY
    )
    return StageReadiness(
        stage=stage,
        status=status,
        checks=list(checks),
        next_command=next_command if status is ReadinessStatus.READY else None,
        limitations=list(limitations),
    )


def _load_optional_model(path: Path | None, model_cls: type[StrictModel]) -> tuple[Any, str | None]:
    if path is None or not path.is_file():
        return None, "未提供输入文件" if path is None else f"输入文件不存在：{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_cls.model_validate(payload), None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"输入不合规：{exc}"


def _load_optional_ablation_input(
    path: Path | None,
) -> tuple[Any, str | None, str | None]:
    """Load one complete ablation schema without projecting unknown fields."""

    if path is None or not path.is_file():
        return (
            None,
            None,
            "未提供输入文件" if path is None else f"输入文件不存在：{path.name}",
        )
    try:
        from app.ablation import PilotAblationSuiteState

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("A/B/C/D 输入顶层必须是 JSON object")
        declared = payload.get("schema_version")
        if declared is not None and not isinstance(declared, str):
            raise ValueError("schema_version 必须是字符串")
        if declared in ABLATION_RUNTIME_SCHEMA_VERSIONS:
            return PilotAblationSuiteState.model_validate(payload), declared, None
        if declared == ABLATION_SCHEMA_VERSION:
            return AblationInput.model_validate(payload), declared, None
        if declared is not None:
            raise ValueError(f"不支持的 ablation schema_version：{declared!r}")

        has_legacy_identity = "protocol_id" in payload
        has_runtime_identity = "suite_id" in payload
        if has_legacy_identity and has_runtime_identity:
            raise ValueError("缺少 schema_version 且 protocol_id/suite_id 混合")
        if has_runtime_identity:
            raise ValueError("PilotAblationSuiteState 必须显式声明 schema_version")
        if not has_legacy_identity:
            raise ValueError("无法明确判别 ablation 输入 Schema")
        # Only the historical formal-analysis payload may omit the version.
        return (
            AblationInput.model_validate(payload),
            ABLATION_SCHEMA_VERSION,
            None,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, None, f"输入不合规：{exc}"


def _corpus_check(repo_root: Path) -> PreflightCheck:
    manifest_path = repo_root / "eval/data/evidence_pool_manifest.json"
    artifact = (
        [ArtifactDigest(path="eval/data/evidence_pool_manifest.json", sha256=_sha256(manifest_path))]
        if manifest_path.is_file()
        else []
    )
    if not manifest_path.is_file():
        return _check("FROZEN_CORPUS_INTEGRITY", False, "冻结语料 manifest 不存在")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_entries = manifest.get("fulltext") or []
        if not all_entries:
            raise ValueError("manifest.fulltext 为空")
        entries = [entry for entry in all_entries if entry.get("path")]
        unavailable = [entry for entry in all_entries if not entry.get("path")]
        if not entries:
            raise ValueError("manifest 没有任何已冻结全文")
        errors: list[str] = []
        verified = 0
        for entry in entries:
            relative = entry.get("path")
            expected = entry.get("sha256")
            if not expected:
                errors.append(f"PMID:{entry.get('pmid', '?')} 已声明 path 但缺 sha256")
                continue
            source = (repo_root / relative).resolve()
            if repo_root.resolve() not in source.parents:
                errors.append(f"路径越界：{relative}")
            elif not source.is_file():
                errors.append(f"文件缺失：{relative}")
            elif _sha256(source) != expected:
                errors.append(f"哈希不匹配：{relative}")
            else:
                verified += 1
        return _check(
            "FROZEN_CORPUS_INTEGRITY",
            not errors,
            f"冻结 OA XML 核验 {verified}/{len(entries)}；另有 {len(unavailable)} 篇明确记录为全文不可用"
            + (f"；{'；'.join(errors)}" if errors else ""),
            artifacts=artifact,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _check(
            "FROZEN_CORPUS_INTEGRITY", False, f"冻结语料 manifest 不合规：{exc}", artifacts=artifact
        )


def _pilot_check(repo_root: Path) -> PreflightCheck:
    path = repo_root / DEFAULT_EXPERT_REFERENCE_PATHS[0]
    if not path.is_file():
        return _check("PILOT_QUESTIONS", False, "Pilot 问题文件不存在")
    try:
        rows = _jsonl(path)
        ids = [row.get("question_id") for row in rows]
        passed = len(rows) >= 5 and None not in ids and len(ids) == len(set(ids))
        return _check(
            "PILOT_QUESTIONS",
            passed,
            f"可识别且 ID 唯一的 Pilot 问题：{len(set(ids)) if None not in ids else 0}/5",
            artifacts=[
                ArtifactDigest(
                    path=DEFAULT_EXPERT_REFERENCE_PATHS[0], sha256=_sha256(path), records=len(rows)
                )
            ],
        )
    except (OSError, ValueError) as exc:
        return _check("PILOT_QUESTIONS", False, f"Pilot 问题不合规：{exc}")


def build_experiment_preflight(
    repo_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    api_key_env: str = "HY3_API_KEY",
    expert_reference_paths: Sequence[str | Path] = DEFAULT_EXPERT_REFERENCE_PATHS,
    reference_authority: str = "user_declared_expert",
    expert_concordance_input: str | Path | None = None,
    validation_input: str | Path | None = None,
    ablation_input: str | Path | None = None,
) -> ExperimentPreflightReport:
    """Build a read-only, network-free readiness report."""

    root = Path(repo_root).resolve()
    env = os.environ if environment is None else environment
    key_present = bool(str(env.get(api_key_env, "")).strip())
    uses_default_reference = tuple(str(Path(path)) for path in expert_reference_paths) == tuple(
        str(Path(path)) for path in DEFAULT_EXPERT_REFERENCE_PATHS
    )
    if uses_default_reference:
        references, reference_audit_ok, reference_audit_detail = _default_expert_gold_summary(
            root
        )
    else:
        references = summarize_expert_reference(
            root, expert_reference_paths, authority=reference_authority
        )
        reference_audit_ok = True
        reference_audit_detail = (
            "使用调用方显式声明的自定义专家参考；已固定逐文件哈希，"
            "但未套用仓库默认 expert_consensus_gold manifest"
        )

    code_checks = [
        _check(
            "HY3_PILOT_RUNNER",
            (root / "scripts/run_pilot_suite.py").is_file()
            and (root / "app/hy3_review.py").is_file(),
            "真实 Hy3 五题入口与客户端均存在",
        ),
        _check(
            "HY3_API_KEY_PRESENT",
            key_present,
            f"环境变量 {api_key_env} {'已设置' if key_present else '未设置'}；报告不读取或保存其值",
        ),
        _pilot_check(root),
        _corpus_check(root),
    ]
    stages: list[StageReadiness] = [
        _stage(
            ExperimentStage.REAL_HY3_PILOT,
            code_checks,
            next_command=(
                ".venv/bin/python scripts/run_pilot_suite.py "
                "--suite-id pilot5-hy3-$(date -u +%Y%m%dT%H%M%SZ) --continue-on-error"
            ),
            limitations=[
                "五题运行是有审计记录的 Hy3 工程 Pilot，不等同于正式 A/B/C/D 性能实验。"
            ],
        )
    ]

    concordance_path = _input_path(root, expert_concordance_input)
    concordance, concordance_error = _load_optional_model(
        concordance_path, ExpertConcordanceInput
    )
    concordance_checks = [
        _check(
            "EXPERT_REFERENCE_PROVENANCE",
            reference_audit_ok,
            reference_audit_detail,
            artifacts=references.files,
        ),
        _check(
            "EXPERT_REFERENCE_ROWS",
            references.total_records > 0 and references.labelled_records == references.total_records,
            f"调用方声明的专家参考共 {references.labelled_records}/{references.total_records} 条带标签",
            artifacts=references.files,
        ),
        _check(
            "INTER_EXPERT_AGREEMENT_AVAILABLE",
            references.inter_expert_agreement_computable is True,
            (
                "存在独立逐评审标签，可另行计算 inter-expert agreement"
                if references.inter_expert_agreement_computable is True
                else "当前是单份合并专家结果；inter-expert κ/ICC 不可计算"
            ),
            required=False,
        ),
        _check(
            "AUTOMATIC_VS_EXPERT_INPUT",
            concordance is not None,
            concordance_error or "自动评估器—专家参考配对输入通过严格 Schema",
        ),
    ]
    if concordance is not None:
        complete_totals = sum(
            row.expert_score is not None and row.automatic_score is not None
            for row in concordance.total_scores
        )
        critical_dimensions = ("D2", "D4", "D5", "D6")
        critical_counts = {
            dimension: sum(
                row.dimension == dimension
                and row.expert_score is not None
                and row.automatic_score is not None
                for row in concordance.ordinal_0_4
            )
            for dimension in critical_dimensions
        }
        concordance_checks.extend(
            [
                _check(
                    "REFERENCE_HASH_MATCH",
                    concordance.reference_manifest_sha256 == references.manifest_sha256,
                    "配对输入绑定到当前专家参考快照"
                    if concordance.reference_manifest_sha256 == references.manifest_sha256
                    else "配对输入的 reference_manifest_sha256 与当前专家快照不一致",
                ),
                _check(
                    "FORMAL_EXPERT_SAMPLE_SIZE",
                    complete_totals >= FORMAL_EXPERT_OUTPUTS,
                    f"完整自动—专家总分对 {complete_totals}/{FORMAL_EXPERT_OUTPUTS}",
                ),
                _check(
                    "FORMAL_CRITICAL_DIMENSIONS",
                    all(
                        count >= FORMAL_EXPERT_OUTPUTS
                        for count in critical_counts.values()
                    ),
                    f"关键维度完整自动—专家评分对 {critical_counts}；"
                    f"正式目标每维 {FORMAL_EXPERT_OUTPUTS}",
                ),
            ]
        )
    stages.append(
        _stage(
            ExperimentStage.EXPERT_REFERENCE_CONCORDANCE,
            concordance_checks,
            next_command=(
                ".venv/bin/python scripts/analyze_expert_concordance.py "
                f"--input {_path_label(root, concordance_path)} "
                "--output results/expert_concordance.json"
                if concordance_path is not None
                else None
            ),
            limitations=[
                "只有一个专家参考时可报告自动评估器—专家 concordance，不能报告双专家 κ/ICC。",
                *references.limitations,
            ],
        )
    )

    validation_path = _input_path(root, validation_input)
    validation, validation_error = _load_optional_model(validation_path, ValidationInput)
    validation_artifact = (
        [ArtifactDigest(path=validation_path.name, sha256=_sha256(validation_path))]
        if validation_path is not None and validation_path.is_file()
        else []
    )
    base_validation_check = _check(
        "VALIDATION_INPUT_SCHEMA",
        validation is not None,
        validation_error or "有效性输入通过严格 Schema",
        artifacts=validation_artifact,
    )
    tier_counts = {tier: 0 for tier in ("good", "medium", "bad")}
    if validation is not None:
        for row in validation.discrimination:
            if row.score is not None:
                tier_counts[row.quality_tier.value] += 1
    terminology_pair_runtime = (
        (root / "app/terminology_pair_pilot.py").is_file()
        and (root / "scripts/run_terminology_pair_pilot.py").is_file()
        and (root / "scripts/analyze_terminology_pair_pilot.py").is_file()
    )
    terminology_pair_check = _check(
        "TERMINOLOGY_PAIR_PILOT_RUNTIME",
        terminology_pair_runtime,
        (
            "已实现盲法 wrong/correct 术语/条件错误二元成对 Pilot；"
            "模型只见左右文本，失败保留且支持严格 resume"
            if terminology_pair_runtime
            else "术语/条件错误二元成对 Pilot runtime 不完整"
        ),
        required=False,
    )
    discrimination_checks = [base_validation_check, terminology_pair_check]
    if validation is not None:
        discrimination_checks.append(
            _check(
                "FORMAL_DISCRIMINATION_SIZE",
                all(count >= FORMAL_DISCRIMINATION_PER_TIER for count in tier_counts.values()),
                f"好/中/差完整分数 {tier_counts}；正式目标每档 {FORMAL_DISCRIMINATION_PER_TIER}",
            )
        )
    stages.append(
        _stage(
            ExperimentStage.DISCRIMINATION,
            discrimination_checks,
            next_command=(
                ".venv/bin/python scripts/analyze_validation.py "
                f"--input {_path_label(root, validation_path)} --output results/validation.json"
                if validation_path is not None
                else None
            ),
            limitations=[
                "现有术语 Pilot 只有 wrong/correct 两档，不能替代正式好/中/差三档判别力设计。",
                "wrong/correct 文本存在很强的长度线索；分析必须同时报告 length-only baseline。",
            ],
        )
    )

    stability_checks = [base_validation_check.model_copy(deep=True)]
    if validation is not None:
        complete_outputs = sum(
            sum(repeat.score is not None for repeat in row.repeats) >= FORMAL_STABILITY_REPEATS
            for row in validation.stability
        )
        stability_checks.append(
            _check(
                "FORMAL_STABILITY_SIZE",
                complete_outputs >= FORMAL_STABILITY_OUTPUTS,
                f"至少 {FORMAL_STABILITY_REPEATS} 次完整重复的输出 "
                f"{complete_outputs}/{FORMAL_STABILITY_OUTPUTS}",
            )
        )
    stages.append(
        _stage(
            ExperimentStage.STABILITY,
            stability_checks,
            next_command=(
                ".venv/bin/python scripts/analyze_validation.py "
                f"--input {_path_label(root, validation_path)} --output results/validation.json"
                if validation_path is not None
                else None
            ),
            limitations=[
                "现有分析器计算逐输出标准差和等级变化率，但尚未实现方案要求的 ICC(A,1) 与聚类 Bootstrap。"
            ],
        )
    )

    adversarial_checks = [
        base_validation_check.model_copy(deep=True),
        terminology_pair_check.model_copy(deep=True),
    ]
    if validation is not None:
        complete_pairs = sum(
            row.clean_score is not None
            and row.attacked_score is not None
            and row.attack_detected is not None
            for row in validation.adversarial
        )
        clean_controls = sum(row.clean_flagged is not None for row in validation.adversarial)
        adversarial_checks.extend(
            [
                _check(
                    "FORMAL_ADVERSARIAL_PAIRS",
                    complete_pairs >= FORMAL_ADVERSARIAL_PAIRS,
                    f"完整干净—攻击配对 {complete_pairs}/{FORMAL_ADVERSARIAL_PAIRS}",
                ),
                _check(
                    "FORMAL_CLEAN_CONTROLS",
                    clean_controls >= FORMAL_CLEAN_CONTROLS,
                    f"带误报判定的干净对照 {clean_controls}/{FORMAL_CLEAN_CONTROLS}",
                ),
            ]
        )
    stages.append(
        _stage(
            ExperimentStage.ADVERSARIAL,
            adversarial_checks,
            next_command=(
                ".venv/bin/python scripts/analyze_validation.py "
                f"--input {_path_label(root, validation_path)} --output results/validation.json"
                if validation_path is not None
                else None
            ),
            limitations=[
                "现有 runner 是术语、因果强度与条件错误的成对 Pilot，不是全文 Claim-Evidence 一致性。",
                "它不覆盖正式 12 类单因素攻击、严重度金标或 60 份独立干净负样本。",
                "TERM-060 在该单轮 pair task 中只能测安全措辞识别，不能测多轮坚持行为。",
            ],
        )
    )

    ablation_path = _input_path(root, ablation_input)
    ablation, ablation_schema, ablation_error = _load_optional_ablation_input(
        ablation_path
    )
    ablation_runtime = (root / "app/ablation.py").is_file() and (
        root / "scripts/run_pilot_ablation.py"
    ).is_file()
    retrieval_runtime = (root / "app/experiment_retrieval.py").is_file()
    arm_checks = [
        _check(
            "HY3_API_KEY_PRESENT",
            key_present,
            f"环境变量 {api_key_env} {'已设置' if key_present else '未设置'}；报告不读取或保存其值",
        ),
        _check(
            "ARM_A_DIRECT_HY3",
            ablation_runtime,
            "已实现实验专用 Hy3 direct/no-retrieval A 臂；不进入生产 ReviewRunner",
        ),
        _check(
            "ARM_B_SPARSE_TFIDF_VECTOR",
            ablation_runtime and retrieval_runtime,
            "已实现确定性稀疏 TF-IDF full-text vector B 臂；明确不是 dense embedding RAG",
        ),
        _check(
            "ARM_C_FROZEN_EVIDENCE_GRAPH",
            ablation_runtime and retrieval_runtime,
            "已实现仅由冻结文本/元数据构造的一跳 evidence-graph 重排/扩展；不读取专家标签",
        ),
        _check(
            "ARM_D_HY3_JUDGE_GATE",
            ablation_runtime,
            "已实现从精确 C artifact 出发、无追加检索的自动 Hy3 claim-evidence Judge gate",
        ),
        _check(
            "ABLATION_GRID_INPUT",
            ablation is not None,
            ablation_error
            or f"A/B/C/D 运行记录通过严格 Schema：{ablation_schema}",
            required=False,
        ),
    ]
    if ablation is not None:
        if ablation_schema in ABLATION_RUNTIME_SCHEMA_VERSIONS:
            from app.ablation import audit_pilot_ablation_grid

            grid = audit_pilot_ablation_grid(ablation)
            grid_complete = bool(grid["runtime_complete"])
        else:
            grid = ablation_grid_audit(ablation)
            grid_complete = bool(grid["grid_complete"])
        arm_checks.append(
            _check(
                "ABLATION_GRID_COMPLETE",
                grid_complete,
                f"运行网格 {grid['recorded_grid_cells']}/{grid['expected_grid_cells']}，"
                f"缺失 {grid['missing_grid_cells']}；input_schema={ablation_schema}",
                required=False,
            )
        )
    stages.append(
        _stage(
            ExperimentStage.ABLATION_ABCD,
            arm_checks,
            next_command=(
                ".venv/bin/python scripts/run_pilot_ablation.py "
                "--suite-id pilot-abcd-hy3-$(date -u +%Y%m%dT%H%M%SZ) "
                "--replicates 1 --judge-k 1"
            ),
            limitations=[
                "B 是稀疏 TF-IDF 向量基线，不等同于方案未来可能采用的 dense embedding RAG。",
                "C 是冻结文本/元数据 evidence graph，不等同于专家审核的条件化科学 Claim 图。",
                "D 是自动 Judge gate 和确定性渲染，不是人工复核或流畅重写。",
                "网格审计只做完整性和描述统计；正式推断统计尚需实现。",
            ],
        )
    )

    return ExperimentPreflightReport(
        repository=root.name,
        api_key_env=api_key_env,
        api_key_present=key_present,
        expert_reference=references,
        stages=stages,
        safety=PreflightSafety(),
    )
