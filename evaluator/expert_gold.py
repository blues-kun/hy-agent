"""Audit and load the project-owner-designated expert consensus gold files.

The source directory is named ``annotation_prelabel`` and several columns retain
historical ``ai_*`` names.  Renaming those columns in place would destroy the
original snapshot and could silently break existing consumers, so the gold
designation lives in ``annotation_prelabel/expert_gold_manifest.json``.  This
module verifies the manifest hashes and exposes the records without changing or
inventing any scientific label.

The snapshot contains one consolidated annotation per item.  It does *not*
contain separate rater-A/rater-B observations, so inter-expert agreement cannot
be reconstructed from these files.  The audit reports that limitation instead
of manufacturing paired ratings.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "annotation_prelabel" / "expert_gold_manifest.json"
EXPECTED_SCHEMA_VERSION = "mitoevidence.expert_consensus_gold.v1"
EXPECTED_DESIGNATION = "expert_consensus_gold"
EXPECTED_DATASET_NAMES = {
    "pilot_questions",
    "claim_reviews",
    "terminology_rules",
    "review_pool",
}
EXPECTED_TOTAL_RECORDS = 127


class ExpertGoldAuditError(ValueError):
    """Raised when a gold snapshot cannot be safely loaded."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_non_empty(value: Any) -> bool:
    # False and zero are meaningful labels/counts and therefore non-empty.
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertGoldAuditError(f"无法读取专家金标 manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExpertGoldAuditError("专家金标 manifest 顶层必须是 JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"无法读取 {path}: {exc}"]
    for lineno, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: 第 {lineno} 行 JSON 解析失败: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}: 第 {lineno} 行必须是 JSON object")
            continue
        rows.append(value)
    return rows, errors


def _distribution(values: list[Any]) -> dict[str, int]:
    def key(value: Any) -> str:
        if value is None:
            return "<null>"
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)

    return dict(sorted(Counter(key(value) for value in values).items()))


def _field_completeness(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = sorted({field for row in rows for field in row})
    return {
        field: {
            "present": sum(field in row for row in rows),
            "non_null": sum(field in row and row[field] is not None for row in rows),
            "non_empty": sum(field in row and _is_non_empty(row[field]) for row in rows),
        }
        for field in fields
    }


def _pilot_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [
        claim
        for row in rows
        for claim in row.get("required_claims", [])
        if isinstance(claim, dict)
    ]
    return {
        "answerability": _distribution([row.get("answerability") for row in rows]),
        "required_claims": len(claims),
        "required_claims_core_true": sum(claim.get("is_core") is True for claim in claims),
        "required_claims_core_false": sum(claim.get("is_core") is False for claim in claims),
        "required_claims_core_missing": sum("is_core" not in claim for claim in claims),
        "optional_claims": sum(
            len(value) for row in rows if isinstance((value := row.get("optional_claims")), list)
        ),
        "evidence_papers": sum(
            len(value) for row in rows if isinstance((value := row.get("evidence_papers")), list)
        ),
        "evidence_spans": sum(
            len(value) for row in rows if isinstance((value := row.get("evidence_spans")), list)
        ),
        "required_context_slots": sum(
            len(value)
            for row in rows
            if isinstance((value := row.get("required_context_slots")), list)
        ),
        "known_conflicts": sum(
            len(value) for row in rows if isinstance((value := row.get("known_conflicts")), list)
        ),
        "prohibited_inferences": sum(
            len(value)
            for row in rows
            if isinstance((value := row.get("prohibited_inferences")), list)
        ),
        "source_reviews": sum(
            len(value) for row in rows if isinstance((value := row.get("source_reviews")), list)
        ),
        "unresolved_note_count": sum(
            len(value)
            for row in rows
            if isinstance((value := row.get("needs_human_verification")), list)
        ),
    }


def _claim_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision": _distribution([row.get("ai_decision") for row in rows]),
        "confidence": _distribution([row.get("ai_confidence") for row in rows]),
        "usable_for_beta_cell_evidence": _distribution(
            [row.get("usable_for_beta_cell_evidence") for row in rows]
        ),
        "empty_recorded_conditions": [
            row.get("review_id")
            for row in rows
            if not _is_non_empty(row.get("recorded_conditions"))
        ],
        "empty_defect_codes": [
            row.get("review_id") for row in rows if not _is_non_empty(row.get("defect_codes"))
        ],
    }


def _terminology_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "category": _distribution([row.get("category") for row in rows]),
        "detector": _distribution([row.get("detector") for row in rows]),
        "confidence": _distribution([row.get("ai_confidence") for row in rows]),
        "explicit_approval_decision_present": any(
            "decision" in row or "ai_decision" in row for row in rows
        ),
        "missing_local_corpus_observation": [
            row.get("term_id")
            for row in rows
            if not _is_non_empty(row.get("observed_in_local_corpus"))
        ],
        "empty_unresolved_notes": [
            row.get("term_id")
            for row in rows
            if not _is_non_empty(row.get("needs_human_verification"))
        ],
    }


def _review_pool_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision": _distribution([row.get("ai_decision") for row in rows]),
        "fulltext_status": _distribution(
            [
                value.get("status") if isinstance((value := row.get("fulltext")), dict) else None
                for row in rows
            ]
        ),
        "bibliography_complete": {
            field: sum(
                isinstance(row.get("bibliography"), dict)
                and _is_non_empty(row["bibliography"].get(field))
                for row in rows
            )
            for field in ("pmid", "doi", "pmcid", "title", "year")
        },
        "fulltext_path_present": sum(
            isinstance(row.get("fulltext"), dict)
            and _is_non_empty(row["fulltext"].get("path"))
            for row in rows
        ),
        "fulltext_sha256_present": sum(
            isinstance(row.get("fulltext"), dict)
            and _is_non_empty(row["fulltext"].get("sha256"))
            for row in rows
        ),
        "total_reference_count": sum(
            value
            for row in rows
            if isinstance((value := row.get("reference_count")), int)
            and not isinstance(value, bool)
        ),
    }


def _semantic_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if name == "pilot_questions":
        return _pilot_summary(rows)
    if name == "claim_reviews":
        return _claim_summary(rows)
    if name == "terminology_rules":
        return _terminology_summary(rows)
    if name == "review_pool":
        return _review_pool_summary(rows)
    return {}


def _safe_dataset_path(repo_root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ExpertGoldAuditError("dataset.path 必须是非空相对路径")
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ExpertGoldAuditError(f"dataset.path 越出仓库根目录: {relative_path!r}") from exc
    return path


def audit_expert_gold(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a complete, JSON-serialisable audit of the designated snapshot.

    Missing/duplicate IDs, malformed JSON, count drift, hash drift and missing
    designated gold fields are errors.  Scientifically meaningful empty/null
    values are reported as completeness gaps but are never imputed.
    """

    manifest_path = Path(manifest_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT.resolve()
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append(
            "manifest.schema_version 不匹配: "
            f"{manifest.get('schema_version')!r} != {EXPECTED_SCHEMA_VERSION!r}"
        )
    if manifest.get("designation") != EXPECTED_DESIGNATION:
        errors.append(
            "manifest.designation 不匹配: "
            f"{manifest.get('designation')!r} != {EXPECTED_DESIGNATION!r}"
        )

    rater_structure = manifest.get("rater_structure")
    if not isinstance(rater_structure, dict):
        errors.append("manifest.rater_structure 必须是 object")
        rater_structure = {}
    if rater_structure.get("inter_expert_agreement_computable") is not False:
        errors.append("当前单份合并标注必须显式声明 inter_expert_agreement_computable=false")

    dataset_specs = manifest.get("datasets")
    if not isinstance(dataset_specs, list):
        raise ExpertGoldAuditError("manifest.datasets 必须是 array")

    dataset_reports: dict[str, dict[str, Any]] = {}
    total_records = 0
    seen_names: set[str] = set()
    for index, spec in enumerate(dataset_specs):
        if not isinstance(spec, dict):
            errors.append(f"manifest.datasets[{index}] 必须是 object")
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"manifest.datasets[{index}].name 必须是非空字符串")
            continue
        if name in seen_names:
            errors.append(f"dataset.name 重复: {name}")
            continue
        seen_names.add(name)
        try:
            path = _safe_dataset_path(root, spec.get("path"))
        except ExpertGoldAuditError as exc:
            errors.append(f"{name}: {exc}")
            continue

        rows, line_errors = _load_jsonl(path)
        errors.extend(line_errors)
        actual_hash: str | None
        try:
            actual_hash = sha256_file(path)
        except OSError as exc:
            actual_hash = None
            errors.append(f"{name}: 无法计算 SHA-256: {exc}")
        expected_hash = spec.get("sha256")
        if actual_hash != expected_hash:
            errors.append(f"{name}: SHA-256 漂移: {actual_hash!r} != {expected_hash!r}")

        expected_count = spec.get("record_count")
        if len(rows) != expected_count:
            errors.append(f"{name}: 记录数漂移: {len(rows)} != {expected_count!r}")

        id_field = spec.get("id_field")
        if not isinstance(id_field, str) or not id_field:
            errors.append(f"{name}: id_field 必须是非空字符串")
            id_field = "<missing-id-field>"
        missing_ids = [position for position, row in enumerate(rows, start=1) if not _is_non_empty(row.get(id_field))]
        if missing_ids:
            errors.append(f"{name}: {id_field} 缺失/为空的记录行: {missing_ids}")
        id_counts = Counter(str(row[id_field]) for row in rows if _is_non_empty(row.get(id_field)))
        duplicate_ids = sorted(item_id for item_id, count in id_counts.items() if count > 1)
        if duplicate_ids:
            errors.append(f"{name}: {id_field} 重复: {duplicate_ids}")

        gold_fields = spec.get("gold_fields")
        if not isinstance(gold_fields, list) or not gold_fields or not all(
            isinstance(field, str) and field for field in gold_fields
        ):
            errors.append(f"{name}: gold_fields 必须是非空字段名数组")
            gold_fields = []
        missing_gold_fields = {
            field: sum(field not in row for row in rows) for field in gold_fields
        }
        missing_gold_fields = {
            field: count for field, count in missing_gold_fields.items() if count
        }
        if missing_gold_fields:
            errors.append(f"{name}: 指定金标字段缺失: {missing_gold_fields}")

        annotators = _distribution([row.get("annotator") for row in rows])
        review_statuses = _distribution([row.get("review_status") for row in rows])
        report = {
            "path": spec.get("path"),
            "record_count": len(rows),
            "expected_record_count": expected_count,
            "sha256": actual_hash,
            "id_field": id_field,
            "unique_non_empty_ids": len(id_counts),
            "fields": _field_completeness(rows),
            "gold_fields": gold_fields,
            "legacy_provenance_values": {
                "annotator": annotators,
                "review_status": review_statuses,
            },
            "summary": _semantic_summary(name, rows),
        }
        dataset_reports[name] = report
        total_records += len(rows)

    if seen_names != EXPECTED_DATASET_NAMES:
        errors.append(
            "manifest 数据集集合不匹配: "
            f"实际={sorted(seen_names)}，预期={sorted(EXPECTED_DATASET_NAMES)}"
        )
    if total_records != EXPECTED_TOTAL_RECORDS:
        errors.append(f"专家金标总记录数不匹配: {total_records} != {EXPECTED_TOTAL_RECORDS}")

    pilot = dataset_reports.get("pilot_questions", {}).get("summary", {})
    if pilot.get("evidence_papers") == 0 and pilot.get("evidence_spans") == 0:
        warnings.append(
            "5 条 Pilot 只有题目级/主张级专家标签，没有 EvidencePaper 或 EvidenceSpan；"
            "不得补造证据锚点，也不能直接冒充完整 QuestionGold。"
        )
    terminology = dataset_reports.get("terminology_rules", {}).get("summary", {})
    if terminology.get("explicit_approval_decision_present") is False:
        warnings.append(
            "60 条术语记录提供 wrong/correct 规则对，但没有独立 approve/reject 标签；"
            "可把记录内容作为金标规则使用，不得虚构批准类别。"
        )
    warnings.append(
        "源文件的 annotator/review_status/ai_* 是历史字段；本审计按项目负责人确认和"
        "哈希 manifest 将原值作为专家共识金标读取，不覆写原始快照。"
    )
    warnings.append(
        "没有专家 A/B 的逐项独立标签，无法计算 Cohen's κ、加权 κ、Gwet's AC1/AC2 "
        "或 ICC；专家一致性实验应报告 unavailable，而不是以模型或同一标签副本代替第二位专家。"
    )

    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "designation": manifest.get("designation"),
        "manifest_path": str(manifest_path),
        "confirmed_at": manifest.get("confirmed_at"),
        "confirmation_source": manifest.get("confirmation_source"),
        "ok": not errors,
        "total_records": total_records,
        "inter_expert_agreement": {
            "computable": False,
            "reason": rater_structure.get("reason")
            or "No independent per-rater columns are present.",
        },
        "datasets": dataset_reports,
        "errors": errors,
        "warnings": warnings,
    }


def load_expert_gold_records(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load the exact snapshot after hash/count/schema verification.

    Returned rows retain all source field names and values.  In particular,
    ``ai_decision`` is not renamed or reinterpreted beyond the manifest's gold
    designation, and null/empty values remain null/empty.
    """

    audit = audit_expert_gold(manifest_path, repo_root=repo_root)
    if not audit["ok"]:
        raise ExpertGoldAuditError("专家金标审计失败: " + "；".join(audit["errors"]))
    manifest = _load_manifest(Path(manifest_path).resolve())
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT.resolve()
    loaded: dict[str, list[dict[str, Any]]] = {}
    for spec_value in manifest["datasets"]:
        spec = dict(spec_value)
        path = _safe_dataset_path(root, spec["path"])
        rows, errors = _load_jsonl(path)
        if errors:  # Defensive: the preceding audit should already have caught these.
            raise ExpertGoldAuditError("；".join(errors))
        loaded[str(spec["name"])] = rows
    return loaded


def selected_gold_fields(
    dataset_name: str,
    row: Mapping[str, Any],
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Return only fields explicitly designated as gold, preserving values.

    This helper intentionally does not derive a common label across the four
    heterogeneous tasks.  A missing designated field is an error, not a value
    to impute.
    """

    manifest = _load_manifest(Path(manifest_path).resolve())
    specs = {
        spec.get("name"): spec for spec in manifest.get("datasets", []) if isinstance(spec, dict)
    }
    if dataset_name not in specs:
        raise ExpertGoldAuditError(f"未知专家金标数据集: {dataset_name!r}")
    fields = specs[dataset_name].get("gold_fields")
    if not isinstance(fields, list):
        raise ExpertGoldAuditError(f"{dataset_name}: manifest.gold_fields 非法")
    missing = [field for field in fields if field not in row]
    if missing:
        raise ExpertGoldAuditError(f"{dataset_name}: 记录缺少指定金标字段: {missing}")
    return {field: row[field] for field in fields}
