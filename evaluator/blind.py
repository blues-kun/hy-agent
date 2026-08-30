"""Archived/optional workflow for a future two-rater reliability study.

This module deliberately separates three things that are easy to conflate:

* ``annotation_prelabel`` is the project-owner-designated consolidated expert gold
  snapshot, despite its legacy directory and ``ai_*`` field names;
* a *neutral packet* contains only an explicit source-fact allowlist;
* newly locked expert files would be observations from a separate A/B study.

The current gold snapshot has no independent A/B columns, so this module cannot
recover historical inter-expert agreement.  It may still copy source facts,
calculate hashes, compare *new* labels and report disagreements if a future
study is explicitly run.  It never copies a gold decision into a neutral packet
and never chooses an adjudication decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0.0"

ITEM_TYPES = ("pilot_question", "claim", "terminology", "review_pool")
DECISIONS: dict[str, tuple[str, ...]] = {
    "pilot_question": (
        "answerable",
        "partial",
        "insufficient",
        "out_of_scope",
        "uncertain",
    ),
    "claim": ("accept", "accept_with_edits", "reject", "uncertain"),
    "terminology": (
        "approve_rule",
        "revise_rule",
        "reject_rule",
        "uncertain",
    ),
    "review_pool": (
        "include_seed",
        "include_context_only",
        "exclude",
        "uncertain",
    ),
}
CONFIDENCE_VALUES = ("low", "medium", "high")

PILOT_ALLOWLIST = ("question_id", "question")
CLAIM_ALLOWLIST = (
    "review_id",
    "statement_id",
    "paper_id",
    "paper_short",
    "section",
    "triple",
    "evidence_text",
)
TERMINOLOGY_CONTEXT_ALLOWLIST = (
    "source_text",
    "source_context",
    "source_locator",
    "source_id",
)

# These are designated source-gold decisions, suggestions or workflow labels.
# Their presence in a neutral packet for a new reliability study is always a
# label-leakage bug. ``annotator_code`` is intentionally not forbidden because
# it belongs only to a separate new review file.
FORBIDDEN_NEUTRAL_KEYS = {
    "annotator",
    "review_status",
    "needs_human_verification",
    "suggested_edits",
    "required_claims",
    "optional_claims",
    "required_context_slots",
    "prohibited_inferences",
    "known_conflicts",
    "wrong",
    "correct",
    "why",
    "example_wrong",
    "example_correct",
    "caveats",
    "recommended_uses",
    "required_human_checks",
    "pool_role",
    "detector",
}


class BlindWorkflowError(ValueError):
    """Raised when a package cannot be constructed without breaking blinding."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_or_none(path: Path, errors: list[str], *, label: str) -> str | None:
    try:
        return sha256_file(path)
    except OSError as exc:
        errors.append(f"{label}: 无法计算文件 SHA-256（{exc}）")
        return None


def source_item_sha256(source: Mapping[str, Any]) -> str:
    """Hash the neutral source facts for one item, independent of formatting."""
    return sha256_bytes(_canonical_json(source))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BlindWorkflowError(f"无法读取 JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BlindWorkflowError(f"无法读取 JSONL {path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise BlindWorkflowError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc
        if not isinstance(value, dict):
            raise BlindWorkflowError(f"{path}:{lineno} 必须是 JSON object")
        records.append(value)
    return records


def _project(record: Mapping[str, Any], allowlist: Iterable[str]) -> dict[str, Any]:
    return {key: record[key] for key in allowlist if key in record}


def _require_nonempty(record: Mapping[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = [key for key in keys if record.get(key) in (None, "")]
    if missing:
        raise BlindWorkflowError(f"{where} 缺少必要来源字段: {missing}")


def _forbidden_neutral_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.startswith("ai_") or key_text in FORBIDDEN_NEUTRAL_KEYS:
                found.append(child_path)
            found.extend(_forbidden_neutral_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_neutral_paths(child, f"{path}[{index}]"))
    return found


def _make_item(item_type: str, item_id: str, source: dict[str, Any]) -> dict[str, Any]:
    if item_type not in ITEM_TYPES:
        raise BlindWorkflowError(f"未知 item_type: {item_type}")
    if not item_id:
        raise BlindWorkflowError(f"{item_type} 的 item_id 为空")
    forbidden = _forbidden_neutral_paths(source)
    if forbidden:
        raise BlindWorkflowError(f"中性来源字段泄露 AI/工作流字段: {forbidden}")
    return {
        "item_type": item_type,
        "item_id": str(item_id),
        "source_item_sha256": source_item_sha256(source),
        "source": source,
    }


def _pilot_items(annotation_root: Path) -> list[dict[str, Any]]:
    path = annotation_root / "pilot_questions" / "pilot_5_questions.jsonl"
    items: list[dict[str, Any]] = []
    for record in _load_jsonl(path):
        _require_nonempty(record, PILOT_ALLOWLIST, where=f"Pilot {record.get('question_id', '?')}")
        source = _project(record, PILOT_ALLOWLIST)
        items.append(_make_item("pilot_question", str(source["question_id"]), source))
    return items


def _claim_items(annotation_root: Path) -> list[dict[str, Any]]:
    path = annotation_root / "claim_review_sample" / "claim_review_sample.jsonl"
    items: list[dict[str, Any]] = []
    required = ("review_id", "statement_id", "paper_id", "section", "triple", "evidence_text")
    for record in _load_jsonl(path):
        _require_nonempty(record, required, where=f"Claim {record.get('review_id', '?')}")
        source = _project(record, CLAIM_ALLOWLIST)
        items.append(_make_item("claim", str(source["review_id"]), source))
    return items


def _terminology_items(annotation_root: Path) -> list[dict[str, Any]]:
    path = annotation_root / "terminology_blacklist" / "terminology_blacklist.jsonl"
    items: list[dict[str, Any]] = []
    for record in _load_jsonl(path):
        term_id = record.get("term_id")
        if not term_id:
            raise BlindWorkflowError("Terminology 记录缺少 term_id")
        source: dict[str, Any] = {"term_id": str(term_id)}
        for key in TERMINOLOGY_CONTEXT_ALLOWLIST:
            if record.get(key) not in (None, "", [], {}):
                source[key] = record[key]
        source["source_context_missing"] = not any(
            key in source for key in TERMINOLOGY_CONTEXT_ALLOWLIST
        )
        items.append(_make_item("terminology", str(term_id), source))
    return items


def _review_pool_items(evidence_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(evidence_manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("reviews"), list):
        raise BlindWorkflowError("evidence_pool_manifest.json 缺少 reviews 列表")
    fulltext_rows = manifest.get("fulltext", [])
    if not isinstance(fulltext_rows, list):
        raise BlindWorkflowError("evidence_pool_manifest.json 的 fulltext 必须是列表")
    fulltext_by_pmid = {
        str(row.get("pmid")): row
        for row in fulltext_rows
        if isinstance(row, dict) and row.get("pmid")
    }

    items: list[dict[str, Any]] = []
    for ordinal, review in enumerate(manifest["reviews"], start=1):
        if not isinstance(review, dict):
            raise BlindWorkflowError(f"reviews[{ordinal - 1}] 不是 object")
        index = review.get("index", ordinal)
        pmid = review.get("pmid")
        _require_nonempty(review, ("pmid", "title", "year"), where=f"Review pool index={index}")
        bibliography = _project(
            review,
            ("pmid", "doi", "pmcid", "title", "year", "is_open_access"),
        )
        reference_retrieval = _project(
            review,
            ("epmc_hit_count", "refs_source", "refs_used"),
        )
        fulltext = fulltext_by_pmid.get(str(pmid), {})
        source = {
            "source_manifest_index": index,
            "bibliography": bibliography,
            "reference_retrieval": reference_retrieval,
            "fulltext": _project(
                fulltext,
                ("pmid", "pmcid", "is_open_access", "path", "bytes", "sha256", "error"),
            ),
        }
        item_id = f"REVPOOL-{int(index):03d}" if str(index).isdigit() else f"REVPOOL-{index}"
        items.append(_make_item("review_pool", item_id, source))
    return items


def collect_neutral_items(
    annotation_root: str | Path,
    evidence_manifest_path: str | Path,
) -> list[dict[str, Any]]:
    """Project AI-prelabel files through the README's explicit neutral allowlist."""
    annotation_root = Path(annotation_root)
    readme = annotation_root / "README.md"
    if not readme.is_file():
        raise BlindWorkflowError(f"缺少白名单规范: {readme}")
    items = (
        _pilot_items(annotation_root)
        + _claim_items(annotation_root)
        + _terminology_items(annotation_root)
        + _review_pool_items(Path(evidence_manifest_path))
    )
    keys = [(item["item_type"], item["item_id"]) for item in items]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise BlindWorkflowError(f"中性记录 ID 重复: {duplicates}")
    forbidden = _forbidden_neutral_paths(items)
    if forbidden:
        raise BlindWorkflowError(f"中性 packet 含禁用字段: {forbidden}")
    return items


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _blank_review(
    *,
    slot: str,
    batch_id: str,
    guideline_version: str,
    source_snapshot_sha256: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": "independent_blind_annotation",
        "annotator_slot": slot,
        "annotator_code": None,
        "batch_id": batch_id,
        "guideline_version": guideline_version,
        "source_snapshot_sha256": source_snapshot_sha256,
        "started_at_utc": None,
        "completed_at_utc": None,
        "blindness_attestation": {
            "neutral_packet_only_confirmed": None,
            "ai_prelabels_not_viewed": None,
            "other_expert_labels_not_viewed": None,
            "independent_work_confirmed": None,
        },
        "allowed_decisions": {key: list(values) for key, values in DECISIONS.items()},
        "records": [
            {
                "item_type": item["item_type"],
                "item_id": item["item_id"],
                "source_item_sha256": item["source_item_sha256"],
                "decision": None,
                "confidence": None,
                "rationale": None,
                "structured_fields": {},
            }
            for item in items
        ],
        "lock": {
            "locked": False,
            "locked_at_utc": None,
            "signature_or_audit_id": None,
            "amendment_of": None,
        },
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def build_blind_package(
    *,
    annotation_root: str | Path,
    evidence_manifest_path: str | Path,
    output_dir: str | Path,
    batch_id: str,
    guideline_version: str,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create an immutable neutral packet plus empty A/B review forms.

    ``output_dir`` must not already exist.  Requiring a fresh directory prevents
    an older human assignment from being silently overwritten.
    """
    if not batch_id.strip() or not guideline_version.strip():
        raise BlindWorkflowError("batch_id 与 guideline_version 均不能为空")
    annotation_root = Path(annotation_root).resolve()
    evidence_manifest_path = Path(evidence_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise BlindWorkflowError(f"输出目录已存在，拒绝覆盖冻结批次: {output_dir}")

    input_paths = {
        "allowlist_readme": annotation_root / "README.md",
        "pilot_prelabel": annotation_root / "pilot_questions" / "pilot_5_questions.jsonl",
        "claim_prelabel": annotation_root / "claim_review_sample" / "claim_review_sample.jsonl",
        "terminology_prelabel": annotation_root
        / "terminology_blacklist"
        / "terminology_blacklist.jsonl",
        "evidence_pool_manifest": evidence_manifest_path,
    }
    # Manifest 中使用逻辑路径，避免盲标包泄露构建机器的用户名与挂载点，
    # 同时让同一输入快照在不同主机上得到相同的 snapshot hash。
    input_labels = {
        "allowlist_readme": "annotation_prelabel/README.md",
        "pilot_prelabel": "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl",
        "claim_prelabel": "annotation_prelabel/claim_review_sample/claim_review_sample.jsonl",
        "terminology_prelabel": (
            "annotation_prelabel/terminology_blacklist/terminology_blacklist.jsonl"
        ),
        "evidence_pool_manifest": "eval/data/evidence_pool_manifest.json",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise BlindWorkflowError(f"缺少输入文件: {missing}")

    items = collect_neutral_items(annotation_root, evidence_manifest_path)
    counts = dict(Counter(item["item_type"] for item in items))
    neutral_payload = _jsonl_bytes(items)
    neutral_hash = sha256_bytes(neutral_payload)
    review_a = _blank_review(
        slot="EXPERT_A",
        batch_id=batch_id,
        guideline_version=guideline_version,
        source_snapshot_sha256=neutral_hash,
        items=items,
    )
    review_b = _blank_review(
        slot="EXPERT_B",
        batch_id=batch_id,
        guideline_version=guideline_version,
        source_snapshot_sha256=neutral_hash,
        items=items,
    )
    a_payload = json.dumps(review_a, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    b_payload = json.dumps(review_b, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"

    input_meta: dict[str, Any] = {}
    for name, path in input_paths.items():
        count: int | None = None
        if path.suffix == ".jsonl":
            count = len(_load_jsonl(path))
        elif name == "evidence_pool_manifest":
            manifest_value = _load_json(path)
            count = len(manifest_value.get("reviews", [])) if isinstance(manifest_value, dict) else None
        input_meta[name] = {
            "path": input_labels[name],
            "sha256": sha256_file(path),
            "record_count": count,
        }
    input_snapshot_hash = sha256_bytes(_canonical_json(input_meta))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "neutral_blind_packet_build",
        "generated_at_utc": generated_at_utc or _utc_now(),
        "batch_id": batch_id,
        "guideline_version": guideline_version,
        "blinding_policy": {
            "source_allowlist": {
                "pilot_question": list(PILOT_ALLOWLIST),
                "claim": list(CLAIM_ALLOWLIST),
                "terminology": ["term_id", *TERMINOLOGY_CONTEXT_ALLOWLIST, "source_context_missing"],
                "review_pool": "facts_from_eval/data/evidence_pool_manifest.json_only",
            },
            "source_records_designated_expert_consensus_gold": True,
            "source_gold_labels_included_in_neutral_packet": False,
            "new_study_adjudication_required_for_disagreements": True,
        },
        "inputs": input_meta,
        "input_snapshot_sha256": input_snapshot_hash,
        "counts": {"total": len(items), **{key: counts.get(key, 0) for key in ITEM_TYPES}},
        "outputs": {
            "neutral_items.jsonl": {
                "sha256": neutral_hash,
                "record_count": len(items),
            },
            "expert_A.json": {"sha256": sha256_bytes(a_payload), "record_count": len(items)},
            "expert_B.json": {"sha256": sha256_bytes(b_payload), "record_count": len(items)},
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        _write_bytes(temp_dir / "neutral_items.jsonl", neutral_payload)
        _write_bytes(temp_dir / "expert_A.json", a_payload)
        _write_bytes(temp_dir / "expert_B.json", b_payload)
        _write_bytes(
            temp_dir / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def load_neutral_packet(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        items = _load_jsonl(Path(path))
    except BlindWorkflowError as exc:
        return [], [str(exc)]
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        item_type = item.get("item_type")
        item_id = item.get("item_id")
        source = item.get("source")
        where = f"neutral_items[{index}]"
        if item_type not in ITEM_TYPES or not isinstance(item_id, str) or not item_id:
            errors.append(f"{where}: item_type/item_id 非法")
            continue
        key = (item_type, item_id)
        if key in seen:
            errors.append(f"{where}: 重复记录 {key}")
        seen.add(key)
        if not isinstance(source, dict):
            errors.append(f"{where}: source 必须是 object")
            continue
        expected = source_item_sha256(source)
        if item.get("source_item_sha256") != expected:
            errors.append(f"{where}: source_item_sha256 不匹配")
        forbidden = _forbidden_neutral_paths(source)
        if forbidden:
            errors.append(f"{where}: 中性来源泄露禁用字段 {forbidden}")
    return items, errors


def _read_review(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    try:
        value = _load_json(Path(path))
    except BlindWorkflowError as exc:
        return {}, [str(exc)]
    if not isinstance(value, dict):
        return {}, [f"{path}: 顶层必须是 JSON object"]
    return value, []


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _valid_utc(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _review_record_map(
    review: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    errors: list[str] = []
    records = review.get("records")
    if not isinstance(records, list):
        return {}, [f"{label}: records 必须是列表"]
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        where = f"{label}.records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{where}: 必须是 object")
            continue
        item_type, item_id = record.get("item_type"), record.get("item_id")
        if item_type not in ITEM_TYPES or not isinstance(item_id, str) or not item_id:
            errors.append(f"{where}: item_type/item_id 非法")
            continue
        key = (item_type, item_id)
        if key in mapped:
            errors.append(f"{where}: 重复记录 {key}")
            continue
        mapped[key] = record
        decision = record.get("decision")
        if decision not in DECISIONS[item_type]:
            errors.append(
                f"{where}: decision={decision!r} 非法；允许 {list(DECISIONS[item_type])}"
            )
        confidence = record.get("confidence")
        if confidence is not None and confidence not in CONFIDENCE_VALUES:
            errors.append(f"{where}: confidence={confidence!r} 非法")
        if not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
            errors.append(f"{where}: 锁定记录必须提供非空 rationale")
        if not isinstance(record.get("structured_fields", {}), dict):
            errors.append(f"{where}: structured_fields 必须是 object")
        if not _is_sha256(record.get("source_item_sha256")):
            errors.append(f"{where}: source_item_sha256 非法")
    return mapped, errors


def _validate_review_metadata(review: Mapping[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    if review.get("workflow") != "independent_blind_annotation":
        errors.append(f"{label}: workflow 必须为 independent_blind_annotation")
    for key in ("annotator_code", "batch_id", "guideline_version"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            errors.append(f"{label}: {key} 不能为空")
    if not _is_sha256(review.get("source_snapshot_sha256")):
        errors.append(f"{label}: source_snapshot_sha256 非法")
    if not _valid_utc(review.get("started_at_utc")):
        errors.append(f"{label}: started_at_utc 必须是带 UTC 时区的 ISO 时间")
    if not _valid_utc(review.get("completed_at_utc")):
        errors.append(f"{label}: completed_at_utc 必须是带 UTC 时区的 ISO 时间")
    attestation = review.get("blindness_attestation")
    required_attestations = (
        "neutral_packet_only_confirmed",
        "ai_prelabels_not_viewed",
        "other_expert_labels_not_viewed",
        "independent_work_confirmed",
    )
    if not isinstance(attestation, dict):
        errors.append(f"{label}: blindness_attestation 必须是 object")
    else:
        for key in required_attestations:
            if attestation.get(key) is not True:
                errors.append(f"{label}: 盲态声明 {key} 必须显式为 true")
    lock = review.get("lock")
    if not isinstance(lock, dict):
        errors.append(f"{label}: lock 必须是 object")
    else:
        if lock.get("locked") is not True:
            errors.append(f"{label}: lock.locked 必须为 true")
        if not _valid_utc(lock.get("locked_at_utc")):
            errors.append(f"{label}: lock.locked_at_utc 非法")
        signature = lock.get("signature_or_audit_id")
        if not isinstance(signature, str) or not signature.strip():
            errors.append(f"{label}: lock.signature_or_audit_id 不能为空")
    return errors


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> tuple[float | None, str | None]:
    if not labels_a:
        return None, "no_records"
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    categories = set(count_a) | set(count_b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None, "marginal_degeneracy_expected_agreement_is_one"
    return (observed - expected) / denominator, None


def _agreement_statistics(
    map_a: Mapping[tuple[str, str], Mapping[str, Any]],
    map_b: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    common = set(map_a) & set(map_b)
    for item_type in ITEM_TYPES:
        keys = sorted(key for key in common if key[0] == item_type)
        labels_a = [str(map_a[key].get("decision")) for key in keys]
        labels_b = [str(map_b[key].get("decision")) for key in keys]
        raw = (
            sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(keys)
            if keys
            else None
        )
        kappa, note = _cohen_kappa(labels_a, labels_b)
        statistics[item_type] = {
            "n": len(keys),
            "n_agree": sum(a == b for a, b in zip(labels_a, labels_b, strict=True)),
            "raw_agreement": raw,
            "cohen_kappa": kappa,
            "kappa_note": note,
            "expert_a_marginals": dict(Counter(labels_a)),
            "expert_b_marginals": dict(Counter(labels_b)),
        }
    return statistics


def _manifest_checks(
    manifest_path: Path,
    neutral_path: Path,
    *,
    batch_id: Any,
    guideline_version: Any,
) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _load_json(manifest_path)
    except BlindWorkflowError as exc:
        return [str(exc)]
    if not isinstance(manifest, dict):
        return ["package manifest 顶层必须是 object"]
    if manifest.get("batch_id") != batch_id:
        errors.append("package manifest 的 batch_id 与专家文件不一致")
    if manifest.get("guideline_version") != guideline_version:
        errors.append("package manifest 的 guideline_version 与专家文件不一致")
    listed = manifest.get("outputs", {}).get("neutral_items.jsonl", {}).get("sha256")
    actual = sha256_file(neutral_path)
    if listed != actual:
        errors.append("package manifest 中 neutral_items.jsonl 的 SHA-256 与文件不一致")
    return errors


def validate_blind_pair(
    expert_a_path: str | Path,
    expert_b_path: str | Path,
    *,
    neutral_packet_path: str | Path | None = None,
    package_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate two locked files and mechanically compare their decisions."""
    expert_a_path, expert_b_path = Path(expert_a_path), Path(expert_b_path)
    review_a, errors_a = _read_review(expert_a_path)
    review_b, errors_b = _read_review(expert_b_path)
    errors = errors_a + errors_b
    warnings: list[str] = []
    if review_a:
        errors.extend(_validate_review_metadata(review_a, label="EXPERT_A"))
    if review_b:
        errors.extend(_validate_review_metadata(review_b, label="EXPERT_B"))
    map_a, record_errors_a = _review_record_map(review_a, label="EXPERT_A") if review_a else ({}, [])
    map_b, record_errors_b = _review_record_map(review_b, label="EXPERT_B") if review_b else ({}, [])
    errors.extend(record_errors_a + record_errors_b)

    for field in ("source_snapshot_sha256", "batch_id", "guideline_version"):
        if review_a.get(field) != review_b.get(field):
            errors.append(f"A/B 的 {field} 不一致")
    if review_a.get("annotator_code") == review_b.get("annotator_code"):
        errors.append("A/B 必须使用不同 annotator_code")
    if review_a.get("annotator_slot") == review_b.get("annotator_slot"):
        errors.append("A/B 的 annotator_slot 必须不同")
    keys_a, keys_b = set(map_a), set(map_b)
    if keys_a != keys_b:
        errors.append(
            "A/B 记录集合不一致："
            f"仅A={sorted(keys_a - keys_b)}；仅B={sorted(keys_b - keys_a)}"
        )

    neutral_items: list[dict[str, Any]] = []
    if neutral_packet_path is not None:
        neutral_path = Path(neutral_packet_path)
        neutral_items, neutral_errors = load_neutral_packet(neutral_path)
        errors.extend(neutral_errors)
        actual_neutral_hash = sha256_file(neutral_path)
        for label, review in (("EXPERT_A", review_a), ("EXPERT_B", review_b)):
            if review.get("source_snapshot_sha256") != actual_neutral_hash:
                errors.append(f"{label}: source_snapshot_sha256 与 neutral packet 不一致")
        neutral_map = {
            (item["item_type"], item["item_id"]): item for item in neutral_items
        }
        if keys_a != set(neutral_map):
            errors.append(
                "专家记录与 neutral packet 不完整对应："
                f"缺失={sorted(set(neutral_map) - keys_a)}；多余={sorted(keys_a - set(neutral_map))}"
            )
        for label, record_map in (("EXPERT_A", map_a), ("EXPERT_B", map_b)):
            for key in set(record_map) & set(neutral_map):
                if record_map[key].get("source_item_sha256") != neutral_map[key].get(
                    "source_item_sha256"
                ):
                    errors.append(f"{label}: {key} 的 source_item_sha256 不匹配")
        if package_manifest_path is not None:
            errors.extend(
                _manifest_checks(
                    Path(package_manifest_path),
                    neutral_path,
                    batch_id=review_a.get("batch_id"),
                    guideline_version=review_a.get("guideline_version"),
                )
            )
    elif package_manifest_path is not None:
        errors.append("给出 package_manifest_path 时必须同时给出 neutral_packet_path")

    disagreements: list[dict[str, Any]] = []
    for key in sorted(keys_a & keys_b):
        a_record, b_record = map_a[key], map_b[key]
        fields: list[str] = []
        if a_record.get("decision") != b_record.get("decision"):
            fields.append("decision")
        if _canonical_json(a_record.get("structured_fields", {})) != _canonical_json(
            b_record.get("structured_fields", {})
        ):
            fields.append("structured_fields")
        if fields:
            disagreements.append(
                {
                    "item_type": key[0],
                    "item_id": key[1],
                    "disagreement_fields": fields,
                    "expert_a_decision": a_record.get("decision"),
                    "expert_b_decision": b_record.get("decision"),
                    "expert_a_structured_fields": a_record.get("structured_fields", {}),
                    "expert_b_structured_fields": b_record.get("structured_fields", {}),
                    "source_item_sha256": a_record.get("source_item_sha256"),
                }
            )

    statistics = _agreement_statistics(map_a, map_b)
    for item_type, stats in statistics.items():
        if stats["n"] and stats["cohen_kappa"] is None:
            warnings.append(
                f"{item_type}: Cohen kappa 因边际退化不可定义；同时报告原始一致率"
            )
    hash_a = _sha256_or_none(expert_a_path, errors, label="EXPERT_A")
    hash_b = _sha256_or_none(expert_b_path, errors, label="EXPERT_B")
    report = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "blind_pair_validation",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "batch_id": review_a.get("batch_id"),
        "guideline_version": review_a.get("guideline_version"),
        "source_snapshot_sha256": review_a.get("source_snapshot_sha256"),
        "files": {
            "expert_a": {"path": str(expert_a_path), "sha256": hash_a},
            "expert_b": {"path": str(expert_b_path), "sha256": hash_b},
        },
        "counts": {
            "expert_a": len(map_a),
            "expert_b": len(map_b),
            "neutral_packet": len(neutral_items) if neutral_packet_path is not None else None,
            "disagreements": len(disagreements),
        },
        "agreement_by_item_type": statistics,
        "disagreements": disagreements,
        "notice": "机械比较结果不是裁决；所有分歧必须由第三位专家处理。",
    }
    return report


def make_adjudication_template(pair_report: Mapping[str, Any]) -> dict[str, Any]:
    """Create a blank adjudication form; final decisions are always left null."""
    if not pair_report.get("ok"):
        raise BlindWorkflowError("A/B 校验未通过，不能建立裁决表")
    files = pair_report.get("files", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": "third_expert_adjudication",
        "adjudicator_code": None,
        "batch_id": pair_report.get("batch_id"),
        "guideline_version": pair_report.get("guideline_version"),
        "source_snapshot_sha256": pair_report.get("source_snapshot_sha256"),
        "expert_a_locked_file_sha256": files.get("expert_a", {}).get("sha256"),
        "expert_b_locked_file_sha256": files.get("expert_b", {}).get("sha256"),
        "started_at_utc": None,
        "completed_at_utc": None,
        "records": [
            {
                "item_type": row["item_type"],
                "item_id": row["item_id"],
                "source_item_sha256": row["source_item_sha256"],
                "expert_a_decision": row["expert_a_decision"],
                "expert_b_decision": row["expert_b_decision"],
                "disagreement_fields": row["disagreement_fields"],
                "final_decision": None,
                "final_structured_fields": {},
                "rationale": None,
                "unresolved_reason": None,
                "ai_prelabel_consulted": False,
            }
            for row in pair_report.get("disagreements", [])
        ],
        "lock": {
            "locked": False,
            "locked_at_utc": None,
            "signature_or_audit_id": None,
        },
        "notice": "本表未自动填写任何最终决策。",
    }


def validate_adjudication(
    adjudication_path: str | Path,
    expert_a_path: str | Path,
    expert_b_path: str | Path,
    *,
    neutral_packet_path: str | Path | None = None,
    package_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a locked third-expert record without filling or changing it."""
    pair = validate_blind_pair(
        expert_a_path,
        expert_b_path,
        neutral_packet_path=neutral_packet_path,
        package_manifest_path=package_manifest_path,
    )
    adjudication, load_errors = _read_review(adjudication_path)
    errors = list(pair["errors"]) + load_errors
    if not adjudication:
        return {
            "schema_version": SCHEMA_VERSION,
            "workflow": "adjudication_validation",
            "ok": False,
            "errors": errors,
            "warnings": [],
        }
    if adjudication.get("workflow") != "third_expert_adjudication":
        errors.append("裁决文件 workflow 必须为 third_expert_adjudication")
    for field in ("batch_id", "guideline_version", "source_snapshot_sha256"):
        if adjudication.get(field) != pair.get(field):
            errors.append(f"裁决文件 {field} 与 A/B 校验结果不一致")
    if adjudication.get("expert_a_locked_file_sha256") != pair["files"]["expert_a"]["sha256"]:
        errors.append("裁决文件 expert_a_locked_file_sha256 与实际 A 文件不一致")
    if adjudication.get("expert_b_locked_file_sha256") != pair["files"]["expert_b"]["sha256"]:
        errors.append("裁决文件 expert_b_locked_file_sha256 与实际 B 文件不一致")
    adjudicator = adjudication.get("adjudicator_code")
    if not isinstance(adjudicator, str) or not adjudicator.strip():
        errors.append("adjudicator_code 不能为空")
    else:
        review_a, _ = _read_review(expert_a_path)
        review_b, _ = _read_review(expert_b_path)
        if adjudicator in {review_a.get("annotator_code"), review_b.get("annotator_code")}:
            errors.append("第三裁决者必须与专家 A/B 使用不同代码")
    if not _valid_utc(adjudication.get("started_at_utc")):
        errors.append("裁决 started_at_utc 非法")
    if not _valid_utc(adjudication.get("completed_at_utc")):
        errors.append("裁决 completed_at_utc 非法")
    lock = adjudication.get("lock")
    if not isinstance(lock, dict) or lock.get("locked") is not True:
        errors.append("裁决文件必须 lock.locked=true")
    else:
        if not _valid_utc(lock.get("locked_at_utc")):
            errors.append("裁决 lock.locked_at_utc 非法")
        if not isinstance(lock.get("signature_or_audit_id"), str) or not lock[
            "signature_or_audit_id"
        ].strip():
            errors.append("裁决 lock.signature_or_audit_id 不能为空")

    expected = {
        (row["item_type"], row["item_id"]): row for row in pair.get("disagreements", [])
    }
    records = adjudication.get("records")
    actual: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(records, list):
        errors.append("裁决 records 必须是列表")
        records = []
    for index, record in enumerate(records):
        where = f"adjudication.records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{where}: 必须是 object")
            continue
        key = (record.get("item_type"), record.get("item_id"))
        if key in actual:
            errors.append(f"{where}: 重复记录 {key}")
            continue
        actual[key] = record
    if set(actual) != set(expected):
        errors.append(
            "裁决记录必须恰好覆盖全部分歧："
            f"缺失={sorted(set(expected) - set(actual))}；多余={sorted(set(actual) - set(expected))}"
        )
    for key in sorted(set(actual) & set(expected)):
        record, disagreement = actual[key], expected[key]
        where = f"adjudication[{key[0]}:{key[1]}]"
        for field in (
            "source_item_sha256",
            "expert_a_decision",
            "expert_b_decision",
            "disagreement_fields",
        ):
            if record.get(field) != disagreement.get(field):
                errors.append(f"{where}: {field} 与机械分歧报告不一致")
        final_decision = record.get("final_decision")
        allowed = set(DECISIONS[key[0]]) | {"unresolved"}
        if final_decision not in allowed:
            errors.append(f"{where}: final_decision={final_decision!r} 非法或尚未填写")
        if not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
            errors.append(f"{where}: 必须填写非空 rationale")
        if final_decision == "unresolved" and (
            not isinstance(record.get("unresolved_reason"), str)
            or not record["unresolved_reason"].strip()
        ):
            errors.append(f"{where}: unresolved 必须填写 unresolved_reason")

    adjudication_hash = _sha256_or_none(Path(adjudication_path), errors, label="ADJUDICATION")
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": "adjudication_validation",
        "ok": not errors,
        "errors": errors,
        "warnings": pair.get("warnings", []),
        "counts": {
            "expected_disagreements": len(expected),
            "adjudication_records": len(actual),
        },
        "files": {
            **pair.get("files", {}),
            "adjudication": {
                "path": str(adjudication_path),
                "sha256": adjudication_hash,
            },
        },
        "notice": "该校验只核验覆盖、哈希和合法性，不替代第三位专家的科学裁决。",
    }
