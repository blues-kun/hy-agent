"""Filesystem-level audit for Pilot A/B/C/D runtime artifacts.

This module is intentionally separate from the suite-state grid audit.  It
opens every recorded cell, verifies its manifest and files, parses the strict
runtime schemas, and then checks the cross-arm invariants that cannot be
established from ``suite_state.json`` alone.  It never repairs or rewrites an
experiment artifact.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from app.ablation import (
    ABLATION_FORMAL_STATUS_BY_SCHEMA_VERSION,
    ARM_DEFINITION_BY_ID,
    FAILURE_REDACTION_POLICY,
    AblationCellArtifact,
    AblationCellRecord,
    CellOutcome,
    ClaimGateAudit,
    PilotAblationSuiteState,
    PilotArm,
    SUITE_EVIDENCE_MANIFEST_COPY,
    SUITE_INPUT_SNAPSHOT_COPY,
    _judge_unit,
    audit_pilot_ablation_grid,
    derive_d_review_and_warnings,
    failure_text_contains_sensitive_material,
)
from app.schemas import CorpusPassage, GeneratedReview, ModelCallAudit
from app.hy3_review import (
    GENERATOR_BASE_PROMPT_HASH_SCOPE,
    GENERATOR_OUTPUT_HASH_SCOPE,
    GENERATOR_PROMPT_HASH_SCOPE,
    GENERATOR_REASONING_EFFORT,
    GENERATOR_RESPONSE_HASH_SCOPE,
    generator_base_messages_for_stage,
    generator_schema_for_stage,
)
from evaluator.judge.hy3_client import JUDGE_OUTPUT_SCHEMA
from evaluator.judge.prompts import build_messages, system_prefix
from evaluator.artifact_security import ArtifactSecurityError, assert_json_safe
from evaluator.pilot_identity import is_formal_hy3_metadata
from evaluator.schemas import StrictModel


ARTIFACT_AUDIT_SCHEMA_VERSION = "mitoevidence.pilot-ablation-artifact-audit.v3"
EXPECTED_SUCCESS_FILES = {
    "artifact.json",
    "review.json",
    "retrieval.jsonl",
    "claim_gates.jsonl",
}


class ManifestFile(StrictModel):
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestSecurity(StrictModel):
    contains_api_key: Literal[False]
    contains_reasoning_content: Literal[False]


class CellManifest(StrictModel):
    schema_version: Literal[
        "mitoevidence.pilot-ablation.v1",
        "mitoevidence.pilot-ablation.v2",
        "mitoevidence.pilot-ablation.v3",
    ]
    question_id: str
    replicate: int = Field(ge=1)
    arm: PilotArm
    formal_status: Literal["pilot_ablation_generation_unscored"]
    files: dict[str, ManifestFile]
    security: ManifestSecurity

    @model_validator(mode="after")
    def _exact_file_set(self) -> "CellManifest":
        if set(self.files) != EXPECTED_SUCCESS_FILES:
            raise ValueError(
                "success manifest.files 必须恰为 "
                f"{sorted(EXPECTED_SUCCESS_FILES)}，得到 {sorted(self.files)}"
            )
        return self


class FailureSecurity(StrictModel):
    contains_api_key: Literal[False]
    contains_reasoning_content: Literal[False] | None = None
    failure_text_sanitized: Literal[True] | None = None
    redaction_policy: Literal["mitoevidence.failure-redaction.v1"] | None = None


class FailureArtifact(StrictModel):
    schema_version: Literal[
        "mitoevidence.pilot-ablation.v1",
        "mitoevidence.pilot-ablation.v2",
        "mitoevidence.pilot-ablation.v3",
        "mitoevidence.pilot-ablation.v3",
    ]
    question_id: str
    replicate: int = Field(ge=1)
    arm: PilotArm
    outcome: Literal["failed"]
    failure_type: str = Field(min_length=1)
    failure_reason: str = Field(min_length=1)
    security: FailureSecurity

    @model_validator(mode="after")
    def _v2_requires_actual_redaction_policy(self) -> "FailureArtifact":
        if self.schema_version in {
            "mitoevidence.pilot-ablation.v2",
            "mitoevidence.pilot-ablation.v3",
        }:
            if (
                self.security.failure_text_sanitized is not True
                or self.security.redaction_policy != FAILURE_REDACTION_POLICY
            ):
                raise ValueError("v2/v3 failure artifact 必须记录已执行的脱敏策略")
        if (
            self.schema_version == "mitoevidence.pilot-ablation.v3"
            and self.security.contains_reasoning_content is not False
        ):
            raise ValueError("v3 failure artifact 必须声明不含 reasoning_content")
        return self


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_model_sha256(value: StrictModel) -> str:
    rendered = (
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    return _sha_bytes(rendered)


def _canonical_json_sha256(value: object) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _generator_json_sha256(value: object) -> str:
    """Match app.hy3_review's recorded json.dumps hash contract."""

    return _sha_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def _issue(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _audit_top_level_files(
    suite: Path,
    state: PilotAblationSuiteState,
    state_bytes: bytes,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Audit the suite journal and immutable source snapshots.

    ``suite_state.json`` has already been established as a non-symlink regular
    file before this helper is called.  A completed runner writes the exact
    same bytes to ``suite_summary.json``; comparing bytes (rather than parsed
    models) prevents a coordinated formatting or duplicate-key ambiguity.

    Archived source snapshots became part of the formal v3 contract.  They are
    not retroactively required from v1/v2 suites, whose audit remains an
    explicitly non-formal legacy structural check.
    """

    errors: list[dict[str, str]] = []
    state_sha256 = _sha_bytes(state_bytes)
    files: dict[str, dict[str, Any]] = {
        "suite_state": {
            "path": "suite_state.json",
            "required": True,
            "regular_file": True,
            "symlink": False,
            "bytes": len(state_bytes),
            "sha256": state_sha256,
        }
    }

    summary_path = suite / "suite_summary.json"
    summary_result: dict[str, Any] = {
        "path": "suite_summary.json",
        "required": True,
        "regular_file": False,
        "symlink": summary_path.is_symlink(),
        "matches_suite_state_bytes": False,
    }
    if summary_path.is_symlink():
        errors.append(
            _issue("SUITE_SUMMARY_SYMLINK_FORBIDDEN", "suite_summary.json")
        )
    elif not summary_path.is_file():
        errors.append(
            _issue(
                "SUITE_SUMMARY_MISSING_OR_NONREGULAR",
                "suite_summary.json 缺失或不是 regular file",
            )
        )
    else:
        summary_result["regular_file"] = True
        try:
            summary_bytes = summary_path.read_bytes()
            summary_result.update(
                {
                    "bytes": len(summary_bytes),
                    "sha256": _sha_bytes(summary_bytes),
                    "matches_suite_state_bytes": summary_bytes == state_bytes,
                }
            )
            if summary_bytes != state_bytes:
                errors.append(
                    _issue(
                        "SUITE_STATE_SUMMARY_BYTES_MISMATCH",
                        "suite_summary.json 与 suite_state.json 不是逐字节一致",
                    )
                )
        except OSError as exc:
            errors.append(_issue("SUITE_SUMMARY_READ_ERROR", str(exc)))
    files["suite_summary"] = summary_result

    snapshots_required = (
        state.schema_version == "mitoevidence.pilot-ablation.v3"
    )
    snapshot_specs = (
        (
            "pilot_input_snapshot",
            SUITE_INPUT_SNAPSHOT_COPY,
            state.input_snapshot.sha256,
            "PILOT_INPUT_SNAPSHOT",
        ),
        (
            "evidence_manifest_snapshot",
            SUITE_EVIDENCE_MANIFEST_COPY,
            state.evidence_manifest_sha256,
            "EVIDENCE_MANIFEST_SNAPSHOT",
        ),
    )
    for result_key, filename, expected_sha256, code_prefix in snapshot_specs:
        path = suite / filename
        result: dict[str, Any] = {
            "path": filename,
            "required": snapshots_required,
            "regular_file": False,
            "symlink": path.is_symlink(),
            "expected_sha256": expected_sha256,
            "hash_matches_state": False,
        }
        if not snapshots_required and not path.exists() and not path.is_symlink():
            result["status"] = "not_required_for_legacy_v1_v2"
            files[result_key] = result
            continue
        if path.is_symlink():
            errors.append(
                _issue(f"{code_prefix}_SYMLINK_FORBIDDEN", filename)
            )
        elif not path.is_file():
            errors.append(
                _issue(
                    f"{code_prefix}_MISSING_OR_NONREGULAR",
                    f"{filename} 缺失或不是 regular file",
                )
            )
        else:
            result["regular_file"] = True
            try:
                snapshot_bytes = path.read_bytes()
                actual_sha256 = _sha_bytes(snapshot_bytes)
                result.update(
                    {
                        "bytes": len(snapshot_bytes),
                        "sha256": actual_sha256,
                        "hash_matches_state": actual_sha256 == expected_sha256,
                    }
                )
                if actual_sha256 != expected_sha256:
                    errors.append(
                        _issue(
                            f"{code_prefix}_HASH_MISMATCH",
                            f"{actual_sha256} != {expected_sha256}",
                        )
                    )
            except OSError as exc:
                errors.append(_issue(f"{code_prefix}_READ_ERROR", str(exc)))
        files[result_key] = result

    return (
        {
            "ok": not errors,
            "formal_v3_snapshots_required": snapshots_required,
            "files": files,
            "errors": errors,
        },
        errors,
    )


def _cell_key(record: AblationCellRecord) -> str:
    return f"{record.question_id}:replicate-{record.replicate:02d}:{record.arm.value}"


def _contained_cell_path(
    suite: Path,
    record: AblationCellRecord,
) -> tuple[Path | None, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    raw = Path(record.cell_dir)
    if raw.is_absolute() or ".." in raw.parts:
        return None, [
            _issue("CELL_PATH_TRAVERSAL", f"非法 cell_dir：{record.cell_dir!r}")
        ]
    candidate = suite / raw
    try:
        resolved = candidate.resolve()
        resolved.relative_to(suite)
    except (OSError, ValueError) as exc:
        return None, [
            _issue(
                "CELL_PATH_TRAVERSAL",
                f"cell_dir 越出 suite：{record.cell_dir!r} ({exc})",
            )
        ]
    if candidate.is_symlink():
        errors.append(_issue("CELL_SYMLINK_FORBIDDEN", record.cell_dir))
    if not resolved.is_dir():
        errors.append(_issue("CELL_DIRECTORY_MISSING", record.cell_dir))
    return resolved, errors


def _regular_contained_file(
    cell: Path,
    name: str,
) -> tuple[Path | None, list[dict[str, str]]]:
    if Path(name).name != name or name in {".", ".."}:
        return None, [_issue("MANIFEST_FILE_PATH_INVALID", repr(name))]
    path = cell / name
    errors: list[dict[str, str]] = []
    try:
        resolved = path.resolve()
        resolved.relative_to(cell)
    except (OSError, ValueError) as exc:
        return None, [
            _issue("MANIFEST_FILE_PATH_TRAVERSAL", f"{name!r}: {exc}")
        ]
    if path.is_symlink():
        errors.append(_issue("ARTIFACT_FILE_SYMLINK_FORBIDDEN", name))
    if not resolved.is_file():
        errors.append(_issue("ARTIFACT_FILE_MISSING", name))
    return resolved, errors


def _parse_jsonl_models(
    path: Path,
    model_cls: type[StrictModel],
) -> list[StrictModel]:
    values: list[StrictModel] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            values.append(model_cls.model_validate(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    return values


def _scan_success_file_security(path: Path) -> dict[str, Any]:
    """Independently scan a persisted success file's parsed JSON values."""

    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        payload: object = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(raw)
    assert_json_safe(payload)
    return {"ok": True, "policy": "shared_assert_json_safe_no_change"}


def _audit_success_cell(
    state: PilotAblationSuiteState,
    record: AblationCellRecord,
    cell: Path,
) -> tuple[dict[str, Any], AblationCellArtifact | None]:
    errors: list[dict[str, str]] = []
    expected_formal_status = ABLATION_FORMAL_STATUS_BY_SCHEMA_VERSION[
        state.schema_version
    ]
    manifest_path, path_errors = _regular_contained_file(cell, "manifest.json")
    errors.extend(path_errors)
    manifest: CellManifest | None = None
    if manifest_path is not None and manifest_path.is_file() and not path_errors:
        try:
            manifest_bytes = manifest_path.read_bytes()
            actual_manifest_hash = _sha_bytes(manifest_bytes)
            if actual_manifest_hash != record.cell_manifest_sha256:
                errors.append(
                    _issue(
                        "SUITE_RECORD_MANIFEST_HASH_MISMATCH",
                        f"{actual_manifest_hash} != {record.cell_manifest_sha256}",
                    )
                )
            manifest = CellManifest.model_validate_json(manifest_bytes)
        except (OSError, ValueError) as exc:
            errors.append(_issue("CELL_MANIFEST_INVALID", str(exc)))

    verified_files: dict[str, dict[str, Any]] = {}
    security_scans: dict[str, dict[str, Any]] = {}
    safe_file_paths: dict[str, Path] = {}
    if manifest is not None:
        if (
            manifest.question_id != record.question_id
            or manifest.replicate != record.replicate
            or manifest.arm is not record.arm
        ):
            errors.append(_issue("CELL_MANIFEST_KEY_MISMATCH", _cell_key(record)))
        if manifest.schema_version != state.schema_version:
            errors.append(
                _issue(
                    "CELL_MANIFEST_SUITE_SCHEMA_MISMATCH",
                    f"{manifest.schema_version} != {state.schema_version}",
                )
            )
        if manifest.formal_status != expected_formal_status:
            errors.append(
                _issue(
                    "CELL_MANIFEST_FORMAL_STATUS_MISMATCH",
                    f"{manifest.formal_status!r} != {expected_formal_status!r}",
                )
            )
        for name, expected in manifest.files.items():
            file_path, file_errors = _regular_contained_file(cell, name)
            errors.extend(file_errors)
            if file_path is None or not file_path.is_file():
                continue
            try:
                actual_size = file_path.stat().st_size
                actual_hash = _sha_file(file_path)
            except OSError as exc:
                errors.append(_issue("ARTIFACT_FILE_READ_ERROR", f"{name}: {exc}"))
                continue
            if actual_size != expected.bytes:
                errors.append(
                    _issue(
                        "ARTIFACT_FILE_SIZE_MISMATCH",
                        f"{name}: {actual_size} != {expected.bytes}",
                    )
                )
            if actual_hash != expected.sha256:
                errors.append(
                    _issue(
                        "ARTIFACT_FILE_HASH_MISMATCH",
                        f"{name}: {actual_hash} != {expected.sha256}",
                    )
                )
            verified_files[name] = {
                "bytes": actual_size,
                "sha256": actual_hash,
                "size_matches": actual_size == expected.bytes,
                "hash_matches": actual_hash == expected.sha256,
            }
            if not file_errors:
                safe_file_paths[name] = file_path
                try:
                    security_scans[name] = _scan_success_file_security(file_path)
                except ArtifactSecurityError as exc:
                    security_scans[name] = {
                        "ok": False,
                        "policy": "shared_assert_json_safe_no_change",
                        "detail": str(exc),
                    }
                    errors.append(
                        _issue(
                            "SUCCESS_FILE_SENSITIVE_MATERIAL_DETECTED",
                            f"{name}: {exc}",
                        )
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    security_scans[name] = {
                        "ok": False,
                        "policy": "shared_assert_json_safe_no_change",
                        "detail": str(exc),
                    }
                    errors.append(
                        _issue("SUCCESS_FILE_SECURITY_SCAN_INVALID", f"{name}: {exc}")
                    )

        try:
            actual_names = {path.name for path in cell.iterdir()}
            expected_names = {*manifest.files, "manifest.json"}
            if actual_names != expected_names:
                errors.append(
                    _issue(
                        "CELL_FILE_SET_MISMATCH",
                        f"actual={sorted(actual_names)} expected={sorted(expected_names)}",
                    )
                )
        except OSError as exc:
            errors.append(_issue("CELL_DIRECTORY_READ_ERROR", str(exc)))

    artifact: AblationCellArtifact | None = None
    artifact_path = safe_file_paths.get("artifact.json")
    if artifact_path is not None:
        try:
            raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            raw_review = raw_artifact.get("review") if isinstance(raw_artifact, dict) else None
            if (
                record.arm is PilotArm.D
                and isinstance(raw_artifact, dict)
                and raw_artifact.get("judge_provenance") is None
            ):
                legacy = (
                    raw_artifact.get("schema_version")
                    == "mitoevidence.pilot-ablation.v1"
                )
                errors.append(
                    _issue(
                        (
                            "D_JUDGE_PROVENANCE_UNAVAILABLE_LEGACY_V1"
                            if legacy
                            else "D_JUDGE_PROVENANCE_MISSING"
                        ),
                        f"{_cell_key(record)}: D artifact 只有 aggregate，"
                        "无法审计 Judge provider/model/prompt/config/schema",
                    )
                )
            if (
                isinstance(raw_review, dict)
                and raw_review.get("answerability") == "out_of_scope"
                and raw_review.get("claims")
            ):
                errors.append(
                    _issue(
                        "OUT_OF_SCOPE_CLAIMS_NONEMPTY",
                        f"{_cell_key(record)}: out_of_scope review 含 "
                        f"{len(raw_review['claims'])} 条 claims",
                    )
                )
            artifact = AblationCellArtifact.model_validate(raw_artifact)
        except (OSError, ValueError) as exc:
            errors.append(_issue("CELL_ARTIFACT_SCHEMA_INVALID", str(exc)))
    if artifact is not None:
        if (
            artifact.question_id != record.question_id
            or artifact.replicate != record.replicate
            or artifact.arm is not record.arm
        ):
            errors.append(_issue("CELL_ARTIFACT_KEY_MISMATCH", _cell_key(record)))
        if artifact.schema_version != state.schema_version:
            errors.append(
                _issue(
                    "CELL_ARTIFACT_SUITE_SCHEMA_MISMATCH",
                    f"{artifact.schema_version} != {state.schema_version}",
                )
            )
        if artifact.formal_status != expected_formal_status:
            errors.append(_issue("CELL_ARTIFACT_FORMAL_STATUS_MISMATCH", _cell_key(record)))
        if artifact.arm_definition != ARM_DEFINITION_BY_ID[artifact.arm]:
            errors.append(_issue("CELL_ARTIFACT_ARM_DEFINITION_DRIFT", _cell_key(record)))
        if (
            artifact.arm is PilotArm.D
            and artifact.judge_provenance is not None
            and artifact.judge_provenance.k != state.judge_k
        ):
            errors.append(
                _issue(
                    "D_JUDGE_K_SUITE_MISMATCH",
                    f"artifact={artifact.judge_provenance.k} suite={state.judge_k}",
                )
            )
        if (
            artifact.evidence_manifest_path != state.evidence_manifest_path
            or artifact.evidence_manifest_sha256 != state.evidence_manifest_sha256
        ):
            errors.append(_issue("CELL_EVIDENCE_SNAPSHOT_MISMATCH", _cell_key(record)))

        # The redundant files are independently hashed by the manifest and
        # must also be semantically identical to artifact.json.
        try:
            review = GeneratedReview.model_validate_json(
                safe_file_paths["review.json"].read_text(encoding="utf-8")
            )
            if review != artifact.review:
                errors.append(_issue("REVIEW_FILE_ARTIFACT_MISMATCH", _cell_key(record)))
        except (KeyError, OSError, ValueError) as exc:
            errors.append(_issue("REVIEW_FILE_SCHEMA_INVALID", str(exc)))
        try:
            passages = _parse_jsonl_models(
                safe_file_paths["retrieval.jsonl"], CorpusPassage
            )
            if passages != artifact.passages:
                errors.append(_issue("RETRIEVAL_FILE_ARTIFACT_MISMATCH", _cell_key(record)))
        except (KeyError, OSError, ValueError) as exc:
            errors.append(_issue("RETRIEVAL_FILE_SCHEMA_INVALID", str(exc)))
        try:
            gates = _parse_jsonl_models(
                safe_file_paths["claim_gates.jsonl"], ClaimGateAudit
            )
            if gates != artifact.claim_gates:
                errors.append(_issue("CLAIM_GATES_FILE_ARTIFACT_MISMATCH", _cell_key(record)))
        except (KeyError, OSError, ValueError) as exc:
            errors.append(_issue("CLAIM_GATES_FILE_SCHEMA_INVALID", str(exc)))

    return (
        {
            "cell": _cell_key(record),
            "cell_dir": record.cell_dir,
            "outcome": record.outcome.value,
            "ok": not errors,
            "manifest_files": verified_files,
            "security_scans": security_scans,
            "errors": errors,
        },
        artifact,
    )


def _audit_failure_cell(
    state: PilotAblationSuiteState,
    record: AblationCellRecord,
    cell: Path,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    failure_path, path_errors = _regular_contained_file(cell, "failure.json")
    errors.extend(path_errors)
    failure: FailureArtifact | None = None
    if failure_path is not None and failure_path.is_file() and not path_errors:
        try:
            failure = FailureArtifact.model_validate_json(
                failure_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            errors.append(_issue("FAILURE_ARTIFACT_SCHEMA_INVALID", str(exc)))
    if failure is not None:
        if failure_text_contains_sensitive_material(failure.failure_type) or (
            failure_text_contains_sensitive_material(failure.failure_reason)
        ):
            errors.append(
                _issue(
                    "FAILURE_SECRET_PATTERN_DETECTED",
                    f"{_cell_key(record)}: failure diagnostic 仍命中敏感模式",
                )
            )
        if failure.schema_version != state.schema_version:
            errors.append(
                _issue(
                    "FAILURE_ARTIFACT_SUITE_SCHEMA_MISMATCH",
                    f"{failure.schema_version} != {state.schema_version}",
                )
            )
        if (
            failure.question_id != record.question_id
            or failure.replicate != record.replicate
            or failure.arm is not record.arm
        ):
            errors.append(_issue("FAILURE_ARTIFACT_KEY_MISMATCH", _cell_key(record)))
        if (
            failure.failure_type != record.failure_type
            or failure.failure_reason != record.failure_reason
        ):
            errors.append(_issue("FAILURE_ARTIFACT_RECORD_MISMATCH", _cell_key(record)))
    try:
        actual_names = {path.name for path in cell.iterdir()}
        if actual_names != {"failure.json"}:
            errors.append(
                _issue(
                    "FAILURE_CELL_FILE_SET_MISMATCH",
                    f"actual={sorted(actual_names)} expected=['failure.json']",
                )
            )
    except OSError as exc:
        errors.append(_issue("CELL_DIRECTORY_READ_ERROR", str(exc)))
    return {
        "cell": _cell_key(record),
        "cell_dir": record.cell_dir,
        "outcome": record.outcome.value,
        "ok": not errors,
        "errors": errors,
    }


def _cross_arm_checks(
    state: PilotAblationSuiteState,
    artifacts: dict[tuple[str, int, PilotArm], AblationCellArtifact],
    artifact_file_sha256: dict[tuple[str, int, PilotArm], str],
    *,
    allow_test_fixture: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # The runtime freezes one plan per question across B/C/D and all repeats.
    for question_id in state.input_snapshot.question_ids:
        scoped = [
            artifact
            for (question, _replicate, arm), artifact in artifacts.items()
            if question == question_id and arm in {PilotArm.B, PilotArm.C, PilotArm.D}
        ]
        plan_hashes = sorted(
            {artifact.shared_plan_sha256 for artifact in scoped}
        )
        ok = len(plan_hashes) <= 1 and None not in plan_hashes
        checks.append(
            {
                "check": "B_C_D_SHARED_PLAN_HASH_ACROSS_REPLICATES",
                "scope": question_id,
                "ok": ok,
                "observed_success_artifacts": len(scoped),
                "plan_hashes": plan_hashes,
                "detail": (
                    "Only successful B/C/D artifacts are comparable; explicit failed cells remain valid."
                ),
            }
        )

    # B and C receive the same declared top-k evidence budget.
    for (question_id, replicate, arm), artifact in sorted(
        artifacts.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2].value),
    ):
        if arm not in {PilotArm.B, PilotArm.C}:
            continue
        selected = len(artifact.passages)
        ok = artifact.retrieval.top_k == state.top_k and selected <= state.top_k
        checks.append(
            {
                "check": "B_C_TOP_K_BUDGET",
                "scope": f"{question_id}:replicate-{replicate:02d}:{arm.value}",
                "ok": ok,
                "declared_suite_top_k": state.top_k,
                "artifact_top_k": artifact.retrieval.top_k,
                "selected_passages": selected,
            }
        )

    # D must be an exact gate over the corresponding C snapshot.
    d_keys = sorted(
        [key for key in artifacts if key[2] is PilotArm.D],
        key=lambda key: (key[0], key[1]),
    )
    for question_id, replicate, _arm in d_keys:
        d_artifact = artifacts[(question_id, replicate, PilotArm.D)]
        c_artifact = artifacts.get((question_id, replicate, PilotArm.C))
        scope = f"{question_id}:replicate-{replicate:02d}:D<-C"
        if c_artifact is None:
            checks.append(
                {
                    "check": "D_EXACT_C_ARTIFACT_BINDING",
                    "scope": scope,
                    "ok": False,
                    "detail": "D succeeded but the corresponding C artifact is unavailable",
                }
            )
            continue
        expected_parent = _canonical_model_sha256(c_artifact)
        expected_parent_raw = artifact_file_sha256.get(
            (question_id, replicate, PilotArm.C)
        )
        canonical_hash_applicable = (
            c_artifact.schema_version == "mitoevidence.pilot-ablation.v3"
        )
        parent_canonical_ok: bool | None = (
            d_artifact.parent_c_artifact_sha256 == expected_parent
            if canonical_hash_applicable
            else None
        )
        parent_raw_ok = (
            expected_parent_raw is not None
            and d_artifact.parent_c_artifact_sha256 == expected_parent_raw
        )
        passages_ok = d_artifact.passages == c_artifact.passages
        retrieval_ok = d_artifact.retrieval == c_artifact.retrieval
        request_ok = d_artifact.request == c_artifact.request
        shared_plan_ok = d_artifact.shared_plan == c_artifact.shared_plan
        shared_plan_hash_ok = (
            d_artifact.shared_plan_sha256 == c_artifact.shared_plan_sha256
        )
        evidence_snapshot_ok = (
            d_artifact.evidence_manifest_path
            == c_artifact.evidence_manifest_path
            and d_artifact.evidence_manifest_sha256
            == c_artifact.evidence_manifest_sha256
        )
        model_calls_ok = d_artifact.model_calls == c_artifact.model_calls
        gate_count_ok = len(d_artifact.claim_gates) == len(c_artifact.review.claims)
        gate_ids_ok = [gate.claim_id for gate in d_artifact.claim_gates] == [
            claim.claim_id for claim in c_artifact.review.claims
        ]
        expected_claims = [
            claim
            for claim, gate in zip(
                c_artifact.review.claims,
                d_artifact.claim_gates,
            )
            if gate.passed
        ] if gate_count_ok else []
        claims_exact_passed_subset_ok = (
            gate_count_ok and d_artifact.review.claims == expected_claims
        )
        deterministic_error = ""
        try:
            expected_review, expected_warnings = derive_d_review_and_warnings(
                c_artifact,
                d_artifact.claim_gates,
                judge_k=state.judge_k,
            )
            answerability_ok = (
                d_artifact.review.answerability is expected_review.answerability
            )
            answer_ok = d_artifact.review.answer == expected_review.answer
            limitations_ok = (
                d_artifact.review.limitations == expected_review.limitations
            )
            review_ok = d_artifact.review == expected_review
            warnings_ok = d_artifact.warnings == expected_warnings
        except ValueError as exc:
            deterministic_error = str(exc)
            answerability_ok = False
            answer_ok = False
            limitations_ok = False
            review_ok = False
            warnings_ok = False
        provenance = d_artifact.judge_provenance
        provenance_details: dict[str, Any]
        judge_suite_identity_ok = True
        if provenance is not None and state.schema_version == "mitoevidence.pilot-ablation.v3":
            recorded_identity = {
                field: getattr(provenance, field)
                for field in type(state.judge_provenance_identity).model_fields
            } if state.judge_provenance_identity is not None else None
            judge_suite_identity_ok = (
                recorded_identity
                == state.judge_provenance_identity.model_dump(mode="python")
                if state.judge_provenance_identity is not None
                else False
            )
        if provenance is None:
            provenance_ok = False
            provenance_details = {
                "available": False,
                "detail": "legacy D artifact has no auditable Judge provenance",
            }
        elif provenance.execution_kind == "test_fixture":
            provenance_ok = allow_test_fixture and judge_suite_identity_ok
            provenance_details = {
                "available": True,
                "execution_kind": "test_fixture",
                "detail": (
                    "explicit non-production test fixture accepted only for structural testing"
                    if allow_test_fixture
                    else "test fixture is forbidden by default production audit"
                ),
            }
        else:
            formal_identity_ok = is_formal_hy3_metadata(
                provenance.model_dump(mode="python")
            )
            schema_ok = provenance.schema_sha256 == _canonical_json_sha256(
                JUDGE_OUTPUT_SCHEMA
            )
            prompt_template_ok = provenance.prompt_template_sha256 == _sha_bytes(
                system_prefix(provenance.structured_output_channel).encode("utf-8")
            )
            endpoint_origin_ok = (
                provenance.endpoint_url.startswith(
                    provenance.endpoint_origin + "/"
                )
                and provenance.endpoint_url.endswith("/chat/completions")
            )
            expected_prompt_hashes: list[str] = []
            prompt_rebuild_error = ""
            try:
                passage_by_id = {
                    passage.passage_id: passage for passage in c_artifact.passages
                }
                for generated in c_artifact.review.claims:
                    claim, spans = _judge_unit(generated, passage_by_id)
                    expected_prompt_hashes.append(
                        _canonical_json_sha256(
                            build_messages(
                                claim,
                                spans,
                                c_artifact.request.question,
                                channel=provenance.structured_output_channel,
                            )
                        )
                    )
            except (KeyError, ValueError) as exc:
                prompt_rebuild_error = str(exc)
            recorded_prompt_hashes = [
                call.prompt_sha256 for call in provenance.calls
            ]
            prompts_ok = (
                not prompt_rebuild_error
                and recorded_prompt_hashes == expected_prompt_hashes
            )
            formal_identity_required = (
                state.schema_version == "mitoevidence.pilot-ablation.v3"
            )
            provenance_ok = (
                schema_ok
                and (formal_identity_ok or not formal_identity_required)
                and prompt_template_ok
                and endpoint_origin_ok
                and prompts_ok
                and judge_suite_identity_ok
            )
            provenance_details = {
                "available": True,
                "execution_kind": provenance.execution_kind,
                "formal_hy3_identity_allowlisted": formal_identity_ok,
                "formal_hy3_identity_allowlist_applicable": (
                    formal_identity_required
                ),
                "schema_sha256_matches_runtime": schema_ok,
                "prompt_template_sha256_matches_runtime": prompt_template_ok,
                "endpoint_url_matches_origin": endpoint_origin_ok,
                "per_claim_base_prompt_sha256_matches_c_inputs": prompts_ok,
                "identity_matches_suite": judge_suite_identity_ok,
                "prompt_rebuild_error": prompt_rebuild_error,
            }
        checks.append(
            {
                "check": "D_EXACT_C_ARTIFACT_BINDING",
                "scope": scope,
                "ok": (
                    parent_raw_ok
                    and parent_canonical_ok is not False
                    and passages_ok
                    and retrieval_ok
                    and request_ok
                    and shared_plan_ok
                    and shared_plan_hash_ok
                    and evidence_snapshot_ok
                    and model_calls_ok
                    and gate_count_ok
                    and gate_ids_ok
                    and claims_exact_passed_subset_ok
                    and review_ok
                    and warnings_ok
                ),
                "parent_c_canonical_sha256_matches": parent_canonical_ok,
                "parent_c_canonical_sha256_applicable": canonical_hash_applicable,
                "parent_c_manifest_file_sha256_matches": parent_raw_ok,
                "passages_identical": passages_ok,
                "retrieval_identical": retrieval_ok,
                "request_identical": request_ok,
                "shared_plan_identical": shared_plan_ok,
                "shared_plan_sha256_identical": shared_plan_hash_ok,
                "evidence_snapshot_identical": evidence_snapshot_ok,
                "model_calls_identical": model_calls_ok,
                "gate_count_equals_c_claim_count": gate_count_ok,
                "gate_claim_ids_equal_c_claim_ids": gate_ids_ok,
                "review_claims_exact_passed_c_subset": claims_exact_passed_subset_ok,
                "answerability_deterministic": answerability_ok,
                "answer_deterministic": answer_ok,
                "limitations_deterministic": limitations_ok,
                "review_exact_deterministic_derivation": review_ok,
                "warnings_deterministic": warnings_ok,
                "deterministic_rebuild_error": deterministic_error,
                "expected_parent_c_artifact_sha256": (
                    expected_parent if canonical_hash_applicable else None
                ),
                "expected_parent_c_manifest_file_sha256": expected_parent_raw,
                "recorded_parent_c_artifact_sha256": d_artifact.parent_c_artifact_sha256,
                "c_claim_count": len(c_artifact.review.claims),
                "d_gate_count": len(d_artifact.claim_gates),
            }
        )
        checks.append(
            {
                "check": "D_JUDGE_PROVENANCE_BINDING",
                "scope": scope,
                "ok": provenance_ok,
                **provenance_details,
            }
        )
    return checks


def _generator_identity_checks(
    state: PilotAblationSuiteState,
    artifacts: dict[tuple[str, int, PilotArm], AblationCellArtifact],
    *,
    allow_test_fixture: bool,
) -> tuple[list[dict[str, Any]], dict[str, str] | None, bool]:
    """Derive and verify generation identity from successful A/B/C cells."""

    checks: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str, str, str]] = []
    plan_calls: dict[str, list[ModelCallAudit]] = {}
    contains_test_fixture = False
    v3 = state.schema_version == "mitoevidence.pilot-ablation.v3"
    for key, artifact in sorted(
        artifacts.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2].value),
    ):
        question_id, replicate, arm = key
        if arm not in {PilotArm.A, PilotArm.B, PilotArm.C}:
            continue
        if (
            artifact.generator_provenance is not None
            and artifact.generator_provenance.execution_kind == "test_fixture"
        ):
            contains_test_fixture = True
        expected_stages = (
            ["ablation_A_direct"]
            if arm is PilotArm.A
            else ["plan", "synthesis"]
        )
        stages = [call.stage for call in artifact.model_calls]
        stage_ok = stages == expected_stages
        identity_fields_ok = True
        for call in artifact.model_calls:
            endpoint = urlsplit(call.endpoint_origin)
            call_ok = (
                bool(call.provider.strip())
                and bool(call.model.strip())
                and bool(endpoint.scheme and endpoint.netloc)
                and bool(re.fullmatch(r"[0-9a-f]{64}", call.config_sha256))
            )
            identity_fields_ok = identity_fields_ok and call_ok
            identities.append(
                (
                    call.provider,
                    call.model,
                    call.endpoint_origin,
                    call.endpoint_url,
                    call.config_sha256,
                )
            )
            call_formal_identity = is_formal_hy3_metadata(
                {
                    "execution_kind": "remote_hy3",
                    "provider": call.provider,
                    "model": call.model,
                    "endpoint_origin": call.endpoint_origin,
                    "endpoint_url": call.endpoint_url,
                }
            )
            if not call_formal_identity:
                contains_test_fixture = True
        if arm in {PilotArm.B, PilotArm.C} and artifact.model_calls:
            plan_calls.setdefault(question_id, []).append(artifact.model_calls[0])
        checks.append(
            {
                "check": "GENERATOR_CELL_STAGE_AND_IDENTITY",
                "scope": f"{question_id}:replicate-{replicate:02d}:{arm.value}",
                "ok": stage_ok and identity_fields_ok,
                "expected_stages": expected_stages,
                "recorded_stages": stages,
                "identity_fields_complete": identity_fields_ok,
            }
        )
        if not v3:
            checks.append(
                {
                    "check": "GENERATOR_V3_PROMPT_SCHEMA_SEED_BINDING",
                    "scope": f"{question_id}:replicate-{replicate:02d}:{arm.value}",
                    "ok": True,
                    "applicable": False,
                    "status": "legacy_v1_v2_provenance_unavailable",
                    "detail": (
                        "v1/v2 可做原始 artifact bytes/hash 的 structural audit，"
                        "但不具备 v3 generator prompt/schema/output/seed contract"
                    ),
                }
            )
            continue

        provenance = artifact.generator_provenance
        provenance_ok = (
            provenance is not None
            and state.generator_provenance is not None
            and provenance == state.generator_provenance
        )
        per_call: list[dict[str, Any]] = []
        for call in artifact.model_calls:
            prompt_request = artifact.request
            if call.stage == "synthesis" and artifact.shared_plan is not None:
                prompt_request = artifact.request.model_copy(
                    update={
                        "answerability_hint": artifact.shared_plan.answerability_hint
                    }
                )
            expected_messages = generator_base_messages_for_stage(
                call.stage,
                prompt_request,
                artifact.passages,
            )
            expected_prompt = _generator_json_sha256(expected_messages)
            expected_schema = _generator_json_sha256(
                generator_schema_for_stage(call.stage)
            )
            output = (
                artifact.shared_plan
                if call.stage == "plan"
                else artifact.review
            )
            expected_output = (
                _generator_json_sha256(output.model_dump(mode="json"))
                if output is not None
                else ""
            )
            expected_max_tokens = 8192 if call.stage == "synthesis" else 4096
            call_ok = (
                call.base_prompt_sha256 == expected_prompt
                and call.base_prompt_hash_scope == GENERATOR_BASE_PROMPT_HASH_SCOPE
                and call.prompt_sha256 == expected_prompt
                and call.prompt_hash_scope == GENERATOR_PROMPT_HASH_SCOPE
                and call.schema_sha256 == expected_schema
                and call.structured_output_sha256 == expected_output
                and call.structured_output_hash_scope == GENERATOR_OUTPUT_HASH_SCOPE
                and bool(re.fullmatch(r"[0-9a-f]{64}", call.response_sha256))
                and call.response_hash_scope == GENERATOR_RESPONSE_HASH_SCOPE
                and call.temperature == 0.2
                and call.reasoning_effort == GENERATOR_REASONING_EFFORT
                and call.max_tokens == expected_max_tokens
                and call.parse_source in {"tool_call", "content_json"}
                and call.attempt_count == 1
            )
            per_call.append(
                {
                    "stage": call.stage,
                    "ok": call_ok,
                    "base_prompt_matches_inputs": call.base_prompt_sha256
                    == expected_prompt,
                    "successful_prompt_matches_base_no_repair": call.prompt_sha256
                    == expected_prompt,
                    "schema_matches_runtime": call.schema_sha256 == expected_schema,
                    "structured_output_matches_artifact": call.structured_output_sha256
                    == expected_output,
                    "response_hash_shape_valid_but_raw_response_not_retained": bool(
                        re.fullmatch(r"[0-9a-f]{64}", call.response_sha256)
                    ),
                    "sampling_and_parse_contract_ok": (
                        call.temperature == 0.2
                        and call.max_tokens == expected_max_tokens
                        and call.parse_source in {"tool_call", "content_json"}
                        and call.attempt_count == 1
                    ),
                }
            )
        checks.append(
            {
                "check": "GENERATOR_V3_PROMPT_SCHEMA_SEED_BINDING",
                "scope": f"{question_id}:replicate-{replicate:02d}:{arm.value}",
                "ok": provenance_ok and all(row["ok"] for row in per_call),
                "applicable": True,
                "cell_provenance_matches_suite": provenance_ok,
                "calls": per_call,
            }
        )

    unique_identities = sorted(set(identities))
    identity_consistent = len(unique_identities) == 1 and bool(identities)
    derived_formal_identity = False
    if identity_consistent:
        identity = unique_identities[0]
        derived_formal_identity = is_formal_hy3_metadata(
            {
                "execution_kind": "remote_hy3",
                "provider": identity[0],
                "model": identity[1],
                "endpoint_origin": identity[2],
                "endpoint_url": identity[3],
            }
        )
    declared_formal_identity = (
        state.generator_provenance is not None
        and is_formal_hy3_metadata(
            state.generator_provenance.model_dump(mode="python")
        )
    )
    production_provider = bool(
        v3 and derived_formal_identity and declared_formal_identity
    )
    identity_accepted_for_schema = (
        production_provider if v3 else identity_consistent
    )
    checks.append(
        {
            "check": "GENERATOR_IDENTITY_CONSISTENT_ACROSS_A_B_C",
            "scope": "suite",
            "ok": identity_consistent
            and (identity_accepted_for_schema or allow_test_fixture),
            "unique_identities": [
                {
                    "provider": identity[0],
                    "model": identity[1],
                    "endpoint_origin": identity[2],
                    "endpoint_url": identity[3],
                    "config_sha256": identity[4],
                }
                for identity in unique_identities
            ],
            "production_provider": production_provider,
            "formal_hy3_identity_allowlisted": production_provider,
            "formal_hy3_identity_allowlist_applicable": v3,
            "derived_call_identity_allowlisted": derived_formal_identity,
            "declared_suite_identity_allowlisted": declared_formal_identity,
            "test_fixture_allowed": allow_test_fixture,
        }
    )
    for question_id, calls in sorted(plan_calls.items()):
        plan_call_ok = bool(calls) and all(call == calls[0] for call in calls[1:])
        checks.append(
            {
                "check": "B_C_SHARED_PLAN_MODEL_CALL",
                "scope": question_id,
                "ok": plan_call_ok,
                "observed_success_calls": len(calls),
                "detail": "all successful B/C replicates must reuse the exact plan call audit",
            }
        )
    identity = (
        {
            "provider": unique_identities[0][0],
            "model": unique_identities[0][1],
            "endpoint_origin": unique_identities[0][2],
            "endpoint_url": unique_identities[0][3],
            "config_sha256": unique_identities[0][4],
        }
        if identity_consistent
        else None
    )
    return checks, identity, contains_test_fixture


def audit_pilot_ablation_artifacts(
    suite_dir: str | Path,
    *,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    """Return a complete read-only artifact audit for one runtime suite."""

    suite = Path(suite_dir).resolve()
    if not suite.is_dir():
        raise ValueError(f"suite 目录不存在：{suite}")
    state_path = suite / "suite_state.json"
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("suite_state.json 缺失或为不允许的 symlink")
    try:
        state_bytes = state_path.read_bytes()
        state = PilotAblationSuiteState.model_validate_json(state_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError(f"suite_state.json 不合规：{exc}") from exc

    top_level_files, top_level_errors = _audit_top_level_files(
        suite,
        state,
        state_bytes,
    )

    grid_audit = audit_pilot_ablation_grid(state)
    cell_results: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, int, PilotArm], AblationCellArtifact] = {}
    artifact_file_sha256: dict[tuple[str, int, PilotArm], str] = {}
    for record in state.records:
        cell, path_errors = _contained_cell_path(suite, record)
        if cell is None or path_errors:
            cell_results.append(
                {
                    "cell": _cell_key(record),
                    "cell_dir": record.cell_dir,
                    "outcome": record.outcome.value,
                    "ok": False,
                    "errors": path_errors,
                }
            )
            continue
        if record.outcome is CellOutcome.SUCCEEDED:
            result, artifact = _audit_success_cell(state, record, cell)
            cell_results.append(result)
            if artifact is not None:
                key = (record.question_id, record.replicate, record.arm)
                artifacts[key] = artifact
                artifact_hash = result.get("manifest_files", {}).get(
                    "artifact.json", {}
                ).get("sha256")
                if isinstance(artifact_hash, str):
                    artifact_file_sha256[key] = artifact_hash
        else:
            cell_results.append(_audit_failure_cell(state, record, cell))

    cross_checks = _cross_arm_checks(
        state,
        artifacts,
        artifact_file_sha256,
        allow_test_fixture=allow_test_fixture,
    )
    generator_checks, generator_identity, generator_has_fixture = (
        _generator_identity_checks(
            state,
            artifacts,
            allow_test_fixture=allow_test_fixture,
        )
    )
    cross_checks.extend(generator_checks)
    errors: list[dict[str, str]] = [
        {
            "scope": "suite",
            "code": str(error["code"]),
            "detail": str(error["detail"]),
        }
        for error in top_level_errors
    ]
    for cell in cell_results:
        for error in cell["errors"]:
            errors.append(
                {
                    "scope": str(cell["cell"]),
                    "code": str(error["code"]),
                    "detail": str(error["detail"]),
                }
            )
    for check in cross_checks:
        if not check["ok"]:
            errors.append(
                {
                    "scope": str(check["scope"]),
                    "code": str(check["check"]),
                    "detail": str(check.get("detail") or "cross-arm invariant failed"),
                }
            )

    outcomes = Counter(record.outcome.value for record in state.records)
    safety_failures = [
        error for error in errors if error["code"] == "OUT_OF_SCOPE_CLAIMS_NONEMPTY"
    ]
    artifact_integrity_ok = not errors
    runtime_complete = bool(grid_audit["runtime_complete"])
    judge_has_fixture = (
        state.judge_provenance_identity is not None
        and state.judge_provenance_identity.execution_kind == "test_fixture"
    ) or any(
        artifact.judge_provenance is not None
        and artifact.judge_provenance.execution_kind == "test_fixture"
        for artifact in artifacts.values()
    )
    generator_has_fixture = generator_has_fixture or (
        state.generator_provenance is not None
        and state.generator_provenance.execution_kind == "test_fixture"
    )
    generator_formal_identity = (
        state.generator_provenance is not None
        and is_formal_hy3_metadata(
            state.generator_provenance.model_dump(mode="python")
        )
    )
    judge_formal_identity = (
        state.judge_provenance_identity is not None
        and is_formal_hy3_metadata(
            state.judge_provenance_identity.model_dump(mode="python")
        )
    )
    formal_runtime_identity_ok = bool(
        state.schema_version == "mitoevidence.pilot-ablation.v3"
        and generator_formal_identity
        and judge_formal_identity
    )
    suite_binding_ok = bool(top_level_files["ok"])
    legacy_structural_only = state.schema_version != "mitoevidence.pilot-ablation.v3"
    non_production = (
        generator_has_fixture
        or judge_has_fixture
        or legacy_structural_only
        or not formal_runtime_identity_ok
    )
    production_ready = (
        artifact_integrity_ok
        and suite_binding_ok
        and runtime_complete
        and not non_production
        and state.schema_version == "mitoevidence.pilot-ablation.v3"
    )
    test_fixture_audit_ok = (
        artifact_integrity_ok and runtime_complete
        if allow_test_fixture
        else None
    )
    return {
        "schema_version": ARTIFACT_AUDIT_SCHEMA_VERSION,
        "audit_kind": "artifact_level_filesystem_and_cross_arm",
        "input_schema": state.schema_version,
        "suite_id": state.suite_id,
        "suite_directory_name": suite.name,
        "formal_status": ABLATION_FORMAL_STATUS_BY_SCHEMA_VERSION[
            state.schema_version
        ],
        "formal_status_source": "runtime_constant_by_input_schema",
        "formal_runtime_identity": {
            "shared_allowlist": "evaluator.pilot_identity.is_formal_hy3_metadata",
            "generator_allowlisted": generator_formal_identity,
            "judge_allowlisted": judge_formal_identity,
            "ok": formal_runtime_identity_ok,
        },
        "audit_mode": (
            "allow_test_fixture" if allow_test_fixture else "production"
        ),
        "state_grid": {
            "expected_grid_cells": grid_audit["expected_grid_cells"],
            "recorded_grid_cells": grid_audit["recorded_grid_cells"],
            "missing_grid_cells": grid_audit["missing_grid_cells"],
            "grid_complete": grid_audit["grid_complete"],
            "suite_finalized": grid_audit["suite_finalized"],
            "runtime_complete": runtime_complete,
        },
        "top_level_files": top_level_files,
        "records": {
            "total": len(state.records),
            "succeeded": outcomes.get(CellOutcome.SUCCEEDED.value, 0),
            "failed": outcomes.get(CellOutcome.FAILED.value, 0),
            "audited_cells": len(cell_results),
        },
        "artifact_integrity_ok": artifact_integrity_ok,
        "suite_binding_ok": suite_binding_ok,
        "structural_audit_ok": artifact_integrity_ok,
        "production_ready": production_ready,
        "non_production": non_production,
        "legacy_structural_only": legacy_structural_only,
        "test_fixture_audit_ok": test_fixture_audit_ok,
        "ok": production_ready,
        "derived_generator_identity": generator_identity,
        "declared_generator_provenance": (
            state.generator_provenance.model_dump(mode="json")
            if state.generator_provenance is not None
            else None
        ),
        "declared_judge_provenance_identity": (
            state.judge_provenance_identity.model_dump(mode="json")
            if state.judge_provenance_identity is not None
            else None
        ),
        "cell_results": cell_results,
        "cross_arm_checks": cross_checks,
        "errors": errors,
        "warnings": [
            "Explicit outcome=failed cells are valid retained observations when failure.json matches the suite record."
        ],
        "limitations": [
            "This audit verifies file integrity and runtime structural invariants; it does not score scientific correctness.",
            "It verifies recorded retrieval artifacts and cross-arm bindings, but does not rerun sparse TF-IDF or graph retrieval from source XML.",
            "New D artifacts retain Judge identity, base-prompt/schema/config hashes and per-sample bindings; this audit does not replay the external calls or retain every repair-attempt message.",
            "Generator v3 rebinds the validated structured output to plan/review and validates response_sha256 shape/scope; the raw provider message is intentionally not retained, so response_sha256 itself cannot be recomputed offline.",
            "The internal hash chain has no external signed root in this report; a party able to rewrite every file and hash could construct a different self-consistent suite.",
            "It does not contact Hy3 or any literature service and never rewrites experiment artifacts.",
        ],
        "safety": {
            "review_boundary_ok": not safety_failures,
            "violations": safety_failures,
            "network_calls_performed": False,
            "artifacts_modified": False,
        },
    }
