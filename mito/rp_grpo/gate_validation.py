"""Strict, CPU-only admission of independently replayed SFT evaluations.

This is file/trace consistency, not hardware attestation or expert approval.
The actual auditor is rerun before admission; a passed flag and an arbitrary
hash-bound document alone never authorize RL. No model generation occurs.
"""
from pathlib import Path

from rp_grpo.train_policy import canonical_hash, sha256_file, strict_json

BUNDLE = Path(__file__).resolve().parents[1]
CHECKS = {"tool_understanding", "schema_arguments", "feedback_adaptation", "end_to_end"}


def _path(value, bundle):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing bound artifact path")
    path = Path(value)
    return (path if path.is_absolute() else bundle / path).resolve(strict=True)


def _read(path):
    value = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Admission artifacts must be objects")
    return value


def _passed(report):
    if (report.get("schema") != "mito.sft-readiness.rp-evaluation.v1"
            or report.get("passed") is not True or report.get("status") != "PASS"
            or report.get("actual_model_evaluation_verified") is not True):
        raise ValueError("Independent actual-policy audit has not passed")
    checks = report.get("checks")
    if (not isinstance(checks, list) or len(checks) != len(CHECKS)
            or any(not isinstance(item, dict) for item in checks)
            or {item.get("id") for item in checks} != CHECKS
            or any(item.get("status") != "PASS" for item in checks)):
        raise ValueError("All four distinct SFT admission checks must pass")
    multi = report.get("multipath_environment_gate", {})
    if multi.get("status") != "PASS" or not multi.get("tasks_with_two_successful_policy_paths"):
        raise ValueError("Actual same-task successful multi-path coverage is missing")
    if not isinstance(report.get("episodes"), list) or not report["episodes"]:
        raise ValueError("Independent policy audit has no episodes")


def _reaudit(source, manifest, bundle):
    from rp_grpo.audit_sft import rp_audit
    config = manifest["config"]
    # Exact current local model/tokenizer/adapter bytes and all recorded action
    # observations are validated again. No generation, training or GPU use.
    return rp_audit(source, _path(config.get("model_path"), bundle),
                    _path(config.get("initial_adapter"), bundle),
                    _path(config.get("train_data"), bundle).parent,
                    _path(config.get("train_data"), bundle))


def validate_sft_gate(path, expected, *, engineering_smoke=False, bundle=None):
    if engineering_smoke:
        return {"passed": False, "engineering_smoke_override": True,
                "formal_training": False, "production_ready": False}
    bundle = Path(bundle or BUNDLE).resolve()
    path = Path(path).resolve(strict=True)
    gate = _read(path)
    if (gate.get("schema") != "mito.rp-grpo.sft-gate.v1"
            or gate.get("passed") is not True):
        raise ValueError("SFT_GATE has not passed; formal RL is disabled")
    bindings = gate.get("bindings", {})
    if not expected or any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("SFT gate binding mismatch")
    report_path = _path(gate.get("evaluation_report_path"), bundle)
    report_hash = sha256_file(report_path)
    if (report_hash != gate.get("evaluation_report_sha256")
            or report_hash != bindings.get("evaluation_report_sha256")):
        raise ValueError("SFT gate report hash mismatch")
    report = _read(report_path)
    _passed(report)
    if any(report.get("bindings", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Independent report binding mismatch")
    if report.get("auditor_sha256") != sha256_file(bundle / "rp_grpo/audit_sft.py"):
        raise ValueError("Independent auditor implementation changed")
    source_info = report.get("source_evaluation_manifest", {})
    source = _path(source_info.get("path"), bundle)
    if sha256_file(source) != source_info.get("sha256"):
        raise ValueError("Actual evaluation manifest hash mismatch")
    manifest = _read(source)
    if (manifest.get("schema") != "mito.rp-grpo.policy-run.v1"
            or manifest.get("record_source") != "actual_hf_model"
            or manifest.get("status") != "completed" or manifest.get("mode") != "evaluate"
            or manifest.get("split") != "dev" or manifest.get("arm") == "B0"):
        raise ValueError("SFT admission requires completed unassisted actual-model dev evaluation")
    if any(manifest.get("bindings", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Source evaluation binding mismatch")
    if canonical_hash(manifest.get("config")) != manifest.get("config_sha256"):
        raise ValueError("Source evaluation configuration hash mismatch")
    episodes_hash = sha256_file(source.parent / "episodes.jsonl")
    if (episodes_hash != manifest.get("episodes_sha256")
            or episodes_hash != report.get("source_episodes_sha256")):
        raise ValueError("Source policy episodes hash mismatch")
    # Replaying protects against edited summary outcomes even when somebody
    # recalculated all outer file hashes. Default thresholds cannot be loosened
    # by a report generated with custom minimums.
    recomputed = _reaudit(source, manifest, bundle)
    _passed(recomputed)
    if any(recomputed.get("bindings", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Recomputed audit binding mismatch")
    def semantic_rows(rows):
        # Replayed wall-clock times alter the environment trace hash only.
        return [{key: value for key, value in row.items() if key != "trace_sha256"} for row in rows]
    if semantic_rows(recomputed["episodes"]) != semantic_rows(report["episodes"]):
        raise ValueError("Saved audit summaries disagree with independent tool replay")
    return {"passed": True, "gate_sha256": sha256_file(path),
            "evaluation_report_sha256": report_hash, "formal_training": True,
            "independent_cpu_reaudit": True, "production_ready": False}
