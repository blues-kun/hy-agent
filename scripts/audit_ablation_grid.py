#!/usr/bin/env python3
"""Audit A/B/C/D run-grid completeness without dropping failures."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LEGACY_INPUT_SCHEMA = "mitoevidence.ablation.v1"
RUNTIME_INPUT_SCHEMAS = (
    "mitoevidence.pilot-ablation.v1",
    "mitoevidence.pilot-ablation.v2",
    "mitoevidence.pilot-ablation.v3",
)
RUNTIME_INPUT_SCHEMA = RUNTIME_INPUT_SCHEMAS[-1]


def _detect_input_schema(payload: object) -> str:
    """Select one strict contract without projecting or dropping fields."""

    if not isinstance(payload, dict):
        raise ValueError("A/B/C/D 网格输入顶层必须是 JSON object")
    declared = payload.get("schema_version")
    if declared is not None:
        if not isinstance(declared, str):
            raise ValueError("schema_version 必须是字符串")
        if declared not in {LEGACY_INPUT_SCHEMA, *RUNTIME_INPUT_SCHEMAS}:
            raise ValueError(f"不支持的 schema_version：{declared!r}")
        return declared

    # Older formal-analysis inputs predate an explicit schema_version and are
    # still accepted by their strict model.  Runtime state uses ``suite_id``;
    # legacy input uses ``protocol_id``.  Mixed/ambiguous payloads fail closed.
    has_legacy_identity = "protocol_id" in payload
    has_runtime_identity = "suite_id" in payload
    if has_legacy_identity and has_runtime_identity:
        raise ValueError(
            "缺少 schema_version，且同时存在 protocol_id/suite_id，输入 Schema 混合"
        )
    if has_runtime_identity:
        raise ValueError(
            "PilotAblationSuiteState runtime 输入必须显式声明 "
            f"schema_version={RUNTIME_INPUT_SCHEMA!r}"
        )
    if not has_legacy_identity:
        raise ValueError(
            "缺少 schema_version，且没有旧 Schema 的 protocol_id，无法明确判别输入"
        )
    return LEGACY_INPUT_SCHEMA


def _write(path: str, rendered: str) -> None:
    if path == "-":
        print(rendered)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from app.ablation import PilotAblationSuiteState, audit_pilot_ablation_grid
    from evaluator.experiment_protocol import AblationInput, ablation_grid_audit

    parser = argparse.ArgumentParser(
        description="审计 A/B/C/D × question × replicate 网格；失败记录保留在分母"
    )
    parser.add_argument("--input", default="-")
    parser.add_argument("--output", default="-")
    parser.add_argument("--print-input-schema", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "存在缺失网格单元时返回 2；runtime 还要求 status=completed；"
            "显式 outcome=failed 不算缺失"
        ),
    )
    args = parser.parse_args(argv)
    if args.print_input_schema:
        _write(
            args.output,
            json.dumps(
                {
                    "supported_input_schemas": {
                        LEGACY_INPUT_SCHEMA: AblationInput.model_json_schema(),
                        **{
                            schema: PilotAblationSuiteState.model_json_schema()
                            for schema in RUNTIME_INPUT_SCHEMAS
                        },
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return 0
    try:
        raw = (
            sys.stdin.read()
            if args.input == "-"
            else Path(args.input).read_text(encoding="utf-8")
        )
        payload = json.loads(raw)
        input_schema = _detect_input_schema(payload)
        if input_schema == LEGACY_INPUT_SCHEMA:
            data = AblationInput.model_validate(payload)
            result = ablation_grid_audit(data)
            # Keep all legacy output fields and add the now-explicit contract.
            result["input_schema"] = input_schema
        else:
            state = PilotAblationSuiteState.model_validate(payload)
            result = audit_pilot_ablation_grid(state)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise SystemExit(f"A/B/C/D 网格输入不合规：{exc}") from exc
    _write(args.output, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    complete = bool(result.get("runtime_complete", result["grid_complete"]))
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
