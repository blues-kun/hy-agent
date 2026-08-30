#!/usr/bin/env python3
"""Audit formal experiment readiness without making network/API calls."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读预检真实 Hy3、专家参考一致度、判别力、稳定性、对抗性及 A/B/C/D 消融"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--expert-reference",
        action="append",
        type=Path,
        help="调用方声明的专家参考 JSONL；可重复。省略时使用仓库四类 127 条标注。",
    )
    parser.add_argument("--reference-authority", default="user_declared_expert")
    parser.add_argument("--expert-concordance-input", type=Path)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--ablation-input", type=Path)
    parser.add_argument("--api-key-env", default="HY3_API_KEY")
    parser.add_argument("--output", default="-")
    parser.add_argument(
        "--print-input-schemas",
        action="store_true",
        help="输出专家 concordance、有效性和消融输入 Schema 后退出",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="任一阶段不是 ready 时返回 2；默认只生成诚实报告并返回 0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from evaluator.experiment_protocol import (
        ABLATION_RUNTIME_SCHEMA_VERSIONS,
        ABLATION_SCHEMA_VERSION,
        AblationInput,
        DEFAULT_EXPERT_REFERENCE_PATHS,
        ExpertConcordanceInput,
        ReadinessStatus,
        build_experiment_preflight,
    )
    from evaluator.validation import ValidationInput
    from app.ablation import PilotAblationSuiteState

    args = build_parser().parse_args(argv)
    if args.print_input_schemas:
        result = {
            "expert_concordance": ExpertConcordanceInput.model_json_schema(),
            "validation": ValidationInput.model_json_schema(),
            "ablation": {
                "supported_input_schemas": {
                    ABLATION_SCHEMA_VERSION: AblationInput.model_json_schema(),
                    **{
                        version: PilotAblationSuiteState.model_json_schema()
                        for version in ABLATION_RUNTIME_SCHEMA_VERSIONS
                    },
                }
            },
        }
        _write(args.output, json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    references = args.expert_reference or [Path(path) for path in DEFAULT_EXPERT_REFERENCE_PATHS]
    report = build_experiment_preflight(
        args.repo_root,
        api_key_env=args.api_key_env,
        expert_reference_paths=references,
        reference_authority=args.reference_authority,
        expert_concordance_input=args.expert_concordance_input,
        validation_input=args.validation_input,
        ablation_input=args.ablation_input,
    )
    _write(
        args.output,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    if args.fail_on_blocked and any(
        stage.status is not ReadinessStatus.READY for stage in report.stages
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
