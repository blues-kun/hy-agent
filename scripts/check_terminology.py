#!/usr/bin/env python3
"""本地版本化术语初筛 CLI；不联网、不冒充 MeSH/GO 核验。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.rules.terminology_check import (
    DEFAULT_VOCABULARY_PATH,
    TerminologyCheckItem,
    TerminologyChecker,
    TerminologyStatus,
)


def _read_jsonl(path: Path) -> list[TerminologyCheckItem]:
    items: list[TerminologyCheckItem] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
            items.append(TerminologyCheckItem.model_validate(payload))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 TerminologyCheckItem：{exc}") from exc
    return items


def _write_json(path: Path | None, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_queue(path: Path, queue: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in queue)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用本地版本化词表进行 verified/rejected/unknown 三态初筛"
    )
    parser.add_argument("--input", required=True, type=Path, help="TerminologyCheckItem JSONL")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY_PATH)
    parser.add_argument("--output", type=Path, help="完整 JSON 报告；省略时写 stdout")
    parser.add_argument("--review-queue", type=Path, help="仅 unknown 项的 JSONL 复核队列")
    parser.add_argument(
        "--fail-on-rejected",
        action="store_true",
        help="存在明确命中本地禁用表的术语时返回退出码 2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checker = TerminologyChecker.from_path(args.vocabulary)
        report = checker.check_many(_read_jsonl(args.input))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    payload = report.model_dump(mode="json")
    _write_json(args.output, payload)
    if args.review_queue is not None:
        _write_queue(args.review_queue, [item.model_dump(mode="json") for item in report.review_queue])
    if args.fail_on_rejected and any(
        result.status is TerminologyStatus.REJECTED for result in report.results
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
