#!/usr/bin/env python3
"""Download only the OA XML bytes pinned by the checked-in manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    import requests

    from tools.literature.epmc_client import EpmcClient
    from tools.literature.frozen_fetch import fetch_frozen_corpus

    parser = argparse.ArgumentParser(
        description="获取并核验 evidence_pool_manifest.json 已冻结的 OA XML，不重建证据池"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "eval" / "data" / "evidence_pool_manifest.json",
    )
    parser.add_argument("--output", type=Path, help="可选 JSON 报告；默认写 stdout")
    parser.add_argument("--trust-env", action="store_true", help="使用系统代理环境变量")
    args = parser.parse_args(argv)

    session = requests.Session()
    session.trust_env = args.trust_env
    client = EpmcClient(session=session)
    try:
        report = fetch_frozen_corpus(
            manifest_path=args.manifest,
            repo_root=REPO_ROOT,
            client=client,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"冻结语料准备失败：{exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
