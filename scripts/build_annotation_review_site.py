#!/usr/bin/env python3
"""Build the static annotation-review dataset from the frozen JSONL sources.

The generated file is a presentation artifact, never a second source of truth.
Every build verifies the manifest record counts and SHA-256 digests before it
writes anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "annotation_prelabel/expert_gold_manifest.json"
DEFAULT_OUTPUT = Path("review_site/data/annotations.json")
SCHEMA_VERSION = "mitoevidence.annotation-review-site.v1"

DATASET_UI = {
    "pilot_questions": {
        "label": "Pilot 问题",
        "short_label": "Pilot",
        "description": "问题级可回答性、必需主张、禁止推断与证据缺口",
    },
    "claim_reviews": {
        "label": "Claim 审核",
        "short_label": "Claim",
        "description": "候选主张的准入判断、证据文本、缺陷与修改建议",
    },
    "terminology_rules": {
        "label": "术语正误对",
        "short_label": "术语",
        "description": "错误表述、推荐表述、检测方式与本地缺陷实例",
    },
    "review_pool": {
        "label": "综述池评估",
        "short_label": "综述",
        "description": "种子综述用途、全文状态、限制与待核事项",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            value["__source_line"] = line_number
            rows.append(value)
    return rows


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        return [text for item in value for text in _flatten_text(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _flatten_text(item)]
    return [str(value)]


def _search_text(row: dict[str, Any]) -> str:
    text = " ".join(_flatten_text(row))
    aliases = {
        "β": " beta ",
        "Ca²⁺": " Ca2+ calcium ",
        "Δψm": " membrane potential delta psi ",
    }
    for source, replacement in aliases.items():
        if source in text:
            text += replacement
    return unicodedata.normalize("NFKC", text).casefold()


def _pilot_risks(row: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    if not row.get("evidence_papers") or not row.get("evidence_spans"):
        risks.append("缺少原文证据锚点")
    confidences = {
        claim.get("ai_confidence")
        for claim in row.get("required_claims", [])
        if isinstance(claim, dict)
    }
    if "low" in confidences:
        risks.append("含低置信主张")
    elif "medium" in confidences:
        risks.append("含中置信主张")
    if row.get("needs_human_verification"):
        risks.append("保留待核事项")
    return risks


def _claim_risks(row: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    decision = row.get("ai_decision")
    if decision in {"uncertain", "reject"}:
        risks.append("准入结论需关注")
    if row.get("ai_confidence") in {"low", "medium"}:
        risks.append("非高置信")
    if row.get("usable_for_beta_cell_evidence") is None:
        risks.append("可用性未确定")
    elif row.get("usable_for_beta_cell_evidence") is False:
        risks.append("不适用于 β 细胞证据")
    if not row.get("recorded_conditions"):
        risks.append("实验条件为空")
    if row.get("needs_human_verification"):
        risks.append("保留待核事项")
    return risks


def _term_risks(row: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    if row.get("ai_confidence") in {"low", "medium"}:
        risks.append("非高置信")
    if row.get("observed_in_local_corpus") is None:
        risks.append("未记录本地实例")
    if row.get("detector") == "human":
        risks.append("依赖人工判断")
    if row.get("needs_human_verification"):
        risks.append("保留待核事项")
    return risks


def _review_risks(row: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    fulltext = row.get("fulltext") or {}
    bibliography = row.get("bibliography") or {}
    if fulltext.get("status") != "local_xml_verified_in_manifest":
        risks.append("无冻结全文")
    if not bibliography.get("pmcid"):
        risks.append("无 PMCID")
    if row.get("caveats"):
        risks.append("存在使用限制")
    if row.get("required_human_checks"):
        risks.append("保留待核事项")
    return risks


def _display_metadata(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    if dataset == "pilot_questions":
        return {
            "title": row["question"],
            "subtitle": row.get("question_type") or "",
            "status": row.get("answerability") or "unknown",
            "confidence": None,
            "risks": _pilot_risks(row),
        }
    if dataset == "claim_reviews":
        return {
            "title": row["triple"],
            "subtitle": row.get("paper_short") or "",
            "status": row.get("ai_decision") or "unknown",
            "confidence": row.get("ai_confidence"),
            "risks": _claim_risks(row),
        }
    if dataset == "terminology_rules":
        return {
            "title": row["wrong"],
            "subtitle": row.get("category") or "",
            "status": row.get("detector") or "unknown",
            "confidence": row.get("ai_confidence"),
            "risks": _term_risks(row),
        }
    if dataset == "review_pool":
        bibliography = row.get("bibliography") or {}
        return {
            "title": bibliography.get("title") or row["assessment_id"],
            "subtitle": f"{bibliography.get('year', '年份未知')} · PMID {bibliography.get('pmid', '—')}",
            "status": (row.get("fulltext") or {}).get("status") or "unknown",
            "confidence": None,
            "risks": _review_risks(row),
        }
    raise KeyError(dataset)


def build_payload(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest_path = repo_root / MANIFEST_PATH.relative_to(REPO_ROOT)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str]] = set()

    for source in manifest["datasets"]:
        name = source["name"]
        if name not in DATASET_UI:
            raise ValueError(f"未配置展示方式的数据集：{name}")
        source_path = repo_root / source["path"]
        actual_hash = _sha256(source_path)
        if actual_hash != source["sha256"]:
            raise ValueError(
                f"{source['path']} SHA-256 漂移：{actual_hash} != {source['sha256']}"
            )
        rows = _load_jsonl(source_path)
        if len(rows) != source["record_count"]:
            raise ValueError(
                f"{source['path']} 条数漂移：{len(rows)} != {source['record_count']}"
            )

        records: list[dict[str, Any]] = []
        for row in rows:
            record_id = str(row.get(source["id_field"], "")).strip()
            if not record_id:
                raise ValueError(f"{source['path']} 缺少 {source['id_field']}")
            identity = (name, record_id)
            if identity in seen_ids:
                raise ValueError(f"重复记录 ID：{name}/{record_id}")
            seen_ids.add(identity)
            line_number = row.pop("__source_line")
            display = _display_metadata(name, row)
            records.append(
                {
                    "dataset": name,
                    "id": record_id,
                    "source_line": line_number,
                    "source_path": source["path"],
                    "title": display["title"],
                    "subtitle": display["subtitle"],
                    "status": display["status"],
                    "confidence": display["confidence"],
                    "risk_flags": display["risks"],
                    "search_text": _search_text(row),
                    "record": row,
                }
            )

        datasets.append(
            {
                "name": name,
                **DATASET_UI[name],
                "record_count": len(records),
                "source_path": source["path"],
                "source_sha256": actual_hash,
                "id_field": source["id_field"],
                "gold_fields": source["gold_fields"],
                "records": records,
            }
        )

    total = sum(dataset["record_count"] for dataset in datasets)
    source_review_statuses = sorted(
        {
            str(record["record"].get("review_status"))
            for dataset in datasets
            for record in dataset["records"]
            if record["record"].get("review_status") is not None
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "repository": "blues-kun/hy-agent",
        "manifest": manifest,
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": _sha256(manifest_path),
        "summary": {
            "total_records": total,
            "dataset_count": len(datasets),
            "records_with_risk": sum(
                bool(record["risk_flags"])
                for dataset in datasets
                for record in dataset["records"]
            ),
            "source_review_statuses": source_review_statuses,
            "manifest_designation": manifest["designation"],
        },
        "datasets": datasets,
    }


def render_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结 JSONL 构建 MitoEvidence 标注审阅台数据"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只验证输出与源数据一致，不写文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(args.repo_root.resolve())
    rendered = render_payload(payload)
    output = args.output
    if not output.is_absolute():
        output = args.repo_root / output
    if args.check:
        if not output.is_file():
            print(f"缺少生成文件：{output}", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != rendered:
            print(f"生成文件已过期：{output}", file=sys.stderr)
            return 1
        print(
            f"annotation review data OK：{payload['summary']['total_records']} records；"
            f"manifest {payload['manifest_sha256']}"
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        f"已生成 {output}：{payload['summary']['total_records']} records；"
        f"{payload['summary']['records_with_risk']} records with review flags"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
