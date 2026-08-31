#!/usr/bin/env python3
"""Build the static annotation-review dataset from the frozen JSONL sources.

The generated file is a presentation artifact, never a second source of truth.
Every build verifies the manifest record counts and SHA-256 digests before it
writes anything.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "annotation_prelabel/expert_gold_manifest.json"
DEFAULT_OUTPUT = Path("review_site/data/annotations.json")
DEFAULT_STANDALONE_OUTPUT = Path("review_site/mitoevidence-annotation-review.html")
SCHEMA_VERSION = "mitoevidence.annotation-review-site.v2"

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


def _counter(values: list[Any]) -> dict[str, int]:
    """Return a stable JSON counter without silently dropping null values."""
    counter = Counter("unknown" if value is None else str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def _build_analytics(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {
        dataset["name"]: [record["record"] for record in dataset["records"]]
        for dataset in datasets
    }
    pilots = rows["pilot_questions"]
    claims = rows["claim_reviews"]
    terms = rows["terminology_rules"]
    reviews = rows["review_pool"]

    pilot_required_claims = [
        claim for row in pilots for claim in row.get("required_claims", [])
    ]
    review_years = [
        int(row["bibliography"]["year"])
        for row in reviews
        if (row.get("bibliography") or {}).get("year") is not None
    ]
    defect_codes = Counter(
        code for row in claims for code in row.get("defect_codes", [])
    )
    term_categories = Counter(str(row.get("category") or "unknown") for row in terms)
    term_dimensions = Counter(
        dimension for row in terms for dimension in row.get("maps_to_dimension", [])
    )
    source_pmids = {
        str(value).split(":", 1)[1]
        for row in pilots
        for value in row.get("source_reviews", [])
        if str(value).startswith("PMID:") and ":" in str(value)
    }
    local_statement_ids = {
        statement_id
        for row in terms
        for statement_id in (row.get("observed_in_local_corpus") or [])
    }
    claim_statement_ids = {str(row.get("statement_id")) for row in claims}
    review_pmids = {
        str((row.get("bibliography") or {}).get("pmid")) for row in reviews
    }
    source_review_occurrences = [
        str(value).split(":", 1)[1]
        for row in pilots
        for value in row.get("source_reviews", [])
        if str(value).startswith("PMID:") and ":" in str(value)
    ]
    local_statement_occurrences = [
        str(statement_id)
        for row in terms
        for statement_id in (row.get("observed_in_local_corpus") or [])
    ]

    return {
        "pilot_questions": {
            "answerability": _counter([row.get("answerability") for row in pilots]),
            "required_claims": len(pilot_required_claims),
            "core_required_claims": sum(
                claim.get("is_core") is True for claim in pilot_required_claims
            ),
            "required_claim_confidence": _counter(
                [claim.get("ai_confidence") for claim in pilot_required_claims]
            ),
            "source_review_links": sum(
                len(row.get("source_reviews", [])) for row in pilots
            ),
            "unique_source_pmids": len(source_pmids),
            "resolvable_source_review_links": sum(
                pmid in review_pmids for pmid in source_review_occurrences
            ),
            "questions_with_source_reviews": sum(
                bool(row.get("source_reviews")) for row in pilots
            ),
            "questions_with_evidence_papers": sum(
                bool(row.get("evidence_papers")) for row in pilots
            ),
            "questions_with_evidence_spans": sum(
                bool(row.get("evidence_spans")) for row in pilots
            ),
            "prohibited_inferences": sum(
                len(row.get("prohibited_inferences", [])) for row in pilots
            ),
            "known_conflicts": sum(
                len(row.get("known_conflicts", [])) for row in pilots
            ),
            "verification_items": sum(
                len(row.get("needs_human_verification", [])) for row in pilots
            ),
        },
        "claim_reviews": {
            "decision": _counter([row.get("ai_decision") for row in claims]),
            "confidence": _counter([row.get("ai_confidence") for row in claims]),
            "usable_for_beta_cell_evidence": _counter(
                [row.get("usable_for_beta_cell_evidence") for row in claims]
            ),
            "with_recorded_conditions": sum(
                bool(row.get("recorded_conditions")) for row in claims
            ),
            "without_recorded_conditions": sum(
                not row.get("recorded_conditions") for row in claims
            ),
            "source_type": _counter([row.get("source_type") for row in claims]),
            "records_with_defects": sum(bool(row.get("defect_codes")) for row in claims),
            "defect_assignments": sum(len(row.get("defect_codes", [])) for row in claims),
            "defect_codes": dict(
                sorted(defect_codes.items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "terminology_rules": {
            "detector": _counter([row.get("detector") for row in terms]),
            "confidence": _counter([row.get("ai_confidence") for row in terms]),
            "local_corpus_checked": sum(
                row.get("observed_in_local_corpus") is not None for row in terms
            ),
            "local_corpus_unchecked": sum(
                row.get("observed_in_local_corpus") is None for row in terms
            ),
            "local_statement_links": sum(
                len(row.get("observed_in_local_corpus") or []) for row in terms
            ),
            "unique_local_statement_ids": len(local_statement_ids),
            "resolvable_local_statement_links": sum(
                statement_id in claim_statement_ids
                for statement_id in local_statement_occurrences
            ),
            "records_with_verification_items": sum(
                bool(row.get("needs_human_verification")) for row in terms
            ),
            "dimension_assignments": dict(
                sorted(term_dimensions.items(), key=lambda item: (-item[1], item[0]))
            ),
            "categories": dict(
                sorted(term_categories.items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "review_pool": {
            "fulltext_status": _counter(
                [(row.get("fulltext") or {}).get("status") for row in reviews]
            ),
            "local_xml_verified": sum(
                (row.get("fulltext") or {}).get("status")
                == "local_xml_verified_in_manifest"
                for row in reviews
            ),
            "with_pmcid": sum(
                bool((row.get("bibliography") or {}).get("pmcid")) for row in reviews
            ),
            "reference_count_total": sum(
                int(row.get("reference_count") or 0) for row in reviews
            ),
            "reference_count_min": min(
                int(row.get("reference_count") or 0) for row in reviews
            ),
            "reference_count_max": max(
                int(row.get("reference_count") or 0) for row in reviews
            ),
            "reference_count_mean": round(
                sum(int(row.get("reference_count") or 0) for row in reviews)
                / len(reviews),
                2,
            ),
            "published_since_2024": sum(
                int((row.get("bibliography") or {}).get("year") or 0) >= 2024
                for row in reviews
            ),
            "year_range": [min(review_years), max(review_years)] if review_years else [],
        },
    }


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
        "analytics": _build_analytics(datasets),
        "datasets": datasets,
    }


def render_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _script_safe_json(payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        rendered.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_standalone(repo_root: Path, payload: dict[str, Any]) -> str:
    """Build one portable HTML file with the verified snapshot embedded."""
    site_root = repo_root / "review_site"
    html = (site_root / "index.html").read_text(encoding="utf-8")
    css = (site_root / "styles.css").read_text(encoding="utf-8")
    javascript = (site_root / "app.js").read_text(encoding="utf-8")
    favicon = base64.b64encode((site_root / "favicon.svg").read_bytes()).decode("ascii")
    style_token = '<link rel="stylesheet" href="./styles.css" />'
    script_token = '<script src="./app.js" defer></script>'
    if style_token not in html or script_token not in html:
        raise ValueError("index.html 缺少 standalone 构建锚点")
    html = html.replace(style_token, f"<style>\n{css}\n</style>", 1)
    html = html.replace(
        'href="./favicon.svg"',
        f'href="data:image/svg+xml;base64,{favicon}"',
        1,
    )
    html = html.replace('class="brand" href="./"', 'class="brand" href="#overview"', 1)
    html = html.replace(
        'href="./mitoevidence-annotation-review.html"', 'href="#review"', 1
    )
    html = html.replace(
        '\n              download="MitoEvidence-专家标注集.html"', "", 1
    )
    html = html.replace(">下载离线 HTML</a>", ">开始离线审阅</a>", 1)
    bootstrap = (
        "<script>\nwindow.__MITOEVIDENCE_ANNOTATIONS__ = "
        + _script_safe_json(payload)
        + ";\n</script>\n<script>\n"
        + javascript.replace("</script", "<\\/script")
        + "\n</script>"
    )
    return html.replace(script_token, bootstrap, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从冻结 JSONL 构建 MitoEvidence 标注审阅台数据"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--standalone-output", type=Path, default=DEFAULT_STANDALONE_OUTPUT
    )
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
    standalone = render_standalone(args.repo_root.resolve(), payload)
    output = args.output
    standalone_output = args.standalone_output
    if not output.is_absolute():
        output = args.repo_root / output
    if not standalone_output.is_absolute():
        standalone_output = args.repo_root / standalone_output
    if args.check:
        if not output.is_file():
            print(f"缺少生成文件：{output}", file=sys.stderr)
            return 1
        if not standalone_output.is_file():
            print(f"缺少 standalone 文件：{standalone_output}", file=sys.stderr)
            return 1
        if standalone_output.read_text(encoding="utf-8") != standalone:
            print(f"standalone 文件已过期：{standalone_output}", file=sys.stderr)
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
    standalone_output.parent.mkdir(parents=True, exist_ok=True)
    standalone_output.write_text(standalone, encoding="utf-8")
    print(
        f"已生成 {output} 与 {standalone_output}："
        f"{payload['summary']['total_records']} records；"
        f"{payload['summary']['records_with_risk']} records with review flags"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
