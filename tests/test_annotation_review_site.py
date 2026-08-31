"""Tests for the static, read-only annotation review site artifact."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/build_annotation_review_site.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_annotation_review_site", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_payload_is_manifest_verified_and_complete():
    builder = _load_builder()
    payload = builder.build_payload(REPO_ROOT)

    assert payload["schema_version"] == "mitoevidence.annotation-review-site.v2"
    assert payload["summary"]["total_records"] == 127
    assert payload["summary"]["manifest_designation"] == "expert_consensus_gold"
    assert payload["summary"]["source_review_statuses"] == [
        "ai_prelabel_pending_human"
    ]
    assert {dataset["name"]: dataset["record_count"] for dataset in payload["datasets"]} == {
        "pilot_questions": 5,
        "claim_reviews": 50,
        "terminology_rules": 60,
        "review_pool": 12,
    }

    identities = [
        (record["dataset"], record["id"])
        for dataset in payload["datasets"]
        for record in dataset["records"]
    ]
    assert len(identities) == len(set(identities)) == 127


def test_review_payload_exposes_denominator_preserving_analytics():
    builder = _load_builder()
    analytics = builder.build_payload(REPO_ROOT)["analytics"]

    assert analytics["pilot_questions"]["answerability"] == {
        "answerable": 2,
        "out_of_scope": 1,
        "partial": 2,
    }
    assert analytics["pilot_questions"]["required_claims"] == 30
    assert analytics["pilot_questions"]["core_required_claims"] == 23
    assert analytics["pilot_questions"]["questions_with_evidence_papers"] == 0
    assert analytics["pilot_questions"]["questions_with_evidence_spans"] == 0
    assert analytics["pilot_questions"]["resolvable_source_review_links"] == 14

    claims = analytics["claim_reviews"]
    assert claims["decision"] == {
        "accept": 8,
        "accept_with_edits": 25,
        "reject": 14,
        "uncertain": 3,
    }
    assert claims["with_recorded_conditions"] == 38
    assert claims["without_recorded_conditions"] == 12
    assert claims["defect_assignments"] == 156

    terms = analytics["terminology_rules"]
    assert terms["detector"] == {"human": 11, "judge": 6, "rule": 43}
    assert terms["local_corpus_checked"] == 38
    assert terms["local_corpus_unchecked"] == 22
    assert terms["local_statement_links"] == 61
    assert terms["resolvable_local_statement_links"] == 59

    reviews = analytics["review_pool"]
    assert reviews["local_xml_verified"] == 7
    assert reviews["with_pmcid"] == 9
    assert reviews["reference_count_total"] == 2043
    assert reviews["year_range"] == [2015, 2026]


def test_review_payload_preserves_source_provenance_and_empty_evidence():
    builder = _load_builder()
    payload = builder.build_payload(REPO_ROOT)
    records = {
        (record["dataset"], record["id"]): record
        for dataset in payload["datasets"]
        for record in dataset["records"]
    }

    pilot = records[("pilot_questions", "PILOT-01")]
    assert pilot["record"]["annotator"] == "claude-fable-5-thinking (AI预标)"
    assert pilot["record"]["review_status"] == "ai_prelabel_pending_human"
    assert pilot["record"]["evidence_papers"] == []
    assert pilot["record"]["evidence_spans"] == []
    assert "缺少原文证据锚点" in pilot["risk_flags"]
    assert pilot["source_line"] == 1

    term = records[("terminology_rules", "TERM-001")]
    assert term["source_path"].endswith("terminology_blacklist.jsonl")
    assert "因果越界" in term["search_text"]


def test_generated_review_data_is_current():
    builder = _load_builder()
    expected = builder.render_payload(builder.build_payload(REPO_ROOT))
    actual = (REPO_ROOT / "review_site/data/annotations.json").read_text(encoding="utf-8")
    assert actual == expected

    standalone = REPO_ROOT / "review_site/mitoevidence-annotation-review.html"
    assert standalone.read_text(encoding="utf-8") == builder.render_standalone(
        REPO_ROOT, builder.build_payload(REPO_ROOT)
    )


def test_site_uses_relative_assets_and_safe_dom_text_rendering():
    html = (REPO_ROOT / "review_site/index.html").read_text(encoding="utf-8")
    script = (REPO_ROOT / "review_site/app.js").read_text(encoding="utf-8")
    data = json.loads(
        (REPO_ROOT / "review_site/data/annotations.json").read_text(encoding="utf-8")
    )

    assert 'href="./styles.css"' in html
    assert 'src="./app.js"' in html
    assert 'const DATA_URL = "./data/annotations.json"' in script
    assert ".innerHTML" not in script
    assert "textContent" in script
    assert '.normalize("NFKC")' in script
    assert 'claim.is_core ? "accept"' not in script
    assert "source_sha256" in script
    assert "window.__MITOEVIDENCE_ANNOTATIONS__" in script
    assert 'id="coverage-bars"' in html
    assert 'id="quick-views"' in html
    assert 'id="download-json"' in html
    assert 'id="export-csv"' in html
    assert 'window.location.protocol === "file:"' in script
    assert "fileHashWrittenByApp" in script
    assert data["repository"] == "blues-kun/hy-agent"
    assert (REPO_ROOT / "review_site/og.png").is_file()


def test_standalone_html_embeds_verified_data_and_all_runtime_assets():
    standalone = (
        REPO_ROOT / "review_site/mitoevidence-annotation-review.html"
    ).read_text(encoding="utf-8")
    assert "window.__MITOEVIDENCE_ANNOTATIONS__" in standalone
    assert "mitoevidence.annotation-review-site.v2" in standalone
    assert '<link rel="stylesheet" href="./styles.css" />' not in standalone
    assert '<script src="./app.js" defer></script>' not in standalone
    assert 'href="./favicon.svg"' not in standalone
    assert 'href="./mitoevidence-annotation-review.html"' not in standalone
    assert 'href="./data/annotations.json"' not in standalone
    assert "data:image/svg+xml;base64," in standalone
    assert "开始离线审阅" in standalone
    assert "127" in standalone


def test_pages_workflow_only_deploys_main_and_checks_javascript():
    workflow = (
        REPO_ROOT / ".github/workflows/deploy-review-site.yml"
    ).read_text(encoding="utf-8")
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "node --check review_site/app.js" in workflow
    assert "actions/deploy-pages@v5" in workflow
