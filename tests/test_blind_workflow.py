from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.blind import (
    BlindWorkflowError,
    build_blind_package,
    load_neutral_packet,
    make_adjudication_template,
    sha256_file,
    validate_adjudication,
    validate_blind_pair,
)


NOW = "2026-08-30T08:00:00Z"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    annotation = tmp_path / "annotation_prelabel"
    annotation.mkdir()
    (annotation / "README.md").write_text(
        "Neutral whitelist: pilot question_id/question; claim source facts only.\n",
        encoding="utf-8",
    )
    _write_jsonl(
        annotation / "pilot_questions" / "pilot_5_questions.jsonl",
        [
            {
                "question_id": "PILOT-01",
                "question": "MCU 改变了什么？",
                "required_claims": [{"text": "AI answer must not leak"}],
                "prohibited_inferences": ["AI reason must not leak"],
                "annotator": "AI",
                "review_status": "ai_prelabel_pending_human",
            }
        ],
    )
    claims = []
    for index in (1, 2):
        claims.append(
            {
                "review_id": f"CLM-{index:02d}",
                "statement_id": f"stmt-{index}",
                "paper_id": f"paper-{index}",
                "paper_short": f"Paper {index}",
                "source_type": "off_domain_primary",
                "section": "results",
                "triple": f"h{index} --rel--> t{index}",
                "evidence_text": f"verbatim source {index}",
                "recorded_conditions": {"species": "AI-inferred"},
                "ai_decision": "reject",
                "ai_reasoning": "must not leak",
                "annotator": "AI",
                "review_status": "ai_prelabel_pending_human",
            }
        )
    _write_jsonl(annotation / "claim_review_sample" / "claim_review_sample.jsonl", claims)
    _write_jsonl(
        annotation / "terminology_blacklist" / "terminology_blacklist.jsonl",
        [
            {
                "term_id": "TERM-001",
                "wrong": "bad AI suggestion",
                "correct": "good AI suggestion",
                "why": "AI reason",
                "annotator": "AI",
                "review_status": "ai_prelabel_pending_human",
            }
        ],
    )
    # This poison assessment must never be loaded as the review-pool source.
    _write_jsonl(
        annotation / "review_pool_assessment" / "review_pool_assessment.jsonl",
        [{"assessment_id": "REVPOOL-001", "bibliography": {"title": "POISON"}}],
    )
    manifest = tmp_path / "evidence_pool_manifest.json"
    _write_json(
        manifest,
        {
            "reviews": [
                {
                    "index": 1,
                    "pmid": "101",
                    "doi": "10.1/one",
                    "pmcid": "PMC101",
                    "title": "Manifest review one",
                    "year": 2020,
                    "is_open_access": True,
                    "epmc_hit_count": 12,
                    "refs_source": "epmc",
                    "refs_used": 12,
                },
                {
                    "index": 2,
                    "pmid": "102",
                    "doi": "10.1/two",
                    "pmcid": None,
                    "title": "Manifest review two",
                    "year": 2021,
                    "is_open_access": False,
                    "epmc_hit_count": 7,
                    "refs_source": "epmc",
                    "refs_used": 7,
                },
            ],
            "fulltext": [
                {
                    "pmid": "101",
                    "pmcid": "PMC101",
                    "path": "eval/data/corpus_raw/PMC101.xml",
                    "bytes": 123,
                    "sha256": "a" * 64,
                },
                {"pmid": "102", "pmcid": None, "error": "无 PMCID"},
            ],
        },
    )
    return annotation, manifest


def _build(tmp_path: Path) -> Path:
    annotation, evidence_manifest = _source_fixture(tmp_path)
    output = tmp_path / "blind-batch"
    build_blind_package(
        annotation_root=annotation,
        evidence_manifest_path=evidence_manifest,
        output_dir=output,
        batch_id="batch-001",
        guideline_version="guide-1",
        generated_at_utc=NOW,
    )
    return output


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _lock_review(
    path: Path,
    *,
    code: str,
    decisions: dict[tuple[str, str], str],
    structured: dict[tuple[str, str], dict] | None = None,
) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["annotator_code"] = code
    value["started_at_utc"] = NOW
    value["completed_at_utc"] = "2026-08-30T09:00:00Z"
    value["blindness_attestation"] = {
        "neutral_packet_only_confirmed": True,
        "ai_prelabels_not_viewed": True,
        "other_expert_labels_not_viewed": True,
        "independent_work_confirmed": True,
    }
    for record in value["records"]:
        key = (record["item_type"], record["item_id"])
        record["decision"] = decisions[key]
        record["confidence"] = "high"
        record["rationale"] = f"独立核验 {key[1]}"
        record["structured_fields"] = (structured or {}).get(key, {})
    value["lock"] = {
        "locked": True,
        "locked_at_utc": "2026-08-30T09:01:00Z",
        "signature_or_audit_id": f"sig-{code}",
        "amendment_of": None,
    }
    _write_json(path, value)


def _locked_pair(batch: Path) -> tuple[Path, Path]:
    a_path, b_path = batch / "expert_A.json", batch / "expert_B.json"
    decisions_a = {
        ("pilot_question", "PILOT-01"): "answerable",
        ("claim", "CLM-01"): "accept",
        ("claim", "CLM-02"): "reject",
        ("terminology", "TERM-001"): "approve_rule",
        ("review_pool", "REVPOOL-001"): "include_seed",
        ("review_pool", "REVPOOL-002"): "exclude",
    }
    decisions_b = {
        **decisions_a,
        ("terminology", "TERM-001"): "revise_rule",
        ("review_pool", "REVPOOL-001"): "include_context_only",
    }
    _lock_review(a_path, code="expert-a-code", decisions=decisions_a)
    _lock_review(b_path, code="expert-b-code", decisions=decisions_b)
    return a_path, b_path


def test_neutral_builder_enforces_allowlists_hashes_and_counts(tmp_path: Path) -> None:
    batch = _build(tmp_path)
    packet = batch / "neutral_items.jsonl"
    items, errors = load_neutral_packet(packet)
    assert errors == []
    assert len(items) == 6
    by_key = {(item["item_type"], item["item_id"]): item for item in items}

    assert by_key[("pilot_question", "PILOT-01")]["source"] == {
        "question_id": "PILOT-01",
        "question": "MCU 改变了什么？",
    }
    claim = by_key[("claim", "CLM-01")]["source"]
    assert set(claim) == {
        "review_id",
        "statement_id",
        "paper_id",
        "paper_short",
        "section",
        "triple",
        "evidence_text",
    }
    assert by_key[("terminology", "TERM-001")]["source"] == {
        "term_id": "TERM-001",
        "source_context_missing": True,
    }
    assert (
        by_key[("review_pool", "REVPOOL-001")]["source"]["bibliography"]["title"]
        == "Manifest review one"
    )

    keys = _all_keys(items)
    assert not any(key.startswith("ai_") for key in keys)
    assert "annotator" not in keys
    assert "review_status" not in keys
    assert "required_claims" not in keys
    assert "wrong" not in keys and "correct" not in keys

    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    assert all(
        not entry["path"].startswith("/") for entry in manifest["inputs"].values()
    )
    assert manifest["counts"] == {
        "total": 6,
        "pilot_question": 1,
        "claim": 2,
        "terminology": 1,
        "review_pool": 2,
    }
    for filename in ("neutral_items.jsonl", "expert_A.json", "expert_B.json"):
        assert manifest["outputs"][filename]["sha256"] == sha256_file(batch / filename)
    assert len(manifest["inputs"]) == 5
    assert len(manifest["input_snapshot_sha256"]) == 64

    with pytest.raises(BlindWorkflowError, match="拒绝覆盖"):
        build_blind_package(
            annotation_root=tmp_path / "annotation_prelabel",
            evidence_manifest_path=tmp_path / "evidence_pool_manifest.json",
            output_dir=batch,
            batch_id="batch-001",
            guideline_version="guide-1",
        )


def test_pair_validation_reports_disagreements_and_kappa_by_type(tmp_path: Path) -> None:
    batch = _build(tmp_path)
    a_path, b_path = _locked_pair(batch)
    report = validate_blind_pair(
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    assert report["ok"], report["errors"]
    assert report["counts"]["disagreements"] == 2
    assert {
        (row["item_type"], row["item_id"]) for row in report["disagreements"]
    } == {("terminology", "TERM-001"), ("review_pool", "REVPOOL-001")}

    claim_stats = report["agreement_by_item_type"]["claim"]
    assert claim_stats["raw_agreement"] == 1.0
    assert claim_stats["cohen_kappa"] == 1.0
    pilot_stats = report["agreement_by_item_type"]["pilot_question"]
    assert pilot_stats["raw_agreement"] == 1.0
    assert pilot_stats["cohen_kappa"] is None
    assert pilot_stats["kappa_note"] == "marginal_degeneracy_expected_agreement_is_one"


def test_pair_validation_rejects_broken_blinding_lock_and_record_integrity(
    tmp_path: Path,
) -> None:
    batch = _build(tmp_path)
    a_path, b_path = _locked_pair(batch)
    b = json.loads(b_path.read_text(encoding="utf-8"))
    b["annotator_code"] = "expert-a-code"
    b["blindness_attestation"]["ai_prelabels_not_viewed"] = False
    b["lock"]["locked"] = False
    b["records"][0]["decision"] = "made_up_label"
    b["records"].append(dict(b["records"][0]))
    b["source_snapshot_sha256"] = "0" * 64
    _write_json(b_path, b)

    report = validate_blind_pair(
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    assert not report["ok"]
    joined = "\n".join(report["errors"])
    assert "不同 annotator_code" in joined
    assert "ai_prelabels_not_viewed" in joined
    assert "lock.locked" in joined
    assert "decision='made_up_label'" in joined
    assert "重复记录" in joined
    assert "source_snapshot_sha256" in joined


def test_adjudication_requires_all_disagreements_hashes_reasons_and_lock(
    tmp_path: Path,
) -> None:
    batch = _build(tmp_path)
    a_path, b_path = _locked_pair(batch)
    pair = validate_blind_pair(
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    template = make_adjudication_template(pair)
    assert all(row["final_decision"] is None for row in template["records"])
    adjudication_path = batch / "adjudication.json"
    _write_json(adjudication_path, template)
    unfinished = validate_adjudication(
        adjudication_path,
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    assert not unfinished["ok"]
    assert "尚未填写" in "\n".join(unfinished["errors"])

    template["adjudicator_code"] = "third-expert-code"
    template["started_at_utc"] = "2026-08-30T10:00:00Z"
    template["completed_at_utc"] = "2026-08-30T11:00:00Z"
    for index, record in enumerate(template["records"]):
        record["rationale"] = "已回到冻结来源独立裁决"
        if index == 0:
            record["final_decision"] = "include_context_only"
        else:
            record["final_decision"] = "unresolved"
            record["unresolved_reason"] = "原始中性语境缺失"
    template["lock"] = {
        "locked": True,
        "locked_at_utc": "2026-08-30T11:01:00Z",
        "signature_or_audit_id": "sig-third-expert",
    }
    _write_json(adjudication_path, template)
    complete = validate_adjudication(
        adjudication_path,
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    assert complete["ok"], complete["errors"]
    assert complete["counts"] == {
        "expected_disagreements": 2,
        "adjudication_records": 2,
    }

    tampered = json.loads(adjudication_path.read_text(encoding="utf-8"))
    tampered["expert_a_locked_file_sha256"] = "f" * 64
    tampered["records"] = tampered["records"][:-1]
    _write_json(adjudication_path, tampered)
    invalid = validate_adjudication(
        adjudication_path,
        a_path,
        b_path,
        neutral_packet_path=batch / "neutral_items.jsonl",
        package_manifest_path=batch / "manifest.json",
    )
    assert not invalid["ok"]
    joined = "\n".join(invalid["errors"])
    assert "expert_a_locked_file_sha256" in joined
    assert "恰好覆盖全部分歧" in joined
