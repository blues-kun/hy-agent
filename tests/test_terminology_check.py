from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluator.rules.terminology_check import (
    DEFAULT_VOCABULARY_PATH,
    LocalTerminologyVocabulary,
    MatchKind,
    TerminologyCheckItem,
    TerminologyChecker,
    TerminologyStatus,
    load_vocabulary,
    normalize_term,
)


def test_default_vocabulary_loads_with_auditable_digest() -> None:
    vocabulary, digest = load_vocabulary()
    assert vocabulary.vocabulary_id == "mitoevidence-project-terms"
    assert vocabulary.version == "0.1.0"
    assert len(digest) == 64
    assert vocabulary.provenance.externally_authority_verified is False
    assert "不是完整 MeSH" in vocabulary.authority_disclaimer


def test_exact_alias_and_conservative_normalization_are_verified_locally() -> None:
    checker = TerminologyChecker.from_path()
    exact = checker.check(
        TerminologyCheckItem(item_id="1", claimed_term="β cell", requested_authority="local")
    )
    normalized = checker.check(
        TerminologyCheckItem(
            item_id="2",
            claimed_term="  MITOCHONDRIAL   FISSION ",
            requested_authority="MeSH",
        )
    )
    assert exact.status is TerminologyStatus.VERIFIED
    assert exact.match_kind is MatchKind.EXACT_ALIAS
    assert exact.external_authority_verified is False
    assert exact.review_required is False
    assert normalized.status is TerminologyStatus.VERIFIED
    assert normalized.match_kind is MatchKind.NORMALIZED_PREFERRED
    assert normalized.review_required is True  # 请求 MeSH，但本地条目未经 MeSH 核验


def test_disabled_form_is_rejected_only_by_explicit_rule() -> None:
    checker = TerminologyChecker.from_path()
    result = checker.check(
        TerminologyCheckItem(item_id="bad", claimed_term="Ca¥", requested_authority="human")
    )
    assert result.status is TerminologyStatus.REJECTED
    assert result.preferred_label == "Ca²⁺"
    assert result.match_kind is MatchKind.EXACT_DISABLED
    assert "禁用表" in result.reason
    assert result.external_authority_verified is False


def test_absent_term_is_unknown_not_rejected_and_enters_queue() -> None:
    checker = TerminologyChecker.from_path()
    summary = checker.check_many(
        [
            TerminologyCheckItem(
                item_id="new",
                claimed_term="mitochondrial crista remodeling",
                requested_authority="GO",
                context="candidate process",
            )
        ]
    )
    assert (summary.verified, summary.rejected, summary.unknown) == (0, 0, 1)
    assert summary.d8_accuracy_ready is False
    assert summary.results[0].status is TerminologyStatus.UNKNOWN
    assert summary.review_queue[0].suggested_next_step == "verify_in_go"
    assert "unknown 不等于术语错误" in summary.results[0].reason


def test_normalization_does_not_do_fuzzy_or_semantic_matching() -> None:
    checker = TerminologyChecker.from_path()
    typo = checker.check(TerminologyCheckItem(item_id="x", claimed_term="mitochondrial fissin"))
    semantic_neighbor = checker.check(
        TerminologyCheckItem(item_id="y", claimed_term="mitochondrial dynamics")
    )
    assert typo.status is TerminologyStatus.UNKNOWN
    assert semantic_neighbor.status is TerminologyStatus.UNKNOWN
    assert normalize_term("A\u2011B  C") == "a-b c"


def test_strict_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TerminologyCheckItem.model_validate(
            {"item_id": "1", "claimed_term": "β cell", "unexpected": True}
        )


def test_vocabulary_rejects_accepted_disabled_collision() -> None:
    payload = json.loads(DEFAULT_VOCABULARY_PATH.read_text(encoding="utf-8"))
    payload["disabled_terms"].append(
        {
            "form": "β cell",
            "reason": "collision",
            "suggested_label": None,
            "provenance": payload["provenance"],
        }
    )
    with pytest.raises(ValidationError, match="禁用词与可接受词冲突"):
        LocalTerminologyVocabulary.model_validate(payload)


def test_cli_writes_report_and_unknown_queue(tmp_path: Path) -> None:
    input_path = tmp_path / "items.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "ok", "claimed_term": "GSIS"}),
                json.dumps(
                    {
                        "item_id": "unknown",
                        "claimed_term": "novel process",
                        "requested_authority": "MeSH",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    queue_path = tmp_path / "queue.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_terminology.py",
            "--input",
            str(input_path),
            "--output",
            str(report_path),
            "--review-queue",
            str(queue_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert (report["verified"], report["unknown"]) == (1, 1)
    queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert [item["item_id"] for item in queue] == ["unknown"]


def test_cli_fail_on_rejected_returns_two(tmp_path: Path) -> None:
    input_path = tmp_path / "items.jsonl"
    input_path.write_text('{"item_id":"bad","claimed_term":"INS_1"}\n', encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_terminology.py",
            "--input",
            str(input_path),
            "--fail-on-rejected",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["rejected"] == 1
