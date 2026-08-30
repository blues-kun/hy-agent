"""MitoEvidence 应用闭环的离线回归测试。

测试只使用临时 JATS XML 与确定性模型替身，不调用 Hy3 或外部文献接口。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.corpus import FrozenReviewCorpus, _safe_parse_xml
from app.offline import OfflineSmokeModel
from app.pipeline import ReviewRunner
from app.pipeline import load_pilot_request
from app.schemas import (
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    RunKind,
    SearchPlan,
)
from evaluator.schemas import Answerability
from tools.literature.xml_anchor import UnsafeXmlError


JATS = """<!DOCTYPE article PUBLIC "-//NLM//DTD JATS 1.4//EN" "JATS.dtd">
<article><body>
  <sec><title>Results</title>
    <p>Pancreatic beta cell mitochondrial calcium uptake supports glucose-stimulated insulin secretion, while experimental context determines how strongly this association can be interpreted.</p>
    <p>Independent measurements of mitochondrial morphology and ATP production are needed because a structural change alone does not establish a causal metabolic mechanism.</p>
  </sec>
</body></article>
"""


def _repo(tmp_path: Path, xml: str = JATS, expected_sha: str | None = None) -> Path:
    corpus = tmp_path / "eval" / "data" / "corpus_raw"
    corpus.mkdir(parents=True)
    path = corpus / "PMC1.xml"
    path.write_text(xml, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "reviews": [{"pmid": "123", "title": "Fixture review"}],
        "fulltext": [
            {
                "pmid": "123",
                "pmcid": "PMC1",
                "path": "eval/data/corpus_raw/PMC1.xml",
                "sha256": expected_sha if expected_sha is not None else digest,
            }
        ],
    }
    (tmp_path / "eval" / "data" / "evidence_pool_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path


def test_corpus_accepts_normal_jats_doctype_and_builds_stable_anchors(tmp_path: Path):
    corpus = FrozenReviewCorpus(_repo(tmp_path))
    rows = corpus.load()
    assert len(rows) == 2
    assert rows[0].passage_id == "PMC1:p0001"
    assert rows[0].section == "Results"
    assert rows[0].anchor_exact == rows[0].text
    assert rows[0].prefix == rows[0].postfix == ""
    hits = corpus.search(["mitochondrial calcium insulin secretion"], top_k=1)
    assert [row.passage_id for row in hits] == ["PMC1:p0001"]
    assert hits[0].score > 0


def test_corpus_rejects_entity_expansion(tmp_path: Path):
    path = tmp_path / "bad.xml"
    path.write_text(
        '<!DOCTYPE article [<!ENTITY x "unsafe">]><article><p>&x;</p></article>',
        encoding="utf-8",
    )
    with pytest.raises(UnsafeXmlError):
        _safe_parse_xml(path)


def test_corpus_refuses_changed_xml_when_manifest_hash_does_not_match(tmp_path: Path):
    corpus = FrozenReviewCorpus(_repo(tmp_path, expected_sha="0" * 64))
    with pytest.raises(ValueError, match="SHA-256"):
        corpus.load()


def test_offline_smoke_materializes_traceable_run_without_claiming_model_result(tmp_path: Path):
    corpus = FrozenReviewCorpus(_repo(tmp_path / "repo"))
    runner = ReviewRunner(
        model=OfflineSmokeModel(), corpus=corpus, run_kind=RunKind.OFFLINE_SMOKE
    )
    artifact = runner.run(
        ReviewRequest(
            question_id="Q1",
            question="How does mitochondrial calcium support insulin secretion?",
            source_pmids=["123"],
        ),
        top_k=2,
    )
    assert artifact.formal_status == "offline_engineering_smoke_not_model_result"
    assert artifact.review.claims[0].evidence_passage_ids
    assert [item.status for item in artifact.anchor_checks] == ["found"]
    out = runner.write_run(artifact, out_root=tmp_path / "runs", run_id="Q1-smoke")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["application_version"] == "mitoevidence-hy3-v0.3.0"
    assert len(manifest["evidence_manifest"]["sha256"]) == 64
    assert not manifest["evidence_manifest"]["path"].startswith("/")
    assert set(manifest["files"]) == {
        "anchor_validation.jsonl",
        "judge_input.jsonl",
        "plan.json",
        "retrieval.jsonl",
        "review.json",
        "review.md",
        "run.json",
    }
    assert manifest["anchor_summary"] == {
        "ambiguous": 0,
        "error": 0,
        "found": 1,
        "not_found": 0,
    }
    assert manifest["security"] == {
        "contains_api_key": False,
        "contains_reasoning_content": False,
    }
    judge_row = json.loads((out / "judge_input.jsonl").read_text(encoding="utf-8"))
    assert judge_row["claim"]["citations"][0]["doi_or_pmid"] == "PMID:123"
    assert judge_row["evidence_spans"][0]["anchor"]["exact"]


def test_out_of_scope_request_skips_retrieval_and_refuses_safely(tmp_path: Path):
    corpus = FrozenReviewCorpus(_repo(tmp_path))
    runner = ReviewRunner(
        model=OfflineSmokeModel(), corpus=corpus, run_kind=RunKind.OFFLINE_SMOKE
    )
    artifact = runner.run(
        ReviewRequest(
            question_id="Q5",
            question="Should this patient stop metformin?",
            answerability_hint=Answerability.OUT_OF_SCOPE,
        )
    )
    assert artifact.passages == []
    assert artifact.review.answerability is Answerability.OUT_OF_SCOPE
    assert artifact.review.claims == []


class _HallucinatingModel:
    def plan(self, request):
        return (
            SearchPlan(
                queries=["mitochondrial calcium"],
                source_pmids=["123"],
                rationale="test",
                answerability_hint=Answerability.ANSWERABLE,
            ),
            ModelCallAudit(stage="plan"),
        )

    def synthesize(self, request, passages):
        return (
            GeneratedReview(
                answerability=Answerability.ANSWERABLE,
                answer="unsupported",
                claims=[
                    GeneratedClaim(
                        claim_id="C1",
                        text="unsupported",
                        evidence_passage_ids=["invented:passage"],
                    )
                ],
            ),
            ModelCallAudit(stage="synthesis"),
        )


def test_pipeline_rejects_invented_passage_identifier(tmp_path: Path):
    runner = ReviewRunner(
        model=_HallucinatingModel(),
        corpus=FrozenReviewCorpus(_repo(tmp_path)),
    )
    with pytest.raises(ValueError, match="证据幻觉"):
        runner.run(ReviewRequest(question_id="QX", question="mitochondrial calcium"))


def test_pilot_loader_does_not_leak_ai_prelabel_or_answerability(tmp_path: Path):
    path = tmp_path / "pilot.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "PILOT-X",
                "question": "neutral question",
                "scope": "declared scope",
                "answerability": "out_of_scope",
                "source_reviews": ["PMID:999"],
                "prohibited_inferences": ["hidden evaluator hint"],
                "required_claims": [{"text": "hidden gold"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request = load_pilot_request(path, "PILOT-X")
    assert request.scope == "declared scope"
    assert request.answerability_hint is None
    assert request.source_pmids == []
    assert request.prohibited_inferences == []


class _UnsafeRefusalModel(_HallucinatingModel):
    def plan(self, request):
        plan, audit = super().plan(request)
        return plan.model_copy(update={"answerability_hint": Answerability.OUT_OF_SCOPE}), audit

    def synthesize(self, request, passages):
        return (
            GeneratedReview(
                answerability=Answerability.ANSWERABLE,
                answer="unsafe answer",
                claims=[GeneratedClaim(claim_id="C1", text="unsafe", evidence_passage_ids=[])],
            ),
            ModelCallAudit(stage="synthesis"),
        )


def test_plan_classified_out_of_scope_must_refuse_even_without_request_hint(tmp_path: Path):
    runner = ReviewRunner(
        model=_UnsafeRefusalModel(),
        corpus=FrozenReviewCorpus(_repo(tmp_path)),
    )
    with pytest.raises(ValueError, match="未被模型拒答"):
        runner.run(ReviewRequest(question_id="Q", question="patient-specific dose"))
