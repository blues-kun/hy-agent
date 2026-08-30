"""Offline tests for the real-Hy3 A/B/C/D Pilot runner contracts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.ablation import (
    AblationCellArtifact,
    CellOutcome,
    PilotAblationRunner,
    PilotArm,
)
from app.corpus import FrozenReviewCorpus
from app.experiment_retrieval import (
    FROZEN_GRAPH_METHOD,
    SPARSE_TFIDF_METHOD,
    FrozenEvidenceGraphRetriever,
    SparseTfidfIndex,
)
from app.hy3_review import Hy3ReviewModel
from app.schemas import (
    CorpusPassage,
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    SearchPlan,
)
from evaluator.judge import JudgeAggregate
from evaluator.experiment_protocol import (
    analyze_expert_concordance,
    build_ablation_answerability_concordance,
)
from evaluator.schemas import Answerability, JudgeVerdict, SupportVerdict


REPO_ROOT = Path(__file__).resolve().parent.parent


def _passage(identifier: str, text: str, *, pmid: str = "1") -> CorpusPassage:
    return CorpusPassage(
        passage_id=identifier,
        paper_id=f"PMID:{pmid}",
        pmid=pmid,
        pmcid=f"PMC{pmid}",
        title="Beta cell mitochondrial evidence",
        section="Results",
        text=text,
        source_path=f"PMC{pmid}.xml",
        source_sha256="0" * 64,
    )


def test_sparse_tfidf_and_frozen_graph_have_honest_bounded_definitions():
    passages = [
        _passage(
            "p1",
            "Mitochondrial calcium uptake supports glucose stimulated insulin secretion in beta cells.",
        ),
        _passage(
            "p2",
            "Adjacent context reports mitochondrial calcium measurements and experimental limitations.",
        ),
        _passage(
            "p3",
            "Mitochondrial calcium signaling changes oxidative metabolism in pancreatic tissue.",
            pmid="2",
        ),
        _passage(
            "p4",
            "Unrelated microscopy acquisition settings and image reconstruction details are reported.",
            pmid="2",
        ),
    ]
    index = SparseTfidfIndex(passages)
    result_b = index.search(["mitochondrial calcium insulin secretion"], top_k=2)
    assert result_b.method == SPARSE_TFIDF_METHOD
    assert result_b.passages[0].passage_id == "p1"
    assert len(result_b.passages) == 2

    graph = FrozenEvidenceGraphRetriever(index, max_lexical_document_frequency=4)
    result_c = graph.search(["mitochondrial calcium insulin secretion"], top_k=1)
    assert result_c.method == FROZEN_GRAPH_METHOD
    assert result_c.graph_node_count == 4
    assert result_c.graph_edge_count >= 2
    assert result_c.expanded_candidate_count >= 1
    assert len(result_c.passages) == 1  # graph replaces evidence under the same budget
    assert result_c.construction_source == "frozen_corpus_text_and_metadata_only"


JATS = """<article><body><sec><title>Results</title>
<p>Mitochondrial calcium uptake supports glucose stimulated insulin secretion in pancreatic beta cells under the measured experimental conditions.</p>
<p>Adjacent experiments report mitochondrial calcium dynamics, oxidative metabolism, and explicit methodological limitations in pancreatic islets.</p>
<p>Independent observations describe insulin secretion and mitochondrial network morphology without establishing every proposed causal direction.</p>
</sec></body></article>"""


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    xml = root / "eval/data/corpus_raw/PMC1.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(JATS, encoding="utf-8")
    digest = hashlib.sha256(xml.read_bytes()).hexdigest()
    (root / "eval/data/evidence_pool_manifest.json").write_text(
        json.dumps(
            {
                "reviews": [{"pmid": "1", "title": "Fixture beta-cell review"}],
                "fulltext": [
                    {
                        "pmid": "1",
                        "pmcid": "PMC1",
                        "path": "eval/data/corpus_raw/PMC1.xml",
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class _Model:
    def __init__(self, *, fail_plan: bool = False):
        self.fail_plan = fail_plan
        self.plan_calls = 0
        self.direct_calls = 0
        self.grounded_calls = 0

    def plan(self, request):
        self.plan_calls += 1
        if self.fail_plan:
            raise RuntimeError("planned fixture failure")
        return (
            SearchPlan(
                queries=["mitochondrial calcium insulin secretion"],
                rationale="shared fixed retrieval plan",
                answerability_hint=Answerability.ANSWERABLE,
            ),
            ModelCallAudit(stage="plan", provider="fixture", model="fake-hy3"),
        )

    def synthesize_direct(self, request):
        self.direct_calls += 1
        return (
            GeneratedReview(
                answerability=Answerability.ANSWERABLE,
                answer="Direct model-memory baseline.",
                claims=[GeneratedClaim(claim_id="A1", text="Ungrounded direct claim.")],
                limitations=["no retrieval"],
            ),
            ModelCallAudit(stage="ablation_A_direct", provider="fixture", model="fake-hy3"),
        )

    def synthesize(self, request, passages):
        self.grounded_calls += 1
        assert passages
        first = passages[0].passage_id
        return (
            GeneratedReview(
                answerability=Answerability.ANSWERABLE,
                answer="Grounded fixture draft.",
                claims=[
                    GeneratedClaim(
                        claim_id="C1",
                        text="Supported mitochondrial calcium claim.",
                        evidence_passage_ids=[first],
                    ),
                    GeneratedClaim(
                        claim_id="C2",
                        text="Unsupported causal overstatement.",
                        evidence_passage_ids=[first],
                    ),
                ],
                limitations=[],
            ),
            ModelCallAudit(stage="synthesis", provider="fixture", model="fake-hy3"),
        )


class _Gate:
    k = 1

    def judge(self, claim, spans, *, question):
        assert question
        assert spans
        supported = claim.claim_id == "C1"
        verdict = (
            SupportVerdict.FULLY_SUPPORTED
            if supported
            else SupportVerdict.NOT_SUPPORTED
        )
        final = JudgeVerdict(
            claim_id=claim.claim_id,
            verdict=verdict,
            confidence=1.0,
            reason="fixture gate",
            evidence_span_refs=[spans[0].span_id] if supported else [],
        )
        return JudgeAggregate(
            claim_id=claim.claim_id,
            k=1,
            n_valid=1,
            votes={verdict.value: 1},
            final_verdict=verdict,
            final=final,
            agreement_rate=1.0,
            escalate_to_human=False,
        )


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "pilot.jsonl"
    path.write_text(
        json.dumps({"question_id": "Q1", "question": "How does mitochondrial calcium help?"})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_suite_runs_all_four_arms_with_fixed_plan_and_exact_c_parent(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    model = _Model()
    runner = PilotAblationRunner(
        model=model,
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    request = ReviewRequest(question_id="Q1", question="How does mitochondrial calcium help?")
    suite_dir, state = runner.run_suite(
        [request],
        replicates=2,
        out_root=tmp_path / "results",
        suite_id="fixture-abcd",
        input_path=_input_file(tmp_path),
    )
    assert state.status.value == "completed"
    assert state.expected_grid_cells == 8
    assert len(state.records) == 8
    assert all(record.outcome is CellOutcome.SUCCEEDED for record in state.records)
    assert model.plan_calls == 1  # one shared immutable B/C/D plan across replicates
    assert model.direct_calls == 2
    assert model.grounded_calls == 4  # B and C for each replicate; D reuses C

    b = AblationCellArtifact.model_validate_json(
        (suite_dir / "Q1/replicate-01/B/artifact.json").read_text(encoding="utf-8")
    )
    c = AblationCellArtifact.model_validate_json(
        (suite_dir / "Q1/replicate-01/C/artifact.json").read_text(encoding="utf-8")
    )
    d = AblationCellArtifact.model_validate_json(
        (suite_dir / "Q1/replicate-01/D/artifact.json").read_text(encoding="utf-8")
    )
    assert b.retrieval.method == SPARSE_TFIDF_METHOD
    assert c.retrieval.method == FROZEN_GRAPH_METHOD
    assert b.retrieval.expert_labels_used is False
    assert c.retrieval.expert_labels_used is False
    assert len(b.passages) <= 2 and len(c.passages) <= 2
    assert d.parent_c_artifact_sha256 is not None
    assert [p.passage_id for p in d.passages] == [p.passage_id for p in c.passages]
    assert [claim.claim_id for claim in d.review.claims] == ["C1"]
    assert [gate.passed for gate in d.claim_gates] == [True, False]
    assert "Judge" in d.review.answer
    assert json.loads((suite_dir / "suite_summary.json").read_text())["formal_status"] == (
        "pilot_ablation_generation_unscored"
    )


def test_planning_failure_is_retained_in_b_c_d_cells_while_a_still_runs(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(fail_plan=True),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    suite_dir, state = runner.run_suite(
        [
            ReviewRequest(
                question_id="Q1",
                question="How does mitochondrial calcium help?",
            )
        ],
        replicates=1,
        out_root=tmp_path / "results",
        suite_id="failure-grid",
        input_path=_input_file(tmp_path),
    )
    assert len(state.records) == state.expected_grid_cells == 4
    by_arm = {record.arm: record for record in state.records}
    assert by_arm[PilotArm.A].outcome is CellOutcome.SUCCEEDED
    assert all(
        by_arm[arm].outcome is CellOutcome.FAILED
        for arm in (PilotArm.B, PilotArm.C, PilotArm.D)
    )
    assert "planned fixture failure" in state.planning_failures["Q1"]
    assert (suite_dir / "Q1/replicate-01/B/failure.json").is_file()
    assert (suite_dir / "Q1/replicate-01/C/failure.json").is_file()
    assert (suite_dir / "Q1/replicate-01/D/failure.json").is_file()


def test_suite_rejects_answerability_or_gold_like_input_hints(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    request = ReviewRequest(
        question_id="Q1",
        question="question",
        answerability_hint=Answerability.ANSWERABLE,
        prohibited_inferences=["hidden gold boundary"],
    )
    with pytest.raises(ValueError, match="中性输入"):
        runner.run_suite(
            [request],
            replicates=1,
            out_root=tmp_path / "results",
            suite_id="leakage-attempt",
            input_path=_input_file(tmp_path),
        )


def test_ablation_suite_builds_four_arm_answerability_concordance_without_dropping_cells(
    tmp_path: Path,
):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    gold_row = next(
        json.loads(line)
        for line in (
            REPO_ROOT / "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if json.loads(line)["question_id"] == "PILOT-01"
    )
    input_path = tmp_path / "pilot01.jsonl"
    input_path.write_text(json.dumps(gold_row) + "\n", encoding="utf-8")
    suite_dir, _ = runner.run_suite(
        [
            ReviewRequest(
                question_id="PILOT-01",
                question=gold_row["question"],
                scope=gold_row["scope"],
            )
        ],
        replicates=1,
        out_root=tmp_path / "results",
        suite_id="gold-linked-abcd",
        input_path=input_path,
    )
    concordance = build_ablation_answerability_concordance(REPO_ROOT, suite_dir)
    assert len(concordance.nominal) == 4
    assert {row.task for row in concordance.nominal} == {
        "ablation_A_answerability",
        "ablation_B_answerability",
        "ablation_C_answerability",
        "ablation_D_answerability",
    }
    assert all(row.expert_label == "answerable" for row in concordance.nominal)
    assert all(row.automatic_label == "answerable" for row in concordance.nominal)
    result = analyze_expert_concordance(concordance)
    assert all(
        task["n_input"] == 1
        for task in result["nominal_tasks"].values()
    )


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.headers = {}
        self.payload = payload
        self.calls = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append(json)
        return _Response(self.payload)


def _direct_body(evidence_ids):
    arguments = {
        "answerability": "answerable",
        "answer": "direct baseline",
        "claims": [
            {
                "claim_id": "C1",
                "text": "direct claim",
                "is_core": True,
                "conditions": {
                    "species": None,
                    "cell_type": None,
                    "perturbation": None,
                    "dose": None,
                    "time": None,
                    "method": None,
                    "outcome": None,
                    "effect_direction": None,
                },
                "evidence_passage_ids": evidence_ids,
            }
        ],
        "limitations": ["no retrieval"],
    }
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {
                                "name": "emit_review",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_production_direct_method_forbids_invented_passage_evidence():
    clean_session = _Session(_direct_body([]))
    model = Hy3ReviewModel(
        api_key="dummy",
        base_url="https://example.invalid/v1",
        session=clean_session,
        sleep_fn=lambda _: None,
    )
    review, audit = model.synthesize_direct(
        ReviewRequest(question_id="Q", question="scientific question")
    )
    assert review.claims[0].evidence_passage_ids == []
    assert audit.stage == "ablation_A_direct"
    assert "没有外部检索" in clean_session.calls[0]["messages"][0]["content"]

    bad_session = _Session(_direct_body(["invented:p1"]))
    model = Hy3ReviewModel(
        api_key="dummy",
        base_url="https://example.invalid/v1",
        session=bad_session,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="不得声称 passage"):
        model.synthesize_direct(ReviewRequest(question_id="Q", question="question"))


def test_ablation_cli_has_no_offline_mode_and_fails_before_network_without_key():
    env = dict(os.environ)
    env.pop("HY3_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, "scripts/run_pilot_ablation.py", "--pilot-id", "PILOT-01"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "不提供离线伪实验模式" in completed.stderr
    help_run = subprocess.run(
        [sys.executable, "scripts/run_pilot_ablation.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "offline-smoke" not in help_run.stdout
