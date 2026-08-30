"""Offline tests for the real-Hy3 A/B/C/D Pilot runner contracts."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ablation import (
    AblationCellArtifact,
    CellOutcome,
    GeneratorProvenance,
    Hy3ClaimGate,
    JudgeClaimProvenance,
    JudgeProvenanceIdentity,
    JudgeSampleBinding,
    PilotAblationRunner,
    PilotArm,
    SUITE_EVIDENCE_MANIFEST_COPY,
    SUITE_INPUT_SNAPSHOT_COPY,
    derive_generator_seed,
    derive_judge_claim_seed,
    generator_cache_namespace,
)
from app.corpus import FrozenReviewCorpus
from app.experiment_retrieval import (
    FROZEN_GRAPH_METHOD,
    SPARSE_TFIDF_METHOD,
    FrozenEvidenceGraphRetriever,
    SparseTfidfIndex,
)
from app.hy3_review import (
    GENERATOR_BASE_PROMPT_HASH_SCOPE,
    GENERATOR_OUTPUT_HASH_SCOPE,
    GENERATOR_PROMPT_HASH_SCOPE,
    GENERATOR_REASONING_EFFORT,
    GENERATOR_RESPONSE_HASH_SCOPE,
    Hy3ReviewModel,
    generator_base_messages_for_stage,
    generator_schema_for_stage,
)
from app.schemas import (
    CorpusPassage,
    GeneratedClaim,
    GeneratedReview,
    ModelCallAudit,
    ReviewRequest,
    SearchPlan,
)
from evaluator.judge import (
    JudgeAggregate,
    JudgeCallResult,
    JudgeSample,
    aggregate_samples,
)
from evaluator.judge.config import JudgeConfig, default_judge_config
from evaluator.experiment_protocol import (
    analyze_expert_concordance,
    build_ablation_answerability_concordance,
)
from evaluator.ablation_artifacts import audit_pilot_ablation_artifacts
from evaluator.artifact_security import ArtifactSecurityError, assert_json_safe
from evaluator.schemas import (
    Answerability,
    AtomicClaim,
    EvidenceSpan,
    JudgeVerdict,
    SourceAccess,
    SupportVerdict,
    TextAnchor,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_MODEL_IDENTITY = {
    "provider": "fixture",
    "model": "fake-hy3",
    "endpoint_origin": "https://fixture.invalid",
    "endpoint_url": "https://fixture.invalid/v1/chat/completions",
    "config_sha256": "0" * 64,
}


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

    @property
    def audit_identity(self):
        return {
            "execution_kind": "test_fixture",
            **FIXTURE_MODEL_IDENTITY,
        }

    @staticmethod
    def _audit(stage, request, passages, output, *, seed, cache_namespace):
        messages = generator_base_messages_for_stage(stage, request, passages)
        digest = lambda value: hashlib.sha256(  # noqa: E731 - compact fixture helper
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prompt_sha256 = digest(messages)
        return ModelCallAudit(
            stage=stage,
            **FIXTURE_MODEL_IDENTITY,
            prompt_sha256=prompt_sha256,
            base_prompt_sha256=prompt_sha256,
            base_prompt_hash_scope=GENERATOR_BASE_PROMPT_HASH_SCOPE,
            prompt_hash_scope=GENERATOR_PROMPT_HASH_SCOPE,
            schema_sha256=digest(generator_schema_for_stage(stage)),
            response_sha256="1" * 64,
            response_hash_scope=GENERATOR_RESPONSE_HASH_SCOPE,
            structured_output_sha256=digest(output.model_dump(mode="json")),
            structured_output_hash_scope=GENERATOR_OUTPUT_HASH_SCOPE,
            temperature=0.2,
            requested_seed=seed,
            cache_namespace=cache_namespace,
            attempt_count=1,
            reasoning_effort=GENERATOR_REASONING_EFFORT,
            max_tokens=8192 if stage == "synthesis" else 4096,
            parse_source="tool_call",
        )

    def plan(self, request, *, seed=None, cache_namespace="fixture"):
        self.plan_calls += 1
        if self.fail_plan:
            raise RuntimeError("planned fixture failure")
        plan = SearchPlan(
                queries=["mitochondrial calcium insulin secretion"],
                rationale="shared fixed retrieval plan",
                answerability_hint=Answerability.ANSWERABLE,
        )
        return plan, self._audit(
            "plan", request, [], plan, seed=seed, cache_namespace=cache_namespace
        )

    def synthesize_direct(self, request, *, seed=None, cache_namespace="fixture"):
        self.direct_calls += 1
        review = GeneratedReview(
                answerability=Answerability.ANSWERABLE,
                answer="Direct model-memory baseline.",
                claims=[GeneratedClaim(claim_id="A1", text="Ungrounded direct claim.")],
                limitations=["no retrieval"],
        )
        return review, self._audit(
            "ablation_A_direct",
            request,
            [],
            review,
            seed=seed,
            cache_namespace=cache_namespace,
        )

    def synthesize(self, request, passages, *, seed=None, cache_namespace="fixture"):
        self.grounded_calls += 1
        assert passages
        first = passages[0].passage_id
        review = GeneratedReview(
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
        )
        return review, self._audit(
            "synthesis",
            request,
            passages,
            review,
            seed=seed,
            cache_namespace=cache_namespace,
        )


class _OutOfScopeModel(_Model):
    def synthesize(self, request, passages, *, seed=None, cache_namespace="fixture"):
        self.grounded_calls += 1
        review = GeneratedReview(
                answerability=Answerability.OUT_OF_SCOPE,
                answer="Refused at the research boundary.",
                claims=[],
                limitations=["out of scope"],
        )
        return review, self._audit(
            "synthesis",
            request,
            passages,
            review,
            seed=seed,
            cache_namespace=cache_namespace,
        )


class _LeakyPlanningModel(_Model):
    def plan(self, request, *, seed=None, cache_namespace="fixture"):
        raise RuntimeError(
            "HY3_API_KEY=" + "sk-" + "abcdefghijklmnopqrstuvwx "
            "Authorization: " + "Bearer " + "bearer-secret-abcdefghijklmnopqrstuvwxyz "
            "https://alice:password@example.invalid/v1?q=secret"
        )


class _Gate:
    k = 1

    def __init__(self):
        self._identity = JudgeProvenanceIdentity(
            execution_kind="test_fixture",
            provider="test-fixture",
            model="deterministic-test-gate",
            endpoint_origin="test://local",
            endpoint_url="test://local/chat/completions",
            config_sha256="1" * 64,
            config_hash_scope="source_file_bytes",
            schema_sha256="2" * 64,
            prompt_template_sha256="3" * 64,
            structured_output_channel="function_calling",
            k=1,
            temperature=0.7,
            base_seed=100,
            min_agreement_votes=1,
            escalate_on_refuted=True,
        )
        self._last_call = None

    @property
    def provenance_identity(self):
        return self._identity

    @property
    def last_call_provenance(self):
        return self._last_call

    def judge(self, claim, spans, *, question, base_seed_override=None):
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
        effective_seed = 100 if base_seed_override is None else base_seed_override
        sample = JudgeSample(
            index=0,
            ok=True,
            verdict=final,
            parse_source="test_fixture",
            response_sha256=hashlib.sha256(
                f"fixture:{claim.claim_id}".encode()
            ).hexdigest(),
            temperature=0.7,
            seed=effective_seed,
        )
        aggregate = aggregate_samples(
            claim.claim_id,
            [sample],
            k=1,
            min_agreement_votes=1,
            escalate_on_refuted=True,
        )
        self._last_call = JudgeClaimProvenance(
            claim_id=claim.claim_id,
            prompt_sha256=hashlib.sha256(
                f"fixture-prompt:{claim.claim_id}".encode()
            ).hexdigest(),
            derived_base_seed=effective_seed,
            samples=[JudgeSampleBinding.from_sample(sample)],
        )
        return aggregate


class _ProvenanceJudgeClient:
    def __init__(self, config):
        self.config = config
        self.model = "hy3-provenance-fixture"
        self.channel = "function_calling"
        self.transport = SimpleNamespace(
            base_url="https://judge.example.invalid/v1"
        )

    def judge_once(
        self,
        claim,
        spans,
        *,
        question,
        temperature,
        seed,
    ):
        assert question == "fixture question"
        verdict = JudgeVerdict(
            claim_id=claim.claim_id,
            verdict=SupportVerdict.FULLY_SUPPORTED,
            confidence=1.0,
            reason="production gate provenance fixture",
            evidence_span_refs=[spans[0].span_id],
        )
        return JudgeCallResult(
            ok=True,
            verdict=verdict,
            parse_source="tool_call",
            response_sha256=hashlib.sha256(f"response:{seed}".encode()).hexdigest(),
            temperature=temperature,
            seed=seed,
        )


def test_hy3_claim_gate_records_identity_prompt_and_every_sample_binding():
    config = default_judge_config()
    client = _ProvenanceJudgeClient(config)
    gate = Hy3ClaimGate(
        client,
        config=config,
        k=2,
        temperature=0.8,
        base_seed=300,
    )
    claim = AtomicClaim(claim_id="C1", text="fixture claim")
    span = EvidenceSpan(
        span_id="S1",
        paper_id="P1",
        doi_or_pmid="PMID:1",
        anchor=TextAnchor(exact="fixture evidence text"),
        source_access=SourceAccess.FULLTEXT,
    )

    aggregate = gate.judge(
        claim,
        [span],
        question="fixture question",
    )
    identity = gate.provenance_identity
    call = gate.last_call_provenance
    assert identity.provider == "tencent-tokenhub"
    assert identity.model == "hy3-provenance-fixture"
    assert identity.endpoint_origin == "https://judge.example.invalid"
    assert identity.endpoint_url == "https://judge.example.invalid/v1/chat/completions"
    assert identity.config_sha256 == config.sha256
    assert len(identity.schema_sha256) == 64
    assert len(identity.prompt_template_sha256) == 64
    assert (identity.k, identity.temperature, identity.base_seed) == (2, 0.8, 300)
    assert identity.min_agreement_votes == 2
    assert identity.escalate_on_refuted is True
    assert call is not None
    assert call.claim_id == "C1"
    assert call.derived_base_seed == 300
    assert call.prompt_hash_scope == "base_messages_before_repair"
    assert len(call.prompt_sha256) == 64
    assert [sample.seed for sample in aggregate.samples] == [300, 301]
    assert call.samples == [
        JudgeSampleBinding.from_sample(sample) for sample in aggregate.samples
    ]


def test_hy3_claim_gate_rejects_config_different_from_actual_client_config():
    config = default_judge_config()
    client = _ProvenanceJudgeClient(config)
    mismatched = JudgeConfig(config.raw, sha256="f" * 64)
    with pytest.raises(ValueError, match="client.config"):
        Hy3ClaimGate(client, config=mismatched, k=1)


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://alice:password@example.invalid/v1/chat/completions",
        "https://example.invalid/v1/chat/completions?api_key=secret",
    ],
)
def test_generator_provenance_rejects_endpoint_credentials_or_query(endpoint_url: str):
    with pytest.raises(ValueError, match="endpoint_url"):
        GeneratorProvenance(
            execution_kind="test_fixture",
            provider="fixture",
            model="fake",
            endpoint_origin="https://example.invalid",
            endpoint_url=endpoint_url,
            config_sha256="0" * 64,
            base_seed=1,
            cache_namespace="mitoevidence-fixture",
        )


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "pilot.jsonl"
    path.write_text(
        json.dumps({"question_id": "Q1", "question": "How does mitochondrial calcium help?"})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_suite_state_accepts_a_long_repo_relative_input_path(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    input_path = (
        root
        / "annotation_prelabel"
        / "pilot_questions"
        / "pilot_5_questions.jsonl"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps(
            {
                "question_id": "Q1",
                "question": "How does mitochondrial calcium help?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )

    _, state = runner.run_suite(
        [
            ReviewRequest(
                question_id="Q1",
                question="How does mitochondrial calcium help?",
            )
        ],
        replicates=1,
        out_root=tmp_path / "results",
        suite_id="long-input-path-v3",
        input_path=input_path,
    )

    assert state.input_snapshot.path == (
        "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
    )


def test_path_field_still_rejects_a_single_opaque_credential_segment():
    payload = {
        "path": (
            "annotation_prelabel/"
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            "/pilot.jsonl"
        )
    }

    with pytest.raises(ArtifactSecurityError, match="credential-like text"):
        assert_json_safe(payload)


def test_path_field_still_rejects_a_known_api_key(monkeypatch: pytest.MonkeyPatch):
    secret = "hy3-test-credential-material-that-must-never-be-persisted"
    monkeypatch.setenv("HY3_API_KEY", secret)

    with pytest.raises(ArtifactSecurityError, match="credential-like text"):
        assert_json_safe({"path": f"annotation_prelabel/{secret}/pilot.jsonl"})


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
    assert state.schema_version == "mitoevidence.pilot-ablation.v3"
    assert state.generator_provenance is not None
    assert state.generator_provenance.base_seed == 20260831
    assert state.generator_provenance.endpoint_url == "https://fixture.invalid/v1/chat/completions"
    assert state.judge_provenance_identity is not None
    assert state.judge_provenance_identity.base_seed == 100
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
    assert d.judge_provenance is not None
    assert d.judge_provenance.execution_kind == "test_fixture"
    assert d.judge_provenance.execution_status == "test_fixture_invoked"
    assert d.judge_provenance.provider == "test-fixture"
    assert d.judge_provenance.model == "deterministic-test-gate"
    assert d.judge_provenance.k == 1
    assert d.judge_provenance.temperature == 0.7
    assert d.judge_provenance.base_seed == 100
    assert [call.claim_id for call in d.judge_provenance.calls] == ["C1", "C2"]
    assert [call.derived_base_seed for call in d.judge_provenance.calls] == [
        derive_judge_claim_seed(100, "Q1", 1, "C1"),
        derive_judge_claim_seed(100, "Q1", 1, "C2"),
    ]
    assert [
        gate.aggregate.samples[0].seed for gate in d.claim_gates
    ] == [call.derived_base_seed for call in d.judge_provenance.calls]
    assert all(call.samples[0].sample_sha256 for call in d.judge_provenance.calls)
    assert "Judge" in d.review.answer
    assert json.loads((suite_dir / "suite_summary.json").read_text())["formal_status"] == (
        "pilot_ablation_generation_unscored"
    )
    assert SUITE_INPUT_SNAPSHOT_COPY == "pilot_input_snapshot.jsonl"
    assert SUITE_EVIDENCE_MANIFEST_COPY == "evidence_manifest_snapshot.json"
    assert (suite_dir / "suite_state.json").read_bytes() == (
        suite_dir / "suite_summary.json"
    ).read_bytes()
    assert (suite_dir / SUITE_INPUT_SNAPSHOT_COPY).read_bytes() == _input_file(tmp_path).read_bytes()
    assert (suite_dir / SUITE_EVIDENCE_MANIFEST_COPY).read_bytes() == (
        root / "eval/data/evidence_pool_manifest.json"
    ).read_bytes()
    audit = audit_pilot_ablation_artifacts(suite_dir, allow_test_fixture=True)
    assert audit["structural_audit_ok"] is True
    assert audit["legacy_structural_only"] is False
    assert audit["test_fixture_audit_ok"] is True
    assert audit["production_ready"] is False
    assert all(
        check["ok"]
        for check in audit["cross_arm_checks"]
        if check["check"] == "GENERATOR_V3_PROMPT_SCHEMA_SEED_BINDING"
    )


def test_v3_artifact_audit_requires_byte_bound_journals_and_archived_snapshots(
    tmp_path: Path,
):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    base_suite, _ = runner.run_suite(
        [ReviewRequest(question_id="Q1", question="How does mitochondrial calcium help?")],
        replicates=1,
        out_root=tmp_path / "results",
        suite_id="top-level-bindings-v3",
        input_path=_input_file(tmp_path),
    )
    cases = (
        ("suite_summary.json", "delete", "SUITE_SUMMARY_MISSING_OR_NONREGULAR"),
        ("suite_summary.json", "tamper", "SUITE_STATE_SUMMARY_BYTES_MISMATCH"),
        ("suite_summary.json", "symlink", "SUITE_SUMMARY_SYMLINK_FORBIDDEN"),
        (
            SUITE_INPUT_SNAPSHOT_COPY,
            "delete",
            "PILOT_INPUT_SNAPSHOT_MISSING_OR_NONREGULAR",
        ),
        (
            SUITE_INPUT_SNAPSHOT_COPY,
            "tamper",
            "PILOT_INPUT_SNAPSHOT_HASH_MISMATCH",
        ),
        (
            SUITE_INPUT_SNAPSHOT_COPY,
            "symlink",
            "PILOT_INPUT_SNAPSHOT_SYMLINK_FORBIDDEN",
        ),
        (
            SUITE_EVIDENCE_MANIFEST_COPY,
            "delete",
            "EVIDENCE_MANIFEST_SNAPSHOT_MISSING_OR_NONREGULAR",
        ),
        (
            SUITE_EVIDENCE_MANIFEST_COPY,
            "tamper",
            "EVIDENCE_MANIFEST_SNAPSHOT_HASH_MISMATCH",
        ),
        (
            SUITE_EVIDENCE_MANIFEST_COPY,
            "symlink",
            "EVIDENCE_MANIFEST_SNAPSHOT_SYMLINK_FORBIDDEN",
        ),
    )
    for index, (filename, mutation, expected_code) in enumerate(cases):
        candidate = tmp_path / f"binding-case-{index}"
        shutil.copytree(base_suite, candidate)
        target = candidate / filename
        if mutation == "delete":
            target.unlink()
        elif mutation == "tamper":
            target.write_bytes(target.read_bytes() + b" ")
        else:
            external = tmp_path / f"external-{index}"
            external.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(external)

        result = audit_pilot_ablation_artifacts(
            candidate,
            allow_test_fixture=True,
        )
        assert result["artifact_integrity_ok"] is False
        assert result["test_fixture_audit_ok"] is False
        assert result["production_ready"] is False
        assert result["suite_binding_ok"] is False
        assert result["top_level_files"]["ok"] is False
        assert any(error["code"] == expected_code for error in result["errors"])


def test_v3_artifact_audit_rebuilds_prompt_instead_of_trusting_hash(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    suite_dir, state = runner.run_suite(
        [ReviewRequest(question_id="Q1", question="How does mitochondrial calcium help?")],
        replicates=1,
        out_root=tmp_path / "results",
        suite_id="prompt-tamper-v3",
        input_path=_input_file(tmp_path),
    )
    record = next(
        row for row in state.records if row.arm is PilotArm.B and row.replicate == 1
    )
    cell = suite_dir / record.cell_dir
    artifact_path = cell / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["model_calls"][1]["prompt_sha256"] = "f" * 64
    json_bytes = lambda value: (  # noqa: E731 - exact runtime serializer fixture
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    artifact_bytes = json_bytes(artifact)
    artifact_path.write_bytes(artifact_bytes)
    manifest_path = cell / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["artifact.json"] = {
        "bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    manifest_bytes = json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    state_payload = state.model_dump(mode="json")
    for row in state_payload["records"]:
        if row["cell_dir"] == record.cell_dir:
            row["cell_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    state_bytes = json_bytes(state_payload)
    (suite_dir / "suite_state.json").write_bytes(state_bytes)
    (suite_dir / "suite_summary.json").write_bytes(state_bytes)

    result = audit_pilot_ablation_artifacts(suite_dir, allow_test_fixture=True)
    assert result["structural_audit_ok"] is False
    assert any(
        error["code"] == "GENERATOR_V3_PROMPT_SCHEMA_SEED_BINDING"
        for error in result["errors"]
    )


def test_runner_refuses_unsafe_resume_before_touching_an_existing_suite(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_Model(),
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    out_root = tmp_path / "results"
    with pytest.raises(RuntimeError, match="严格 --resume 尚未实现"):
        runner.run_suite(
            [ReviewRequest(question_id="Q1", question="question")],
            replicates=1,
            out_root=out_root,
            suite_id="must-not-be-created",
            input_path=_input_file(tmp_path),
            resume=True,
        )
    assert not out_root.exists()


def test_zero_claim_d_records_identity_without_claiming_a_judge_request(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    model = _OutOfScopeModel()
    runner = PilotAblationRunner(
        model=model,
        corpus=FrozenReviewCorpus(root),
        claim_gate=_Gate(),
        top_k=2,
    )
    request = ReviewRequest(
        question_id="Q1",
        question="Personalized clinical dose?",
    )
    plan_seed = derive_generator_seed(
        runner.generator_provenance.base_seed,
        request.question_id,
        0,
        "shared",
        "plan",
    )
    plan_namespace = generator_cache_namespace(
        runner.generator_provenance.cache_namespace,
        request.question_id,
        0,
        "shared",
        "plan",
    )
    plan, plan_audit = model.plan(
        request,
        seed=plan_seed,
        cache_namespace=plan_namespace,
    )
    c_artifact = runner.run_c(request, 1, plan, plan_audit)
    d_artifact = runner.run_d(c_artifact)
    assert d_artifact.claim_gates == []
    assert d_artifact.judge_provenance is not None
    assert d_artifact.judge_provenance.calls == []
    assert d_artifact.judge_provenance.execution_status == "no_claims_no_request"
    assert d_artifact.judge_provenance.prompt_template_sha256


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


def test_planning_failure_is_redacted_before_any_suite_file_is_written(tmp_path: Path):
    root = _fixture_repo(tmp_path)
    runner = PilotAblationRunner(
        model=_LeakyPlanningModel(),
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
        suite_id="redacted-failure-grid",
        input_path=_input_file(tmp_path),
    )
    rendered = (suite_dir / "suite_state.json").read_text(encoding="utf-8")
    for secret in (
        "sk-" + "abcdefghijklmnopqrstuvwx",
        "bearer-secret-abcdefghijklmnopqrstuvwxyz",
        "alice",
        "password",
        "q=secret",
    ):
        assert secret not in rendered
        assert secret not in state.planning_failures["Q1"]
    assert "[REDACTED]" in state.planning_failures["Q1"]
    for arm in ("B", "C", "D"):
        failure = json.loads(
            (suite_dir / f"Q1/replicate-01/{arm}/failure.json").read_text(
                encoding="utf-8"
            )
        )
        assert failure["security"]["failure_text_sanitized"] is True
        assert failure["security"]["contains_reasoning_content"] is False
    audit = audit_pilot_ablation_artifacts(suite_dir, allow_test_fixture=True)
    assert audit["test_fixture_audit_ok"] is True
    assert audit["records"]["failed"] == 3


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
    with pytest.raises(ValueError, match="expert manifest"):
        build_ablation_answerability_concordance(REPO_ROOT, suite_dir)
    original_journal = (suite_dir / "suite_state.json").read_bytes()
    original_input_copy = (suite_dir / SUITE_INPUT_SNAPSHOT_COPY).read_bytes()
    pinned_input = (
        REPO_ROOT / "annotation_prelabel/pilot_questions/pilot_5_questions.jsonl"
    ).read_bytes()
    state_payload = json.loads(original_journal)
    state_payload["input_snapshot"]["sha256"] = hashlib.sha256(pinned_input).hexdigest()
    pinned_state = (
        json.dumps(state_payload, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    (suite_dir / SUITE_INPUT_SNAPSHOT_COPY).write_bytes(pinned_input)
    (suite_dir / "suite_state.json").write_bytes(pinned_state)
    (suite_dir / "suite_summary.json").write_bytes(pinned_state)
    with pytest.raises(ValueError, match="evidence manifest hash"):
        build_ablation_answerability_concordance(REPO_ROOT, suite_dir)
    (suite_dir / SUITE_INPUT_SNAPSHOT_COPY).write_bytes(original_input_copy)
    (suite_dir / "suite_state.json").write_bytes(original_journal)
    (suite_dir / "suite_summary.json").write_bytes(original_journal)
    concordance = build_ablation_answerability_concordance(
        REPO_ROOT,
        suite_dir,
        allow_nonformal=True,
    )
    assert concordance.automatic_system_role.startswith("nonformal_")
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

    state_bytes = (suite_dir / "suite_state.json").read_bytes()
    summary_path = suite_dir / "suite_summary.json"
    summary_path.write_bytes(state_bytes + b" ")
    with pytest.raises(ValueError, match="逐字节一致"):
        build_ablation_answerability_concordance(
            REPO_ROOT, suite_dir, allow_nonformal=True
        )
    summary_path.write_bytes(state_bytes)

    input_copy = suite_dir / SUITE_INPUT_SNAPSHOT_COPY
    original_input = input_copy.read_bytes()
    input_copy.write_bytes(original_input + b" ")
    with pytest.raises(ValueError, match="input snapshot SHA-256"):
        build_ablation_answerability_concordance(
            REPO_ROOT, suite_dir, allow_nonformal=True
        )
    input_copy.write_bytes(original_input)

    evidence_copy = suite_dir / SUITE_EVIDENCE_MANIFEST_COPY
    original_evidence = evidence_copy.read_bytes()
    evidence_copy.write_bytes(original_evidence + b" ")
    with pytest.raises(ValueError, match="evidence manifest SHA-256"):
        build_ablation_answerability_concordance(
            REPO_ROOT, suite_dir, allow_nonformal=True
        )
    evidence_copy.write_bytes(original_evidence)


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
    assert "out_of_scope 且 claims 必须为空数组" in clean_session.calls[0]["messages"][0]["content"]

    bad_session = _Session(_direct_body(["invented:p1"]))
    model = Hy3ReviewModel(
        api_key="dummy",
        base_url="https://example.invalid/v1",
        session=bad_session,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="不得声称 passage"):
        model.synthesize_direct(ReviewRequest(question_id="Q", question="question"))


def test_ablation_cli_has_no_offline_mode_and_fails_before_network_without_key(
    tmp_path: Path,
):
    env = dict(os.environ)
    env.pop("HY3_API_KEY", None)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_pilot_ablation.py",
            "--pilot-id",
            "PILOT-01",
            "--judge-base-seed",
            "31001",
        ],
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
    missing_seed = subprocess.run(
        [sys.executable, "scripts/run_pilot_ablation.py", "--pilot-id", "PILOT-01"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert "formal v3 要求显式 --judge-base-seed" in missing_seed.stderr
    custom_input = tmp_path / "custom.jsonl"
    custom_input.write_text(
        json.dumps({"question_id": "PILOT-01", "question": "tampered"}) + "\n",
        encoding="utf-8",
    )
    unbound = subprocess.run(
        [
            sys.executable,
            "scripts/run_pilot_ablation.py",
            "--pilot-file",
            str(custom_input),
            "--pilot-id",
            "PILOT-01",
            "--judge-base-seed",
            "31001",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert "必须精确绑定 expert manifest" in unbound.stderr
    assert "缺少 HY3_API_KEY" not in unbound.stderr
    assert "--resume" in help_run.stdout
    resume_run = subprocess.run(
        [sys.executable, "scripts/run_pilot_ablation.py", "--resume"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resume_run.returncode != 0
    assert "拒绝 --resume" in resume_run.stderr
    assert "缺少 HY3_API_KEY" not in resume_run.stderr


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("HY3_MODEL", "not-hy3"),
        ("HY3_BASE_URL", "https://evil.example/v1"),
    ],
)
def test_ablation_cli_rejects_nonformal_generator_judge_environment_identity(
    env_name: str,
    env_value: str,
):
    env = dict(os.environ)
    env["HY3_API_KEY"] = "test-only-key"
    env["HY3_MODEL"] = "hy3"
    env.pop("HY3_BASE_URL", None)
    env[env_name] = env_value
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_pilot_ablation.py",
            "--pilot-id",
            "PILOT-01",
            "--judge-base-seed",
            "31001",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Generator/Judge 身份未通过共享 allowlist" in completed.stderr
