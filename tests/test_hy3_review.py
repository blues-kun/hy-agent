"""Hy3 应用模型的离线协议测试；所有响应均为 mock。"""
from __future__ import annotations

import json

from app.hy3_review import Hy3ReviewModel
from app.schemas import CorpusPassage, ReviewRequest


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class _Session:
    def __init__(self, bodies):
        self.headers = {}
        self.bodies = list(bodies)
        self.calls = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return _Response(self.bodies.pop(0))


def _tool_body(name, arguments):
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "not persisted",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


def test_hy3_plan_keeps_only_caller_supplied_pmids_and_synthesis_uses_evidence_ids():
    session = _Session(
        [
            _tool_body(
                "emit_search_plan",
                {
                    "queries": [" pancreatic beta cell mitochondria ", "pancreatic beta cell mitochondria"],
                    "source_pmids": ["123", "invented"],
                    "rationale": "retrieve evidence",
                    "answerability_hint": "answerable",
                },
            ),
            _tool_body(
                "emit_review",
                {
                    "answerability": "partial",
                    "answer": "Only the supplied passage was used.",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "The passage reports a mitochondrial observation.",
                            "is_core": True,
                            "conditions": {
                                "species": None,
                                "cell_type": "beta cell",
                                "perturbation": None,
                                "dose": None,
                                "time": None,
                                "method": None,
                                "outcome": "mitochondrial observation",
                                "effect_direction": "unknown",
                            },
                            "evidence_passage_ids": ["PMC1:p0001"],
                        }
                    ],
                    "limitations": ["review evidence only"],
                },
            ),
        ]
    )
    model = Hy3ReviewModel(
        api_key="dummy",
        base_url="https://example.invalid/v1",
        model="hy3",
        session=session,
        sleep_fn=lambda _: None,
    )
    request = ReviewRequest(question_id="Q1", question="test", source_pmids=["123"])
    plan, plan_audit = model.plan(request)
    assert plan.queries == ["pancreatic beta cell mitochondria"]
    assert plan.source_pmids == ["123"]
    passage = CorpusPassage(
        passage_id="PMC1:p0001",
        paper_id="PMID:123",
        pmid="123",
        pmcid="PMC1",
        text="A sufficiently long supplied evidence paragraph about beta cell mitochondria.",
        source_path="fixture.xml",
        source_sha256="0" * 64,
    )
    review, review_audit = model.synthesize(request, [passage])
    assert review.claims[0].evidence_passage_ids == ["PMC1:p0001"]
    assert plan_audit.parse_source == review_audit.parse_source == "tool_call"
    assert plan_audit.cached_tokens == 2
    assert plan_audit.provider == "tencent-tokenhub"
    assert plan_audit.model == "hy3"
    assert plan_audit.endpoint_origin == "https://example.invalid"
    assert len(plan_audit.prompt_sha256) == len(plan_audit.schema_sha256) == 64
    assert all(call["headers"]["X-Session-ID"].startswith("mitoevidence-review") for call in session.calls)
    # 思考内容只参与修复上下文，不进入应用审计对象。
    assert "reasoning_content" not in plan_audit.model_dump()
