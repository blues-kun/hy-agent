"""No large model weights, GPU jobs, inference APIs or production data writes."""
from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rp_grpo.algorithms import (clipped_surrogate, loo_advantages,
    original_grpo_advantages, paired_utility, rp_grpo_advantages, token_policy_loss)
from rp_grpo.train_policy import (HFPolicy, append_observation, canonical_hash,
    canonical_json, guard_action, initial_messages, load_tasks, paired_tasks,
    parse_action, public_episode, rollout_policy, select_training_groups,
    selected_logprobs, sha256_file, update_from_episodes, validate_sft_gate)


@pytest.mark.parametrize("eta", [0.0, 0.2, 0.5, 1.0])
def test_utility_identity_and_symmetry(eta):
    for a, b in itertools.product([-1, -0.3, 0, 0.8, 1], repeat=2):
        assert paired_utility(a, b, eta) == pytest.approx((a+b)/2-eta*abs(a-b)/2)
        assert paired_utility(a, b, eta) == paired_utility(b, a, eta)


def test_eta_zero_exact_loo_and_real_rollout_count():
    r = {"retrievable": [0.2, 1, -1, 0.4], "needs_measurement": [0.6, -0.1, 0.2, 1]}
    result = rp_grpo_advantages(r, eta=0)
    for side in r:
        assert result["advantages"][side] == pytest.approx(loo_advantages(r[side]))
    assert result["real_rollouts"] == 8
    assert result["utility_operations"] == 16


def test_binary_coefficients_and_std_cancels_pair_effect():
    r = {"strong": [1, 1, 1, 0], "weak": [1, 0, 0, 0]}
    result = rp_grpo_advantages(r, eta=0.5)
    for side, slope in (("strong", 0.75), ("weak", 1.25)):
        assert result["advantages"][side] == pytest.approx([slope*a for a in loo_advantages(r[side])])
        # The discarded design: a within-side std would remove the slope.
        assert original_grpo_advantages(result["conditional_utilities"][side], 1e-14) == pytest.approx(
            original_grpo_advantages(r[side], 1e-14), abs=1e-10)


def test_exact_enumeration_shared_partner_loo_gradient():
    p0, p1, eta, g = 0.75, 0.25, 0.5, 2
    gradient = [0.0, 0.0]
    for bits in itertools.product([0, 1], repeat=2*g):
        left, right = list(bits[:g]), list(bits[g:])
        probability = math.prod((p0 if r else 1-p0) for r in left) * math.prod((p1 if r else 1-p1) for r in right)
        advantages = rp_grpo_advantages({"0": left, "1": right}, eta)["advantages"]
        for side, values, p in ((0, left, p0), (1, right, p1)):
            gradient[side] += probability * sum((r-p)*a for r, a in zip(values, advantages[str(side)]))/(2*g)
    # d E[U] / d(logit(p_side)); both shared-parameter terms are needed.
    assert gradient == pytest.approx([((1-eta)/2+eta*p1)*p0*(1-p0),
                                     ((1-eta)/2+eta*p0)*p1*(1-p1)])


def test_side_permutation_and_flat_group():
    first = rp_grpo_advantages({"a": [1, 1], "b": [0, 1]}, 0.5)
    second = rp_grpo_advantages({"b": [0, 1], "a": [1, 1]}, 0.5)
    assert first["advantages"] == second["advantages"]
    assert first["advantages"]["a"] == [0, 0]
    assert original_grpo_advantages([-1, -1]) == [0, 0]


@pytest.mark.parametrize("groups,eta", [({}, .5), ({"a": [0, 1]}, .5),
    ({"a": [0, 1], "b": [0]}, .5), ({"a": [0, 1], "b": [0, 1, 1]}, .5),
    ({"a": [float("nan"), 1], "b": [0, 1]}, .5),
    ({"a": [float("inf"), 1], "b": [0, 1]}, .5),
    ({"a": [True, 1], "b": [0, 1]}, .5), ({"a": [2, 1], "b": [0, 1]}, .5),
    ({"a": [0, 1], "b": [0, 1]}, -1), ({"a": [0, 1], "b": [0, 1]}, float("nan")),
    ({"a": [0, 1], "b": [0, 1]}, 1.1)])
def test_invalid_groups_and_rewards_rejected(groups, eta):
    with pytest.raises(ValueError):
        rp_grpo_advantages(groups, eta)


def test_ppo_clipping_depends_on_advantage_sign():
    assert clipped_surrogate(2, 1) == pytest.approx(1.2)
    assert clipped_surrogate(2, -1) == -2
    assert clipped_surrogate(.1, -1) == pytest.approx(-.8)
    assert clipped_surrogate(.1, 1) == .1


def test_token_mask_and_detached_behavior_reference():
    torch = pytest.importorskip("torch")
    new = torch.tensor([-.2, -.4, -.7, -.8], requires_grad=True)
    old = new.detach().clone().requires_grad_()
    reference = new.detach().clone().requires_grad_()
    loss = token_policy_loss(new, old, reference, torch.tensor([0, 1, 0, 1]), 0.5)
    loss.backward()
    assert new.grad[0] == 0 and new.grad[2] == 0
    assert new.grad[1] != 0 and new.grad[3] != 0
    assert old.grad is None and reference.grad is None
    with pytest.raises(ValueError, match="No policy"):
        token_policy_loss(new, old, reference, torch.zeros(4), 1)


def _tiny_model():
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 8)
        @property
        def device(self):
            return self.embedding.weight.device
        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
            logits = self.embedding(input_ids)
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)
    torch.manual_seed(9)
    return Tiny()


def test_selected_likelihood_excludes_prompt_and_update_changes_weights():
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    ids, outputs = [1, 2, 3], [4, 0]
    expected = -torch.nn.functional.cross_entropy(model.embedding(torch.tensor(ids+outputs))[-3:-1],
                                                 torch.tensor(outputs), reduction="none")
    assert torch.allclose(selected_logprobs(model, ids, outputs), expected)
    groups = []
    for _ in range(2):
        group = []
        for reward, completion in ((0., [4]), (1., [5])):
            probabilities = selected_logprobs(model, ids, completion).detach().tolist()
            group.append({"reward": reward, "loss_eligible": True, "steps": [{"tokens": {
                "prompt_ids": ids, "completion_ids": completion, "old_logprobs": probabilities,
                "reference_logprobs": probabilities}}]})
        groups.append(group)
    before = model.embedding.weight.detach().clone()
    metrics = update_from_episodes(HFPolicy(model, None), torch.optim.AdamW(model.parameters(), lr=.01),
                                    groups, arm="B3", eta=.5, epsilon=.2, beta=.04)
    assert metrics["real_rollouts"] == 4 and metrics["updated_episodes"] == 4
    assert not torch.equal(before, model.embedding.weight)


def test_real_hf_peft_sampling_ref_adapter_and_cpu_only(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    config = transformers.Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, eos_token_id=2, pad_token_id=2, attention_dropout=0.0)
    model = peft.get_peft_model(transformers.Qwen3ForCausalLM(config),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                        task_type="CAUSAL_LM", lora_dropout=0.0), adapter_name="policy")
    # A real local adapter checkpoint/reference is used, not a fake loss stub.
    model.save_pretrained(tmp_path / "snapshot", selected_adapters=["policy"])
    model.load_adapter(tmp_path / "snapshot/policy", adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    class Tokenizer:
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"
        def __call__(self, prompt, **kwargs):
            return {"input_ids": [1, 3, 4]}
        def decode(self, values, **kwargs):
            return str(values)
    policy = HFPolicy(model, Tokenizer(), reference_adapter="reference")
    one = policy.sample([], seed=111, max_new_tokens=3, max_context_tokens=16,
                        deadline=time.monotonic()+30, sample=True, with_logprobs=True)
    two = policy.sample([], seed=111, max_new_tokens=3, max_context_tokens=16,
                        deadline=time.monotonic()+30, sample=True, with_logprobs=True)
    assert model.device.type == "cpu"
    assert one["completion_ids"] == two["completion_ids"]
    assert one["old_logprobs"] == pytest.approx(one["reference_logprobs"])
    assert model.active_adapter == "policy"
    assert all(not p.requires_grad for name, p in model.named_parameters() if ".reference." in name)
    assert any(p.requires_grad for name, p in model.named_parameters() if ".policy." in name)
    from rp_grpo.train_policy import configure_gradient_checkpointing
    assert configure_gradient_checkpointing(model)["use_reentrant"] is False
    reference_before = {name: p.detach().clone() for name, p in model.named_parameters() if ".reference." in name}
    policy_before = {name: p.detach().clone() for name, p in model.named_parameters() if ".policy." in name}
    groups = []
    for _ in range(2):
        group = []
        for reward, completion in ((0., [4]), (1., [5])):
            old = selected_logprobs(model, [1, 3], completion).detach().tolist()
            group.append({"reward": reward, "loss_eligible": True, "steps": [{"tokens": {
                "prompt_ids": [1, 3], "completion_ids": completion,
                "old_logprobs": old, "reference_logprobs": old}}]})
        groups.append(group)
    update_from_episodes(policy, torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01),
                         groups, arm="B3", eta=.5, epsilon=.2, beta=.04)
    assert model.is_gradient_checkpointing
    assert all(torch.equal(reference_before[name], p) for name, p in model.named_parameters() if name in reference_before)
    assert any(not torch.equal(policy_before[name], p) for name, p in model.named_parameters() if name in policy_before)
    assert all(not module.training for module in model.modules() if isinstance(module, torch.nn.Dropout))
    model.config.attention_dropout = .1
    with pytest.raises(ValueError, match="attention dropout"):
        configure_gradient_checkpointing(model)


class FakePolicy:
    def __init__(self, texts, terminated=True):
        self.texts = list(texts)
        self.calls = []
        self.terminated = terminated
    def sample(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        text = self.texts.pop(0)
        return {"text": text, "prompt_ids": [1, 2], "completion_ids": [3, 4],
                "terminated": self.terminated, "reason": "eos" if self.terminated else "generation_truncated",
                "prompt_sha256": canonical_hash(messages)}


class FakeEnvironment:
    calls = []
    def reset(self, task):
        return {"question": "Read the actual evidence"}
    def step(self, action):
        self.calls.append(action)
        done = action["tool"] == "finalize"
        return {"observation": {"status": "ok", "text": "actual-tool-bytes"},
                "done": done, "reward": .8 if done else None,
                "info": {"hidden_reward_audit": "MUST_NOT_ENTER_PROMPT"}}
    def export_trace(self):
        return {"actions": copy.deepcopy(self.calls)}


def _rollout(policy, **kwargs):
    return rollout_policy(policy, FakeEnvironment, {"id": "x", "split": "train", "side": "a", "pair_id": "p"},
        seed=7, max_turns=4, max_new_tokens=16, max_context_tokens=1024,
        deadline=time.monotonic()+20, **kwargs)


def test_actual_tool_loop_and_observation_mask_contract():
    FakeEnvironment.calls = []
    policy = FakePolicy([canonical_json({"tool": "read_passage", "arguments": {"evidence_id": "x"}}),
                         canonical_json({"tool": "finalize", "arguments": {}})])
    episode = _rollout(policy)
    assert episode["done"] and episode["reward"] == .8
    assert len(FakeEnvironment.calls) == 2
    assert "actual-tool-bytes" in canonical_json(policy.calls[1])
    assert "MUST_NOT_ENTER_PROMPT" not in canonical_json(policy.calls[1])
    assert "side" not in policy.calls[0][-1]["content"]
    public = public_episode(episode)
    assert public["steps"][0]["completion_ids"] == [3, 4]


@pytest.mark.parametrize("text,terminated", [('{}', True), ('{"tool":"x","arguments":', False),
                                          ('{"tool":"x","arguments":{},"tool":"y"}', True)])
def test_invalid_or_truncated_action_never_dispatches(text, terminated):
    FakeEnvironment.calls = []
    episode = _rollout(FakePolicy([text], terminated=terminated))
    assert not FakeEnvironment.calls and not episode["done"]
    assert episode["loss_eligible"] is terminated


def test_cancel_deadline_no_policy_or_tools():
    policy = FakePolicy([])
    with pytest.raises(TimeoutError):
        rollout_policy(policy, FakeEnvironment, {"id": "x"}, seed=1, max_turns=2,
            max_new_tokens=2, max_context_tokens=10, deadline=time.monotonic()-1)
    assert not policy.calls


def test_b0_guard_does_not_fabricate_final_answer():
    action = {"tool": "finalize", "arguments": {"answerability": "answerable",
              "scope": "authorized_snapshot_only", "claims": [{"evidence_id": "id1", "quote": "guess"}], "measurements": []}}
    result = guard_action(action, initial_messages({"question": "x"}))
    assert result["intervened"]
    assert result["action"] == {"tool": "read_passage", "arguments": {"evidence_id": "id1"}}
    assert action["tool"] == "finalize"  # original preserved for audit


def test_b0_fills_only_missing_singleton_required_enum():
    messages = initial_messages({"question": "x"})
    messages.append({"role": "user", "content": canonical_json({"tool": "list_documents", "observation": {"complete": True}})})
    action = {"tool": "finalize", "arguments": {"answerability": "insufficient", "claims": [], "measurements": []}}
    result = guard_action(action, messages)
    assert result["reason"] == "required_singleton_enum_filled"
    assert result["action"]["arguments"]["scope"] == "authorized_snapshot_only"
    assert result["filled_fields"] == ["scope"]
    assert "scope" not in action["arguments"]
    action["arguments"]["scope"] = "invalid_explicit_scope"
    assert guard_action(action, messages)["action"]["tool"] == "list_documents"
    action["arguments"].pop("answerability")  # More than one enum value: never guess.
    action["arguments"].pop("scope")
    assert guard_action(action, messages)["reason"] == "invalid_action_recover_catalog"


def test_pair_membership_and_seed_reproducibility():
    tasks = [{"id": str(i), "pair_id": str(i//2), "side": str(i%2)} for i in range(8)]
    pairs = paired_tasks(tasks)
    for arm in ("B1", "B2", "B3", "LOO"):
        one, two = random.Random(97), random.Random(97)
        assert [select_training_groups(tasks, pairs, arm, one) for _ in range(10)] == [
            select_training_groups(tasks, pairs, arm, two) for _ in range(10)]
    with pytest.raises(ValueError, match="exactly two"):
        paired_tasks(tasks[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_tasks(tasks+[tasks[0]])


def test_gate_blocks_and_exact_bindings(tmp_path):
    gate_path = tmp_path / "SFT_GATE.json"
    expected = {"base_snapshot_sha256": "base", "initial_adapter_sha256": "adapter"}
    with pytest.raises(FileNotFoundError):
        validate_sft_gate(gate_path, expected)
    assert validate_sft_gate(gate_path, expected, engineering_smoke=True)["formal_training"] is False
    report = tmp_path / "audit.json"
    report.write_text('{"independent_audit":true}')
    gate = {"schema": "mito.rp-grpo.sft-gate.v1", "passed": True, "bindings": expected,
            "evaluation_report_path": str(report), "evaluation_report_sha256": sha256_file(report)}
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):  # Arbitrary hash-bound documents cannot unlock RL.
        validate_sft_gate(gate_path, expected)
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_sft_gate(gate_path, {**expected, "initial_adapter_sha256": "new"})
    report.write_text("changed")
    with pytest.raises(ValueError, match="report hash"):
        validate_sft_gate(gate_path, expected)


def test_train_loader_refuses_test_and_duplicate_ids(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"id":"x","split":"train"}\n{"id":"y","split":"dev"}\n')
    assert [t["id"] for t in load_tasks(path, "train")] == ["x"]
    with pytest.raises(ValueError, match="frozen test"):
        load_tasks(path, "test")
    path.write_text('{"id":"x","split":"train"}\n{"id":"x","split":"dev"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(path, "train")


@pytest.mark.parametrize("text", ['{"tool":"x","arguments":{"v":NaN}}',
    '{"tool":"x","arguments":{"v":1e999}}', '[{"tool":"x","arguments":{}}]',
    '```json\n{"tool":"x","arguments":{}}\n```'])
def test_action_parser_strict(text):
    with pytest.raises(ValueError):
        parse_action(text)
