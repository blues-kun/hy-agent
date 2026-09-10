"""Portable, local-only HF/PEFT tool-policy evaluation and bounded RL.

Default is validation without loading weights. Formal training needs a passed,
hash-bound SFT_GATE.json. Engineering smoke is explicit and never production
admission. This is a small single-GPU reference implementation, not a claim of
distributed rollout throughput or proven scientific/RP improvement.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rp_grpo.algorithms import (finite, original_grpo_advantages,
                                rp_grpo_advantages, token_policy_loss)

BUNDLE = Path(__file__).resolve().parents[1]
SCHEMA = "mito.rp-grpo.policy-run.v1"
PROMPT_VERSION = "mito.rp-grpo.json-tools.v1"
ARMS = {"B0": "sft_rules_evaluation", "B1": "original_grpo_unpaired",
        "B2": "original_grpo_paired", "B3": "rp_fixed_loo_paired",
        "LOO": "eta0_fixed_loo_paired"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value: str) -> Any:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def bad(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    result = json.loads(value, object_pairs_hook=pairs, parse_constant=bad)
    canonical_json(result)  # catches float overflow, e.g. 1e999
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()


def verifier_module():
    name = "mito_portable_verifier_helpers"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BUNDLE / "code" / "train_verifier.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def model_snapshot(path: Path) -> dict:
    return verifier_module().model_snapshot(Path(path))


def adapter_snapshot(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).resolve(strict=True)
    names = ("adapter_config.json", "adapter_model.safetensors")
    files = {}
    for name in names:
        item = (path / name).resolve(strict=True)
        if not item.is_relative_to(path):
            raise ValueError("Adapter files must remain inside their snapshot")
        files[name] = sha256_file(item)
    config = strict_json((path / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Only local causal-LM LoRA adapters are supported")
    return {"path": str(path), "files_sha256": files, "snapshot_sha256": canonical_hash(files)}


def build_system_prompt() -> str:
    from rp_grpo.environment import SYSTEM_PROMPT, TOOL_SCHEMAS
    return (SYSTEM_PROMPT + "\n\nTool schema:\n" + canonical_json(TOOL_SCHEMAS)
            + '\nReturn exactly one JSON object with keys "tool" and "arguments".'
              ' No markdown or hidden reasoning.')


def initial_messages(observation: dict) -> list[dict]:
    if not isinstance(observation, dict):
        raise ValueError("reset must return an actor-visible observation object")
    return [{"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": canonical_json({"observation": observation})}]


def append_observation(messages: list[dict], action_text: str, action: dict, result: dict) -> None:
    # Deliberately excludes reward, info, scorer IDs and hidden target labels.
    messages.extend([{"role": "assistant", "content": action_text},
                     {"role": "user", "content": canonical_json(
                         {"tool": action["tool"], "observation": result["observation"]})}])


def format_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def prompt_contract(snapshot: dict) -> dict:
    files = snapshot["files_sha256"]
    contract = {"version": PROMPT_VERSION, "system": build_system_prompt(),
                "initial_user": {"observation": "reset actor observation"},
                "subsequent_user": {"tool": "actual tool name", "observation": "actual tool observation"},
                "chat_template": {"add_generation_prompt": True, "enable_thinking": False,
                                  "native_tools_argument": False},
                "tokenizer_files_sha256": {k: v for k, v in files.items()
                    if "token" in k or "template" in k or k in {"vocab.json", "merges.txt"}}}
    return {"contract": contract, "sha256": canonical_hash(contract)}


def parse_action(text: str) -> dict:
    action = strict_json(text)
    if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        raise ValueError("Expected exactly {tool, arguments}; incomplete JSON never executes")
    if not isinstance(action["tool"], str) or not action["tool"].strip() or not isinstance(action["arguments"], dict):
        raise ValueError("Tool must be a name and arguments must be an object")
    return action


def guard_action(action: dict | None, messages: list[dict]) -> dict:
    """B0-only deterministic correction, using actor-visible bytes only.

    This is deliberately not the environment's reference solution and never
    inspects a scorer, task side, answerability label or private view table.
    All arms still share the environment's authorization/validation checks.
    """
    from rp_grpo.environment import TOOL_SCHEMAS, _validate
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}
    replacement, reason = copy.deepcopy(action), None
    filled = []
    try:
        if action is None or action["tool"] not in specs:
            raise ValueError("unknown action")
        schema = specs[action["tool"]]
        for name in schema.get("required", []):
            allowed = schema.get("properties", {}).get(name, {}).get("enum", [])
            if name not in replacement["arguments"] and len(allowed) == 1:
                replacement["arguments"][name] = copy.deepcopy(allowed[0])
                filled.append(name)
        _validate(replacement["arguments"], schema)
    except (ValueError, TypeError, KeyError):
        replacement, reason = {"tool": "list_documents", "arguments": {"limit": 100}}, "invalid_action_recover_catalog"
    if reason is None:
        reads, catalog_complete = {}, False
        for message in messages:
            if message.get("role") != "user":
                continue
            try:
                payload = strict_json(message["content"])
            except ValueError:
                continue
            obs = payload.get("observation", {})
            if payload.get("tool") == "list_documents" and obs.get("complete") is True:
                catalog_complete = True
            if payload.get("tool") == "read_passage" and obs.get("status") == "ok":
                reads[obs.get("evidence_id")] = obs
        args = replacement["arguments"]
        if replacement["tool"] in {"finalize", "stop"}:
            if replacement["tool"] == "finalize" and args["answerability"] == "answerable":
                for claim in args["claims"]:
                    if claim["evidence_id"] not in reads or claim["quote"] != reads[claim["evidence_id"]].get("text"):
                        replacement = {"tool": "read_passage", "arguments": {"evidence_id": claim["evidence_id"]}}
                        reason = "source_must_be_read_and_quote_bound"
                        break
            elif not catalog_complete:
                replacement, reason = {"tool": "list_documents", "arguments": {"limit": 100}}, "inspect_scope_before_abstention"
    if reason is None and filled:
        reason = "required_singleton_enum_filled"
    result = {"action": replacement, "intervened": reason is not None, "reason": reason}
    if filled:
        result["filled_fields"] = filled
    return result


def load_tasks(path: Path, split: str) -> list[dict]:
    if split not in {"train", "dev"}:
        raise ValueError("This training launcher never consumes frozen test tasks")
    tasks, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = strict_json(line)
        if not isinstance(task, dict):
            raise ValueError("Task rows must be objects")
        task_id = task.get("id", task.get("task_id"))
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("Missing or duplicate task ID")
        seen.add(task_id)
        if task.get("split") == split:
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks in explicitly requested split {split}")
    return tasks


def task_id(task: dict) -> str:
    return task.get("id", task.get("task_id"))


def paired_tasks(tasks: list[dict]) -> list[list[dict]]:
    pairs = {}
    for task in tasks:
        pair, side = task.get("pair_id"), task.get("side")
        if not isinstance(pair, str) or not pair or not isinstance(side, str) or not side:
            raise ValueError("Paired arms require explicit nonempty pair_id and side")
        group = pairs.setdefault(pair, {})
        if side in group:
            raise ValueError("Duplicate environment side in pair")
        group[side] = task
    if any(len(group) != 2 for group in pairs.values()):
        raise ValueError("Every pair must have exactly two complete sides in the same split")
    return [list(group.values()) for group in pairs.values()]


def validate_sft_gate(path: Path, expected: dict, *, engineering_smoke: bool = False) -> dict:
    from rp_grpo.gate_validation import validate_sft_gate as validate
    return validate(path, expected, engineering_smoke=engineering_smoke, bundle=BUNDLE)


def selected_logprobs(model, prompt_ids: list[int], completion_ids: list[int]):
    """Teacher-forced action likelihoods, excluding every prompt/tool token."""
    import torch
    if not prompt_ids or not completion_ids:
        raise ValueError("Empty prompt or policy action")
    ids = torch.tensor([prompt_ids + completion_ids], device=model.device, dtype=torch.long)
    # Pinned Transformers/Qwen supports logits_to_keep: avoid materializing
    # float32 log-softmax for the entire long observation history.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(completion_ids) + 1).logits[0, -len(completion_ids)-1:-1]
    target = ids[0, -len(completion_ids):]
    return -torch.nn.functional.cross_entropy(logits.float(), target, reduction="none")


class HFPolicy:
    """Actual autoregressive local policy; no fixed/replayed answer generation."""
    def __init__(self, model, tokenizer, *, reference_adapter: str | None = None):
        self.model, self.tokenizer, self.reference_adapter = model, tokenizer, reference_adapter

    def sample(self, messages: list[dict], *, seed: int, max_new_tokens: int,
               max_context_tokens: int, deadline: float, sample: bool, with_logprobs: bool) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        prompt = format_prompt(self.tokenizer, messages)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) + max_new_tokens > max_context_tokens:
            return {"text": "", "prompt_ids": ids, "completion_ids": [],
                    "terminated": False, "reason": "context_budget", "prompt_sha256": canonical_hash(prompt)}
        if time.monotonic() >= deadline:
            raise TimeoutError("Run wall-clock deadline reached")

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        self.model.eval()  # dropout-free sampling AND subsequent policy likelihoods
        eos = self.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("Tokenizer must define an integer assistant end token")
        tensor = torch.tensor([ids], device=self.model.device)
        devices = [self.model.device.index] if self.model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            kwargs = {"do_sample": sample, "max_new_tokens": max_new_tokens,
                      "eos_token_id": eos, "pad_token_id": eos, "use_cache": True,
                      "repetition_penalty": 1.0, "stopping_criteria": StoppingCriteriaList([Deadline()])}
            if sample:
                kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
            output = self.model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), **kwargs)
        completion = output[0, len(ids):].tolist()
        terminated = bool(completion and completion[-1] == eos)
        record = {"text": self.tokenizer.decode(completion, skip_special_tokens=True),
                  "prompt_ids": ids, "completion_ids": completion, "terminated": terminated,
                  "reason": "eos" if terminated else "generation_truncated",
                  "prompt_sha256": canonical_hash(prompt)}
        if with_logprobs and completion:
            with torch.no_grad():
                record["old_logprobs"] = selected_logprobs(self.model, ids, completion).cpu().tolist()
                if self.reference_adapter:
                    try:
                        self.model.set_adapter(self.reference_adapter)
                        record["reference_logprobs"] = selected_logprobs(self.model, ids, completion).cpu().tolist()
                    finally:
                        self.model.set_adapter("policy")
                else:
                    with self.model.disable_adapter():
                        record["reference_logprobs"] = selected_logprobs(self.model, ids, completion).cpu().tolist()
        return record


def rollout_policy(policy, environment_factory, task: dict, *, seed: int,
                   max_turns: int, max_new_tokens: int, max_context_tokens: int,
                   deadline: float, sample: bool = True, with_logprobs: bool = False,
                   rules: bool = False) -> dict:
    env = environment_factory()
    observation = env.reset(copy.deepcopy(task))
    messages = initial_messages(observation)
    episode = {"task_id": task_id(task), "pair_id": task.get("pair_id"), "side": task.get("side"),
               "split": task.get("split"), "seed": seed, "initial_observation": observation,
               "initial_prompt_sha256": canonical_hash(messages), "steps": [], "reward": -1.0,
               "done": False, "loss_eligible": True, "termination_reason": "turn_budget"}
    for turn in range(max_turns):
        if time.monotonic() >= deadline:
            raise TimeoutError("Run deadline: no further policy or tool request")
        generated = policy.sample(messages, seed=seed + turn, max_new_tokens=max_new_tokens,
                                  max_context_tokens=max_context_tokens, deadline=deadline,
                                  sample=sample, with_logprobs=with_logprobs)
        step = {"turn": turn, "messages": copy.deepcopy(messages), "raw_text": generated["text"],
                "prompt_sha256": generated["prompt_sha256"],
                "prompt_tokens": len(generated["prompt_ids"]),
                "completion_tokens": len(generated["completion_ids"]),
                "terminated": generated["terminated"], "tokens": generated}
        episode["steps"].append(step)
        if not generated["terminated"]:
            episode.update(reward=-0.25, loss_eligible=False, termination_reason=generated["reason"])
            break  # Never parse or execute an incomplete action.
        try:
            action = parse_action(generated["text"])
        except (ValueError, TypeError) as exc:
            step["action_error"] = str(exc)
            if not rules:
                episode.update(reward=-1.0, termination_reason="invalid_action_json")
                break
            action = None
        if rules:
            step["model_original_action"] = copy.deepcopy(action)
            intervention = guard_action(action, messages)
            step["rule_intervention"] = intervention
            action = intervention["action"]
        if time.monotonic() >= deadline:
            raise TimeoutError("Run deadline: parsed action not executed")
        result = env.step(action)
        if not isinstance(result, dict) or not isinstance(result.get("observation"), dict) or not isinstance(result.get("done"), bool):
            raise ValueError("Environment must return {observation:object,done:bool,reward,info}")
        step.update(action=action, result=copy.deepcopy(result))
        if result["done"]:
            reward = finite(result.get("reward"), "terminal reward")
            if not -1 <= reward <= 1:
                raise ValueError("Environment reward violates frozen [-1,1] contract")
            episode.update(done=True, reward=reward, termination_reason="environment_terminal")
            break
        append_observation(messages, generated["text"], action, result)
    if hasattr(env, "export_trace"):
        episode["environment_trace"] = env.export_trace()
    return episode


def public_episode(episode: dict) -> dict:
    result = copy.deepcopy(episode)
    for step in result["steps"]:
        tokens = step.pop("tokens")
        step["policy_tokens_sha256"] = canonical_hash(tokens["completion_ids"])
        step["prompt_ids"] = tokens["prompt_ids"]
        step["completion_ids"] = tokens["completion_ids"]
        if "old_logprobs" in tokens:
            step["behavior_logprobs"] = tokens["old_logprobs"]
            step["reference_logprobs"] = tokens["reference_logprobs"]
    return result


def select_training_groups(tasks: list[dict], pairs: list[list[dict]], arm: str,
                           rng: random.Random) -> list[dict]:
    if arm == "B1":
        return [rng.choice(tasks), rng.choice(tasks)]  # two independent prompts, G each
    return list(rng.choice(pairs))


def configure_gradient_checkpointing(model) -> dict:
    """Bound activation memory without introducing sampling/update dropout."""
    config = model.config
    if float(getattr(config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Nonzero functional attention dropout changes behavior log-probabilities")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return {"enabled": True, "use_reentrant": False, "input_require_grads": True,
            "sampling_mode": "eval", "update_mode": "train_with_all_Dropout_eval",
            "attention_dropout": 0.0,
            "memory_limit": "Checkpointing reduces activations; this is not a tested 4B/8k VRAM guarantee."}


def update_from_episodes(policy: HFPolicy, optimizer, groups: list[list[dict]], *, arm: str,
                         eta: float, epsilon: float, beta: float, deadline: float = math.inf) -> dict:
    import torch
    if len(groups) != 2 or len(groups[0]) != len(groups[1]):
        raise ValueError("Every update requires two equal-sized rollout groups")
    returns = {str(side): [e["reward"] for e in group] for side, group in enumerate(groups)}
    if arm in {"B1", "B2"}:
        advantages = {side: original_grpo_advantages(values) for side, values in returns.items()}
        audit = {"normalization": "within_side_population_std_eps1e-8", "eta": None}
    elif arm in {"B3", "LOO"}:
        audit = rp_grpo_advantages(returns, eta=0.0 if arm == "LOO" else eta)
        advantages = audit["advantages"]
    else:
        raise ValueError("B0 is an evaluation-only baseline")
    optimizer.zero_grad(set_to_none=True)
    policy.model.train()  # Transformers activation checkpointing requires training mode.
    for module in policy.model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()  # Preserve the same dropout-free behavior/reference distribution.
    count, loss_sum, trained = sum(map(len, groups)), 0.0, 0
    for side, group in enumerate(groups):
        for index, episode in enumerate(group):
            if not episode["loss_eligible"]:
                continue
            total = sum(len(step["tokens"]["completion_ids"]) for step in episode["steps"])
            if not total:
                continue
            for step in episode["steps"]:
                if time.monotonic() >= deadline:
                    optimizer.zero_grad(set_to_none=True)
                    raise TimeoutError("Deadline during gradient accumulation; no optimizer update")
                tokens = step["tokens"]
                current = selected_logprobs(policy.model, tokens["prompt_ids"], tokens["completion_ids"])
                old = torch.tensor(tokens["old_logprobs"], device=current.device)
                reference = torch.tensor(tokens["reference_logprobs"], device=current.device)
                loss = token_policy_loss(current, old, reference, torch.ones_like(current),
                                         advantages[str(side)][index], epsilon=epsilon, beta=beta)
                # All arms: mean over policy tokens within episode, then over
                # 2G actual episodes. Tool/prompt tokens never contribute.
                weighted = loss * (len(tokens["completion_ids"]) / total / count)
                weighted.backward()
                loss_sum += float(weighted.detach())
            trained += 1
    if time.monotonic() >= deadline:
        optimizer.zero_grad(set_to_none=True)
        raise TimeoutError("Deadline before optimizer step")
    if trained:
        norm = torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad], 1.0,
                                            error_if_nonfinite=True)
        optimizer.step()
    else:
        norm = torch.tensor(0.0)
    return {"loss": loss_sum, "gradient_norm": float(norm), "updated_episodes": trained,
            "real_rollouts": count, "rewards": returns, "advantage_audit": audit,
            "flat_groups": sum(len(set(v)) == 1 for v in returns.values())}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("validate", "train", "evaluate"), default="validate")
    p.add_argument("--arm", choices=tuple(ARMS), default="B3")
    p.add_argument("--train-data", type=Path, default=BUNDLE / "rp_grpo/data/tasks.jsonl")
    p.add_argument("--scorers", type=Path, default=BUNDLE / "rp_grpo/data/scorers.jsonl")
    p.add_argument("--model-path", type=Path, default=BUNDLE / "models/Qwen3-4B-Instruct-2507")
    p.add_argument("--initial-adapter", type=Path)
    p.add_argument("--sft-gate", type=Path, default=BUNDLE / "rp_grpo/SFT_GATE.json")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--split", choices=("train", "dev"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--engineering-smoke", action="store_true")
    p.add_argument("--eval-sample", action="store_true", help="Stochastic evaluation; greedy by default")
    p.add_argument("--gpu", type=int)
    for name, default in (("group-size", 4), ("max-steps", 12), ("max-seconds", 1800),
                          ("max-turns", 6), ("max-new-tokens", 512), ("max-context-tokens", 8192),
                          ("save-steps", 4), ("max-eval-tasks", 20), ("eval-rollouts", 1)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--seed", type=int, default=20260909)
    for name, default in (("eta", 0.5), ("learning-rate", 1e-6), ("beta", 0.04), ("clip-epsilon", 0.2)):
        p.add_argument("--" + name, type=float, default=default)
    return p


def prepare(args) -> tuple[list[dict], dict]:
    for key in ("group_size", "max_steps", "max_seconds", "max_turns", "max_new_tokens",
                "max_context_tokens", "save_steps", "max_eval_tasks", "eval_rollouts"):
        if isinstance(getattr(args, key), bool) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.group_size < 2 or not 0 <= finite(args.eta, "eta") <= 1:
        raise ValueError("G>=2 and eta in [0,1] are required")
    if finite(args.learning_rate, "learning_rate") <= 0 or finite(args.beta, "beta") < 0 or not 0 < finite(args.clip_epsilon, "clip_epsilon") < 1:
        raise ValueError("Invalid optimizer configuration")
    split = args.split or ("dev" if args.mode == "evaluate" else "train")
    if args.mode == "train" and (split != "train" or args.arm == "B0"):
        raise ValueError("Training requires split=train and a non-B0 arm")
    tasks = load_tasks(args.train_data.resolve(strict=True), split)
    if args.train_data.name != "tasks.jsonl" or args.scorers.resolve() != args.train_data.resolve().parent / "scorers.jsonl":
        raise ValueError("Use the manifest-bound tasks.jsonl/scorers.jsonl in the same data directory")
    from rp_grpo.environment import ResearchEnvironment
    # CPU only: verifies every data-manifest hash before any model execution.
    ResearchEnvironment(data_dir=args.train_data.parent, max_steps=args.max_turns)
    if args.mode != "evaluate":
        paired_tasks(tasks)
    snapshot = model_snapshot(args.model_path)
    adapter = adapter_snapshot(args.initial_adapter)
    contract = prompt_contract(snapshot)
    code_files = {str(path.relative_to(BUNDLE)): sha256_file(path) for path in
                  sorted((BUNDLE / "rp_grpo").glob("*.py"))}
    bindings = {"base_snapshot_sha256": snapshot["snapshot_sha256"],
                "initial_adapter_sha256": adapter["snapshot_sha256"] if adapter else None,
                "training_data_sha256": sha256_file(args.train_data),
                "scorers_sha256": sha256_file(args.scorers),
                "data_manifest_sha256": sha256_file(args.train_data.parent / "MANIFEST.json"),
                "environment_sha256": sha256_file(BUNDLE / "rp_grpo/environment.py"),
                "prompt_contract_sha256": contract["sha256"]}
    if args.mode == "train":
        if not adapter and not args.engineering_smoke:
            raise ValueError("Formal RL requires the exact admitted SFT adapter")
        gate = validate_sft_gate(args.sft_gate, bindings, engineering_smoke=args.engineering_smoke)
    else:
        gate = {"passed": False, "formal_training": False, "reason": "not_a_training_invocation"}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return tasks, {"schema": SCHEMA, "status": "validated", "created_at": now(),
                   "record_source": "not_executed_validation_only",
                   "mode": args.mode, "arm": args.arm, "arm_definition": ARMS[args.arm],
                   "split": split, "bindings": bindings, "model_snapshot": snapshot,
                   "adapter_snapshot": adapter, "prompt_contract": contract,
                   "config": config, "config_sha256": canonical_hash(config),
                   "code_files_sha256": code_files, "code_sha256": canonical_hash(code_files),
                   "gate": gate, "production_ready": False, "new_expert_approval": False,
                   "algorithm_improvement_proven": False, "external_api_calls": 0,
                   "formal_training": gate["formal_training"],
                   "objective": "paired_reward_mean_minus_eta_half_absolute_difference",
                   "loss_reduction": "mean_policy_tokens_per_episode_then_mean_2G_episodes",
                   "rollout_sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
                   "task_count": len(tasks)}


def run(args) -> dict:
    tasks, manifest = prepare(args)
    if args.mode == "validate" or not args.execute:
        return manifest
    if args.output_dir is None or args.gpu is None:
        raise ValueError("Execution needs a new output directory and explicit GPU")
    output = args.output_dir.resolve()
    if output.exists() or output in {Path(output.anchor), Path.home(), BUNDLE, Path.cwd()}:
        raise ValueError("Output must be a new dedicated run directory")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false")
    # A single selected GPU is locked/checked before importing torch. Never
    # stop, move or reuse another job; the launcher cannot claim global lock.
    with verifier_module().gpu_lease(args.gpu) as device:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, LoraConfig, get_peft_model
        from rp_grpo.environment import ResearchEnvironment
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("One isolated CUDA device is required")
        output.mkdir(parents=True, exist_ok=False)
        manifest.update(status="loading", gpu=device, started_at=now(), pid=os.getpid(),
                        installed_versions=verifier_module().installed_versions())
        write_json(output / "run_manifest.json", manifest)
        started = time.monotonic()
        deadline = started + args.max_seconds
        completed_steps = 0
        try:
            torch.manual_seed(args.seed)
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True,
                                                      trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
            reference = None
            if args.initial_adapter:
                model = PeftModel.from_pretrained(model, str(args.initial_adapter), adapter_name="policy",
                                                 is_trainable=args.mode == "train")
                if args.mode == "train":
                    model.load_adapter(str(args.initial_adapter), adapter_name="reference", is_trainable=False)
                    model.set_adapter("policy")
                    reference = "reference"
            elif args.mode == "train":
                model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    task_type="CAUSAL_LM"), adapter_name="policy")
            manifest["gradient_checkpointing"] = (configure_gradient_checkpointing(model)
                if args.mode == "train" else {"enabled": False, "reason": "inference_only"})
            policy = HFPolicy(model, tokenizer, reference_adapter=reference)
            environment_factory = lambda: ResearchEnvironment(data_dir=args.train_data.parent, max_steps=args.max_turns)
            manifest.update(status="evaluating" if args.mode == "evaluate" else "training",
                            record_source="actual_hf_model",
                            generation_pad_token_id=tokenizer.eos_token_id,
                            generation_eos_token_id=tokenizer.eos_token_id)
            write_json(output / "run_manifest.json", manifest)
            if args.mode == "evaluate":
                for index, task in enumerate(tasks[:args.max_eval_tasks]):
                    for repeat in range(args.eval_rollouts):
                        episode = rollout_policy(policy, environment_factory, task,
                            seed=args.seed + index * 10000 + repeat * 100,
                            max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                            max_context_tokens=args.max_context_tokens, deadline=deadline,
                            sample=args.eval_sample, with_logprobs=False, rules=args.arm == "B0")
                        append_json(output / "episodes.jsonl", public_episode(episode))
                        completed_steps += 1
            else:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if not trainable:
                    raise ValueError("No trainable policy parameters")
                optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
                pairs = paired_tasks(tasks)
                rng = random.Random(args.seed)
                for update in range(args.max_steps):
                    selected = select_training_groups(tasks, pairs, args.arm, rng)
                    groups = []
                    for side_index, task in enumerate(selected):
                        group = []
                        for sample_index in range(args.group_size):
                            episode = rollout_policy(policy, environment_factory, task,
                                seed=args.seed + update * 100000 + side_index * 10000 + sample_index * 100,
                                max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                                max_context_tokens=args.max_context_tokens, deadline=deadline,
                                sample=True, with_logprobs=True)
                            group.append(episode)
                            append_json(output / "episodes.jsonl", {"update": update, "group_side": side_index,
                                                                   **public_episode(episode)})
                        groups.append(group)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Deadline before optimizer update")
                    metrics = update_from_episodes(policy, optimizer, groups, arm=args.arm, eta=args.eta,
                                                  epsilon=args.clip_epsilon, beta=args.beta, deadline=deadline)
                    completed_steps += 1
                    append_json(output / "metrics.jsonl", {"update": update, "at": now(), **metrics})
                    if completed_steps % args.save_steps == 0:
                        checkpoint = output / f"checkpoint-{completed_steps}"
                        model.save_pretrained(checkpoint, selected_adapters=["policy"], safe_serialization=True)
                        write_json(checkpoint / "CHECKPOINT.json", {"step": completed_steps,
                            "bindings": manifest["bindings"], "config_sha256": manifest["config_sha256"],
                            "code_sha256": manifest["code_sha256"],
                            "adapter_snapshot": adapter_snapshot(checkpoint / "policy"),
                            "continuation_only_not_optimizer_resume": True,
                            "production_ready": False})
                model.save_pretrained(output / "adapter", selected_adapters=["policy"], safe_serialization=True)
                tokenizer.save_pretrained(output / "adapter")
            manifest.update(status="completed", completed_steps=completed_steps, ended_at=now(),
                            elapsed_seconds=time.monotonic()-started,
                            episodes_sha256=sha256_file(output / "episodes.jsonl"))
            if (output / "adapter/policy/adapter_model.safetensors").is_file():
                manifest["output_adapter"] = adapter_snapshot(output / "adapter/policy")
            write_json(output / "run_manifest.json", manifest)
            return manifest
        except BaseException as exc:
            manifest.update(status="incomplete" if isinstance(exc, TimeoutError) else "failed",
                            ended_at=now(), completed_steps=completed_steps,
                            error_type=type(exc).__name__, error=str(exc),
                            elapsed_seconds=time.monotonic()-started)
            write_json(output / "run_manifest.json", manifest)
            raise


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
